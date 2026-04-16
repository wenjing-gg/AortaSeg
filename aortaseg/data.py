from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


TARGET_BLOCK_SHAPE = (64, 64, 64)


def _pad_block(block: np.ndarray, fill_value: float) -> np.ndarray:
    if block.shape == TARGET_BLOCK_SHAPE:
        return block
    if block.ndim != 3:
        raise ValueError(f"Expected 3D block, got {block.shape}")
    if any(cur > tgt for cur, tgt in zip(block.shape, TARGET_BLOCK_SHAPE)):
        raise ValueError(f"Block shape {block.shape} exceeds target shape {TARGET_BLOCK_SHAPE}")
    pad_width = tuple((0, tgt - cur) for cur, tgt in zip(block.shape, TARGET_BLOCK_SHAPE))
    return np.pad(block, pad_width, mode="constant", constant_values=fill_value)


def _load_spacing_map(excel_path: Optional[str]) -> Dict[int, Tuple[float, float, float]]:
    if not excel_path or not os.path.isfile(excel_path):
        return {}
    spacing_map: Dict[int, Tuple[float, float, float]] = {}
    try:
        import pandas as pd

        dataframe = pd.read_excel(excel_path)
    except Exception as exc:
        print(f"[Warning] Failed to load spacing file {excel_path}: {exc}")
        return spacing_map
    dataframe.columns = [str(col).strip().lower() for col in dataframe.columns]
    required = {"patient_index", "xy_pixel_spacing", "z_pixel_spacing"}
    if not required.issubset(set(dataframe.columns)):
        return spacing_map

    def parse_float(value) -> float:
        try:
            return float(value)
        except Exception:
            matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
            if not matches:
                raise
            return float(matches[0])

    for _, row in dataframe.iterrows():
        try:
            patient_index = int(row["patient_index"])
            xy = parse_float(row["xy_pixel_spacing"])
            z = parse_float(row["z_pixel_spacing"])
        except Exception:
            continue
        spacing_map[patient_index] = (float(z), float(xy), float(xy))
    return spacing_map


class CTABlockDataset3D(Dataset):
    def __init__(
        self,
        data_dir: str,
        phase: str = "train",
        window_level: float = 1024.0,
        window_width: float = 4096.0,
        normalize: bool = True,
        if_flt: bool = True,
        augmentation: bool = True,
        excel_path: Optional[str] = None,
        block_list: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        self.data_dir = data_dir
        self.phase = phase
        self.window_level = float(window_level)
        self.window_width = float(window_width)
        self.normalize = bool(normalize)
        self.if_flt = bool(if_flt)
        self.augmentation = bool(augmentation) and phase == "train"
        self.excel_path = excel_path
        self.a_min = self.window_level - self.window_width / 2.0
        self.a_max = self.window_level + self.window_width / 2.0
        self.spacing_map = _load_spacing_map(excel_path)
        self.block_list = list(block_list) if block_list is not None else self._collect_blocks()
        if not self.block_list:
            raise ValueError(f"No valid blocks found in {data_dir}")

    def _collect_blocks(self) -> List[Dict[str, object]]:
        block_list: List[Dict[str, object]] = []
        for npz_path in sorted(glob.glob(os.path.join(self.data_dir, "*_blocks.npz"))):
            try:
                with np.load(npz_path, allow_pickle=True) as data:
                    case_id = str(data["case_id"]) if "case_id" in data else os.path.basename(npz_path).replace("_blocks.npz", "")
                    num_blocks = int(data["blocks_img"].shape[0])
            except Exception as exc:
                print(f"[Warning] Failed to load {npz_path}: {exc}")
                continue
            for block_idx in range(num_blocks):
                block_list.append(
                    {
                        "npz_path": npz_path,
                        "case_id": case_id,
                        "block_idx": block_idx,
                    }
                )
        return block_list

    def subset(self, block_list: List[Dict[str, object]], phase: str, augmentation: bool) -> "CTABlockDataset3D":
        return CTABlockDataset3D(
            data_dir=self.data_dir,
            phase=phase,
            window_level=self.window_level,
            window_width=self.window_width,
            normalize=self.normalize,
            if_flt=self.if_flt,
            augmentation=augmentation,
            excel_path=self.excel_path,
            block_list=block_list,
        )

    def _get_spacing(self, case_id: str) -> Tuple[float, float, float]:
        try:
            patient_index = int(str(case_id).split("_")[0])
        except Exception:
            return (1.0, 1.0, 1.0)
        return self.spacing_map.get(patient_index, (1.0, 1.0, 1.0))

    def _apply_window(self, image: np.ndarray) -> np.ndarray:
        image = np.clip(image, self.a_min, self.a_max)
        return (image - self.a_min) / (self.a_max - self.a_min)

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        nonzero = image > 0
        if np.any(nonzero):
            mean = float(np.mean(image[nonzero]))
            std = float(np.std(image[nonzero]))
            if std > 1e-8:
                image = image.copy()
                image[nonzero] = (image[nonzero] - mean) / std
        return image

    def _clean_label(self, label: np.ndarray) -> np.ndarray:
        if self.if_flt:
            return label
        return label * np.isin(label, (1, 2)).astype(label.dtype)

    def _augment(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.augmentation:
            return image, label
        aug_image = image.copy()
        aug_label = label.copy()
        if np.random.rand() < 0.5:
            k = np.random.randint(1, 4)
            aug_image = np.rot90(aug_image, k=k, axes=(1, 2)).copy()
            aug_label = np.rot90(aug_label, k=k, axes=(1, 2)).copy()
        if np.random.rand() < 0.5:
            aug_image = np.flip(aug_image, axis=1).copy()
            aug_label = np.flip(aug_label, axis=1).copy()
        if np.random.rand() < 0.5:
            aug_image = np.flip(aug_image, axis=2).copy()
            aug_label = np.flip(aug_label, axis=2).copy()
        if np.random.rand() < 0.4:
            aug_image = aug_image + np.random.normal(loc=0.0, scale=0.08, size=aug_image.shape).astype(np.float32)
        if np.random.rand() < 0.4:
            gamma = float(np.random.uniform(0.75, 1.25))
            min_val = float(aug_image.min())
            max_val = float(aug_image.max())
            if max_val > min_val:
                normalized = (aug_image - min_val) / (max_val - min_val)
                aug_image = np.power(np.clip(normalized, 0.0, 1.0), gamma).astype(np.float32)
                aug_image = aug_image * (max_val - min_val) + min_val
        if np.random.rand() < 0.4:
            aug_image = aug_image + float(np.random.uniform(-0.15, 0.15))
        return aug_image.astype(np.float32), aug_label.astype(np.int64)

    def __len__(self) -> int:
        return len(self.block_list)

    def __getitem__(self, idx: int):
        block_info = self.block_list[idx]
        npz_path = str(block_info["npz_path"])
        block_idx = int(block_info["block_idx"])
        case_id = str(block_info["case_id"])
        with np.load(npz_path, allow_pickle=True) as data:
            block_img = data["blocks_img"][block_idx].astype(np.float32)
            block_label = data["blocks_label"][block_idx].astype(np.int64)
        block_img = self._apply_window(block_img)
        if self.normalize:
            block_img = self._normalize_image(block_img)
        block_label = self._clean_label(block_label)
        block_img = _pad_block(block_img, 0.0)
        block_label = _pad_block(block_label, 0)
        if self.augmentation:
            block_img, block_label = self._augment(block_img, block_label)
        image = torch.from_numpy(block_img).float().unsqueeze(0)
        label = torch.from_numpy(block_label).long()
        spacing = torch.tensor(self._get_spacing(case_id), dtype=torch.float32)
        filename = f"{case_id}_block{block_idx:03d}"
        return image, label, filename, spacing


def get_cta_block_data_loaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    train_split: float = 0.8,
    window_level: float = 1024.0,
    window_width: float = 4096.0,
    normalize: bool = True,
    if_flt: bool = True,
    excel_path: Optional[str] = None,
    return_case_splits: bool = False,
    seed: int = 42,
):
    base_dataset = CTABlockDataset3D(
        data_dir=data_dir,
        phase="train",
        window_level=window_level,
        window_width=window_width,
        normalize=normalize,
        if_flt=if_flt,
        augmentation=True,
        excel_path=excel_path,
    )
    case_ids = sorted({str(item["case_id"]) for item in base_dataset.block_list})
    if not case_ids:
        raise ValueError(f"No cases found in {data_dir}")
    rng = np.random.default_rng(seed)
    rng.shuffle(case_ids)
    if len(case_ids) == 1:
        train_case_ids = set(case_ids)
        val_case_ids = set(case_ids)
    else:
        train_count = int(round(len(case_ids) * float(train_split)))
        train_count = min(max(train_count, 1), len(case_ids) - 1)
        train_case_ids = set(case_ids[:train_count])
        val_case_ids = set(case_ids[train_count:])
    train_blocks = [item for item in base_dataset.block_list if str(item["case_id"]) in train_case_ids]
    val_blocks = [item for item in base_dataset.block_list if str(item["case_id"]) in val_case_ids]
    train_dataset = base_dataset.subset(train_blocks, phase="train", augmentation=True)
    val_dataset = base_dataset.subset(val_blocks, phase="val", augmentation=False)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    dataset_name = "AortaSeg"
    if return_case_splits:
        return train_loader, val_loader, dataset_name, train_case_ids, val_case_ids
    return train_loader, val_loader, dataset_name
