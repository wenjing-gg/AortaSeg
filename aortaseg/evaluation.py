from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .metrics import calculate_all_metrics_3d_multiclass_mm


def _forward(model, images: torch.Tensor, return_all: bool = False):
    try:
        return model(images, return_all=return_all)
    except TypeError:
        return model(images)


def _parse_case_block(filename: str) -> tuple[str, int]:
    parts = filename.rsplit("_block", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected filename format: {filename}")
    return parts[0], int(parts[1])


def _load_case_metadata(data_dir: str | Path, case_id: str) -> Dict[str, Any]:
    npz_path = Path(data_dir) / f"{case_id}_blocks.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        return {
            "ind_block": np.array(data["ind_block"]),
            "original_shape": tuple(int(v) for v in data["original_shape"]),
        }


def _reconstruct_volume(
    blocks: List[Tuple[int, np.ndarray]],
    ind_block: np.ndarray,
    original_shape: Tuple[int, int, int],
) -> np.ndarray:
    volume = np.zeros(original_shape, dtype=np.float32)
    count = np.zeros(original_shape, dtype=np.float32)
    for block_idx, block in blocks:
        sx, ex, sy, ey, sz, ez = map(int, ind_block[block_idx])
        depth, height, width = block.shape
        ex_adj = min(sx + depth, original_shape[0])
        ey_adj = min(sy + height, original_shape[1])
        ez_adj = min(sz + width, original_shape[2])
        cropped = block[: ex_adj - sx, : ey_adj - sy, : ez_adj - sz].astype(np.float32)
        volume[sx:ex_adj, sy:ey_adj, sz:ez_adj] += cropped
        count[sx:ex_adj, sy:ey_adj, sz:ez_adj] += 1.0
    count[count < 0.5] = 1.0
    return volume / count


def _summarize_case_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        raise ValueError("No case metrics to summarize")
    summary: Dict[str, float] = {}
    for key in metrics_list[0]:
        try:
            values = np.asarray([float(item[key]) for item in metrics_list], dtype=np.float64)
        except Exception:
            continue
        summary[key] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1) if values.size > 1 else 0.0)
    summary["num_cases"] = len(metrics_list)
    return summary


def evaluate_case_level_with_loss(
    dataloader,
    model,
    device: torch.device,
    data_dir: str | Path,
    criterion,
):
    if criterion is None:
        raise ValueError("criterion is required")
    grouped_predictions = defaultdict(list)
    grouped_targets = defaultdict(list)
    grouped_spacings: Dict[str, Tuple[float, float, float]] = {}
    total_loss = 0.0
    num_batches = 0
    model.eval()
    with torch.no_grad():
        for images, masks, filenames, spacings in tqdm(dataloader, desc="Case Eval"):
            images = images.to(device, non_blocking=True)
            targets = masks.to(device, non_blocking=True).long()
            logits_all = _forward(model, images, return_all=True)
            logits = logits_all[0] if isinstance(logits_all, (list, tuple)) else logits_all
            loss = criterion(logits_all, targets)
            total_loss += float(loss.item())
            num_batches += 1
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            masks_np = masks.cpu().numpy()
            spacing_np = spacings.cpu().numpy()
            for idx, filename in enumerate(filenames):
                case_id, block_idx = _parse_case_block(filename)
                grouped_predictions[case_id].append((block_idx, predictions[idx]))
                grouped_targets[case_id].append((block_idx, masks_np[idx]))
                grouped_spacings[case_id] = tuple(float(v) for v in spacing_np[idx].tolist())
    all_metrics: List[Dict[str, float]] = []
    case_names: List[str] = []
    for case_id in tqdm(sorted(grouped_predictions), desc="Case Metrics"):
        metadata = _load_case_metadata(data_dir, case_id)
        pred_volume = np.round(
            _reconstruct_volume(sorted(grouped_predictions[case_id]), metadata["ind_block"], metadata["original_shape"])
        ).astype(np.int64)
        true_volume = np.round(
            _reconstruct_volume(sorted(grouped_targets[case_id]), metadata["ind_block"], metadata["original_shape"])
        ).astype(np.int64)
        pred_onehot = torch.nn.functional.one_hot(torch.from_numpy(pred_volume).unsqueeze(0), num_classes=3).permute(0, 4, 1, 2, 3).float()
        true_tensor = torch.from_numpy(true_volume).unsqueeze(0)
        spacing_tensor = torch.tensor([grouped_spacings[case_id]], dtype=torch.float32)
        metrics = calculate_all_metrics_3d_multiclass_mm(pred_onehot, true_tensor, spacings=spacing_tensor, num_classes=3)
        all_metrics.append(metrics)
        case_names.append(case_id)
    summarized_metrics = _summarize_case_metrics(all_metrics)
    avg_loss = total_loss / max(1, num_batches)
    return avg_loss, summarized_metrics, case_names, all_metrics


def print_case_level_results(avg_metrics: Dict[str, float], dataset_type: str = "Validation") -> None:
    print(f"\n{'=' * 70}")
    print(f"{dataset_type} Case-Level Results")
    print(f"{'=' * 70}")
    print(f"Cases: {avg_metrics.get('num_cases', 0)}")
    print(f"TL Dice: {avg_metrics['dice_TL']:.4f} ± {avg_metrics['dice_TL_std']:.4f}")
    print(f"TL IoU: {avg_metrics['iou_TL']:.4f} ± {avg_metrics['iou_TL_std']:.4f}")
    print(f"TL Sensitivity: {avg_metrics['sensitivity_TL']:.4f} ± {avg_metrics['sensitivity_TL_std']:.4f}")
    print(f"TL PPV: {avg_metrics['ppv_TL']:.4f} ± {avg_metrics['ppv_TL_std']:.4f}")
    print(f"TL HD95: {avg_metrics['hd95_TL']:.2f} ± {avg_metrics['hd95_TL_std']:.2f} mm")
    print(f"TL ASSD: {avg_metrics['assd_TL']:.2f} ± {avg_metrics['assd_TL_std']:.2f} mm")
    print(f"FL Dice: {avg_metrics['dice_FL']:.4f} ± {avg_metrics['dice_FL_std']:.4f}")
    print(f"FL IoU: {avg_metrics['iou_FL']:.4f} ± {avg_metrics['iou_FL_std']:.4f}")
    print(f"FL Sensitivity: {avg_metrics['sensitivity_FL']:.4f} ± {avg_metrics['sensitivity_FL_std']:.4f}")
    print(f"FL PPV: {avg_metrics['ppv_FL']:.4f} ± {avg_metrics['ppv_FL_std']:.4f}")
    print(f"FL HD95: {avg_metrics['hd95_FL']:.2f} ± {avg_metrics['hd95_FL_std']:.2f} mm")
    print(f"FL ASSD: {avg_metrics['assd_FL']:.2f} ± {avg_metrics['assd_FL_std']:.2f} mm")
    print(f"Mean Dice: {avg_metrics['dice_mean']:.4f} ± {avg_metrics['dice_mean_std']:.4f}")
    print(f"{'=' * 70}\n")
