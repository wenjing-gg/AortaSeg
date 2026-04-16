from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aortaseg import build_aortaseg
from aortaseg.data import CTABlockDataset3D
from aortaseg.metrics import calculate_all_metrics_3d_multiclass_mm
from aortaseg.utils import filter_topk_lumen_components, postprocess_lumen_and_refine_channels


BG_CLASS = 0
TL_CLASS = 1
FL_CLASS = 2
WINDOW_LEVEL = 1024.0
WINDOW_WIDTH = 4096.0
MODEL_NAME = "AortaSeg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AortaSeg test-time adaptation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--t_data_dir", type=str, required=True)
    parser.add_argument("--s_data_dir", type=str, required=True)
    parser.add_argument("--excel_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tta_steps", type=int, default=5)
    parser.add_argument("--tta_lr", type=float, default=1e-4)
    parser.add_argument("--pseudo_conf_threshold", type=float, default=0.8)
    parser.add_argument("--pseudo_loss_weight", type=float, default=1.0)
    parser.add_argument("--shape_loss_weight", type=float, default=1.0)
    parser.add_argument("--entropy_weight", type=float, default=1.0)
    parser.add_argument("--gap_boost", type=float, default=3.0)
    parser.add_argument("--distance_scale", type=float, default=2.5)
    parser.add_argument("--distance_decay", type=float, default=6.0)
    parser.add_argument("--post_topk", type=int, default=1)
    parser.add_argument("--closing_shape", type=str, default="ball", choices=["ball", "ellipsoid", "cube"])
    parser.add_argument("--closing_radius", type=int, default=1)
    parser.add_argument("--closing_iterations", type=int, default=5)
    parser.add_argument("--prior_topk", type=int, default=1)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--skip_cases", type=str, default="")
    parser.add_argument("--save_post_tta_dir", type=str, default="")
    parser.add_argument("--case_fusion_roi", type=str, default="prior", choices=["prior", "all"])
    parser.add_argument("--case_fusion_eps", type=float, default=1e-6)
    parser.add_argument("--debug_case_alpha", action="store_true")
    parser.add_argument("--stat_metrics", type=str, default="dice_mean,dice_TL,dice_FL,iou_mean,hd95_mean,assd_mean")
    parser.add_argument("--bootstrap_iters", type=int, default=5000)
    parser.add_argument("--perm_iters", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model(device: torch.device):
    return build_aortaseg(device=device, deep_supervision=True)


def _collect_case_block_counts(data_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    npz_files = sorted(data_dir.glob("*_blocks.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No *_blocks.npz files found in {data_dir}")
    for npz_path in npz_files:
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                case_id = str(data["case_id"]) if "case_id" in data else npz_path.name.replace("_blocks.npz", "")
                if "ind_block" in data:
                    counts[case_id] = int(data["ind_block"].shape[0])
                elif "blocks_img" in data:
                    counts[case_id] = int(data["blocks_img"].shape[0])
        except Exception as exc:
            print(f"[Warning] Failed to inspect {npz_path}: {exc}")
    return counts


def verify_s_t_alignment(s_data_dir: Path, t_data_dir: Path) -> List[str]:
    s_counts = _collect_case_block_counts(s_data_dir)
    t_counts = _collect_case_block_counts(t_data_dir)
    s_cases = set(s_counts)
    t_cases = set(t_counts)
    missing_in_s = sorted(t_cases - s_cases)
    missing_in_t = sorted(s_cases - t_cases)
    mismatched_blocks = [case_id for case_id in sorted(s_cases & t_cases) if s_counts[case_id] != t_counts[case_id]]
    if missing_in_s or missing_in_t or mismatched_blocks:
        if missing_in_s:
            print(f"[Mismatch] Missing in source-like data: {missing_in_s}")
        if missing_in_t:
            print(f"[Mismatch] Missing in target data: {missing_in_t}")
        if mismatched_blocks:
            print(
                "[Mismatch] Block count differs: "
                + ", ".join(f"{case_id} ({s_counts[case_id]} vs {t_counts[case_id]})" for case_id in mismatched_blocks)
            )
        raise ValueError("Source-like and target block sets do not match")
    return sorted(s_cases)


def create_dataloader(data_dir: str, args: argparse.Namespace) -> DataLoader:
    dataset = CTABlockDataset3D(
        data_dir=data_dir,
        phase="test",
        window_level=WINDOW_LEVEL,
        window_width=WINDOW_WIDTH,
        normalize=True,
        if_flt=True,
        augmentation=False,
        excel_path=args.excel_path or None,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and args.device.startswith("cuda"),
        drop_last=False,
    )


def _parse_case_block(filename: str) -> tuple[str, int]:
    parts = filename.rsplit("_block", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected filename format: {filename}")
    return parts[0], int(parts[1])


def _load_case_metadata(case_id: str, data_dir: Path, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    cached = cache.get(case_id)
    if cached is not None:
        return cached
    npz_path = data_dir / f"{case_id}_blocks.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        metadata = {
            "ind_block": np.array(data["ind_block"]),
            "original_shape": tuple(int(v) for v in data["original_shape"]),
        }
        if "original_image_shape" in data:
            metadata["original_image_shape"] = tuple(int(v) for v in data["original_image_shape"])
        if "crop_bbox" in data:
            metadata["crop_bbox"] = tuple(int(v) for v in data["crop_bbox"])
        if "affine" in data:
            metadata["affine"] = np.array(data["affine"], dtype=np.float64)
    cache[case_id] = metadata
    return metadata


def _fill_volume_to_original_shape(
    cropped_volume: np.ndarray,
    crop_bbox: Tuple[int, int, int, int, int, int],
    original_shape: Tuple[int, int, int],
) -> np.ndarray:
    output = np.zeros(original_shape, dtype=cropped_volume.dtype)
    sx, ex, sy, ey, sz, ez = map(int, crop_bbox)
    output[sx:ex, sy:ey, sz:ez] = cropped_volume
    return output


def _save_post_tta_prediction(case_id: str, pred_volume: np.ndarray, metadata: Dict[str, Any], save_dir: str) -> Path:
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    affine = np.array(metadata.get("affine", np.eye(4)), dtype=np.float64)
    crop_bbox = metadata.get("crop_bbox")
    original_image_shape = metadata.get("original_image_shape")
    if crop_bbox is not None and original_image_shape is not None:
        volume_to_save = _fill_volume_to_original_shape(pred_volume.astype(np.int16), crop_bbox, original_image_shape)
        output_path = output_dir / f"{case_id}_mask_ori.nii.gz"
    else:
        volume_to_save = pred_volume.astype(np.int16)
        output_path = output_dir / f"{case_id}_pred.nii.gz"
    nib.save(nib.Nifti1Image(volume_to_save, affine), str(output_path))
    return output_path


def _crop_dims(ind: np.ndarray, original_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    sx, ex, sy, ey, sz, ez = map(int, ind)
    return (
        int(min(ex - sx + 1, original_shape[0] - sx)),
        int(min(ey - sy + 1, original_shape[1] - sy)),
        int(min(ez - sz + 1, original_shape[2] - sz)),
    )


def _reconstruct_prob_volume(
    blocks: List[Tuple[int, np.ndarray]],
    ind_block: np.ndarray,
    original_shape: Tuple[int, int, int],
) -> np.ndarray:
    channels = blocks[0][1].shape[0]
    volume = np.zeros((channels,) + original_shape, dtype=np.float32)
    count = np.zeros(original_shape, dtype=np.float32)
    for block_idx, block in blocks:
        sx, ex, sy, ey, sz, ez = map(int, ind_block[block_idx])
        depth, height, width = block.shape[1:]
        ex_adj = min(sx + depth, original_shape[0])
        ey_adj = min(sy + height, original_shape[1])
        ez_adj = min(sz + width, original_shape[2])
        cropped = block[:, : ex_adj - sx, : ey_adj - sy, : ez_adj - sz].astype(np.float32)
        volume[:, sx:ex_adj, sy:ey_adj, sz:ez_adj] += cropped
        count[sx:ex_adj, sy:ey_adj, sz:ez_adj] += 1.0
    count[count < 0.5] = 1.0
    return volume / count


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


def infer_blocks(model, entries, ind_block, original_shape, device: torch.device, batch_size: int) -> List[Tuple[int, np.ndarray]]:
    model.eval()
    prob_blocks: List[Tuple[int, np.ndarray]] = []
    with torch.no_grad():
        for start in range(0, len(entries), batch_size):
            batch_entries = entries[start : start + batch_size]
            images = torch.stack([entry["image"] for entry in batch_entries]).to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for offset, entry in enumerate(batch_entries):
                block_idx = entry["block_idx"]
                depth, height, width = _crop_dims(ind_block[block_idx], original_shape)
                prob_blocks.append((block_idx, probs[offset][:, :depth, :height, :width]))
    return prob_blocks


def generate_lumen_prior(
    prob_volume: np.ndarray,
    topk: int,
    closing_shape: str,
    closing_radius: int,
    closing_iterations: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.ndimage import binary_closing, binary_dilation, generate_binary_structure

    pred_labels = np.argmax(prob_volume[:3], axis=0).astype(np.int64)
    raw_union_mask = (pred_labels == TL_CLASS) | (pred_labels == FL_CLASS)
    if topk > 0:
        raw_union_mask = filter_topk_lumen_components(raw_union_mask.astype(np.int64), topk=topk) > 0
    if int(closing_iterations) <= 0:
        lumen_prior = raw_union_mask.astype(bool)
    else:
        radius = max(int(closing_radius), 1)
        if closing_shape == "ball":
            structure = generate_binary_structure(3, 2)
            if radius > 1:
                structure = binary_dilation(structure, iterations=radius - 1)
        elif closing_shape == "ellipsoid":
            z_radius = max(1, radius // 2)
            structure = np.ones((2 * z_radius + 1, 2 * radius + 1, 2 * radius + 1), dtype=bool)
        else:
            structure = np.ones((2 * radius + 1, 2 * radius + 1, 2 * radius + 1), dtype=bool)
        lumen_prior = binary_closing(raw_union_mask.astype(bool), structure=structure, iterations=int(closing_iterations))
    gap_mask = lumen_prior & (~raw_union_mask)
    return lumen_prior.astype(bool), raw_union_mask.astype(bool), gap_mask.astype(bool)


def refine_pseudo_labels_with_prior(pseudo_labels: np.ndarray, prob_volume: np.ndarray, lumen_prior: np.ndarray) -> np.ndarray:
    refined = np.asarray(pseudo_labels, dtype=np.int64).copy()
    refined[~lumen_prior] = BG_CLASS
    inside = lumen_prior & (refined == BG_CLASS)
    refined[inside] = np.where(prob_volume[TL_CLASS][inside] >= prob_volume[FL_CLASS][inside], TL_CLASS, FL_CLASS)
    return refined


def compute_high_confidence_mask(prob_volume: np.ndarray, threshold: float) -> np.ndarray:
    return prob_volume.max(axis=0) >= float(threshold)


def distance_weight_map_from_prior(lumen_prior: np.ndarray, distance_scale: float, distance_decay: float) -> np.ndarray:
    if not lumen_prior.any():
        return np.ones_like(lumen_prior, dtype=np.float32)
    distance = distance_transform_edt(~lumen_prior.astype(bool))
    decay = max(float(distance_decay), 1e-3)
    return (1.0 + float(distance_scale) * np.exp(-distance / decay)).astype(np.float32)


def shape_weight_map(lumen_prior: np.ndarray, gap_mask: np.ndarray, gap_boost: float) -> np.ndarray:
    weights = np.ones_like(lumen_prior, dtype=np.float32)
    if float(gap_boost) != 0.0:
        weights[gap_mask] += float(gap_boost)
    return weights


def _finalize_prediction(prob_volume: np.ndarray, lumen_prior: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    pred = np.argmax(prob_volume, axis=0).astype(np.int64)
    pred[~lumen_prior] = BG_CLASS
    inside = lumen_prior & (pred == BG_CLASS)
    pred[inside] = np.where(prob_volume[TL_CLASS][inside] >= prob_volume[FL_CLASS][inside], TL_CLASS, FL_CLASS)
    pred = postprocess_lumen_and_refine_channels(
        pred_volume=pred,
        prob_volume=prob_volume,
        topk=int(args.post_topk),
        closing_shape=args.closing_shape,
        closing_radius=int(args.closing_radius),
        closing_iterations=int(args.closing_iterations),
    )
    pred[~lumen_prior] = BG_CLASS
    return pred


def _compute_case_metrics(pred_volume: np.ndarray, true_volume: np.ndarray, spacing: Tuple[float, float, float]) -> Dict[str, float]:
    pred_tensor = torch.from_numpy(pred_volume).unsqueeze(0)
    true_tensor = torch.from_numpy(true_volume).unsqueeze(0)
    pred_onehot = F.one_hot(pred_tensor.long(), num_classes=3).permute(0, 4, 1, 2, 3).float()
    spacing_tensor = torch.tensor([spacing], dtype=torch.float32)
    return calculate_all_metrics_3d_multiclass_mm(pred_onehot, true_tensor, spacings=spacing_tensor, num_classes=3)


def _summarize_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {}
    summary: Dict[str, float] = {}
    for key in metrics_list[0]:
        values = [float(item[key]) for item in metrics_list if key in item]
        summary[key] = float(np.mean(values))
    return summary


def _check_numpy_prob_volume(name: str, array: np.ndarray, case_id: str) -> Optional[str]:
    if not np.isfinite(array).all():
        return f"{name} contains NaN/Inf for {case_id}"
    min_val = float(array.min())
    max_val = float(array.max())
    if min_val < -1e-6 or max_val > 1.0 + 1e-6:
        return f"{name} out of range [{min_val:.4f}, {max_val:.4f}] for {case_id}"
    return None


def _case_mean_entropy(prob_volume: np.ndarray, roi_mask: Optional[np.ndarray], eps: float) -> float:
    prob = np.asarray(prob_volume[:3], dtype=np.float32)
    prob = np.clip(prob, float(eps), 1.0)
    entropy = -np.sum(prob * np.log(prob), axis=0)
    if roi_mask is not None and roi_mask.any():
        return float(entropy[roi_mask].mean())
    return float(entropy.mean())


def _case_alpha_scheme2_softmax(
    s_prob: np.ndarray,
    t_prob: np.ndarray,
    lumen_prior: np.ndarray,
    roi_mode: str,
    eps: float,
) -> Tuple[float, float, float]:
    roi = lumen_prior.astype(bool) if roi_mode == "prior" else None
    source_entropy = _case_mean_entropy(s_prob, roi, eps)
    target_entropy = _case_mean_entropy(t_prob, roi, eps)
    source_logit = -source_entropy
    target_logit = -target_entropy
    max_logit = max(source_logit, target_logit)
    source_weight = np.exp(source_logit - max_logit)
    target_weight = np.exp(target_logit - max_logit)
    alpha = float(target_weight / (source_weight + target_weight + 1e-12))
    return alpha, source_entropy, target_entropy


def _bootstrap_ci_mean(delta: np.ndarray, iters: int, rng: np.random.Generator) -> Tuple[float, float]:
    if delta.shape[0] <= 1:
        return float("nan"), float("nan")
    means = np.empty(int(iters), dtype=np.float64)
    for idx in range(int(iters)):
        sample_idx = rng.integers(0, delta.shape[0], size=delta.shape[0])
        means[idx] = float(delta[sample_idx].mean())
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    return float(ci_low), float(ci_high)


def _perm_signflip_pvalue(delta: np.ndarray, iters: int, rng: np.random.Generator) -> float:
    if delta.shape[0] <= 1:
        return float("nan")
    observed = abs(float(delta.mean()))
    count = 0
    for _ in range(int(iters)):
        signs = rng.choice([-1.0, 1.0], size=delta.shape[0], replace=True)
        if abs(float((delta * signs).mean())) >= observed:
            count += 1
    return float((count + 1.0) / (iters + 1.0))


def _compute_ci_pvalue(
    pre_list: List[Dict[str, float]],
    post_list: List[Dict[str, float]],
    keys: List[str],
    bootstrap_iters: int,
    perm_iters: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    output: Dict[str, Dict[str, float]] = {}
    n_cases = min(len(pre_list), len(post_list))
    for key in keys:
        pre_vals = []
        post_vals = []
        for idx in range(n_cases):
            if key in pre_list[idx] and key in post_list[idx]:
                pre_vals.append(float(pre_list[idx][key]))
                post_vals.append(float(post_list[idx][key]))
        if not pre_vals:
            continue
        pre_arr = np.asarray(pre_vals, dtype=np.float64)
        post_arr = np.asarray(post_vals, dtype=np.float64)
        delta = post_arr - pre_arr
        ci_low, ci_high = _bootstrap_ci_mean(delta, bootstrap_iters, rng)
        output[key] = {
            "n": float(len(delta)),
            "pre_mean": float(pre_arr.mean()),
            "post_mean": float(post_arr.mean()),
            "delta_mean": float(delta.mean()),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": _perm_signflip_pvalue(delta, perm_iters, rng),
        }
    return output


def adapt_target_model(
    t_model,
    entries: List[Dict[str, Any]],
    ind_block: np.ndarray,
    original_shape: Tuple[int, int, int],
    pseudo_labels: np.ndarray,
    high_conf_mask: np.ndarray,
    lumen_prior: np.ndarray,
    gap_mask: np.ndarray,
    distance_weights: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    case_id: str,
) -> Optional[str]:
    optimizer = torch.optim.Adam(t_model.parameters(), lr=float(args.tta_lr))
    shape_weights = shape_weight_map(lumen_prior, gap_mask, args.gap_boost)
    t_model.train()
    for step in range(int(args.tta_steps)):
        for start in range(0, len(entries), int(args.batch_size)):
            batch_entries = entries[start : start + int(args.batch_size)]
            images = torch.stack([entry["image"] for entry in batch_entries]).to(device, non_blocking=True)
            pseudo_blocks = []
            conf_blocks = []
            distance_blocks = []
            shape_blocks = []
            prior_blocks = []
            crop_sizes = []
            for entry in batch_entries:
                block_idx = entry["block_idx"]
                depth, height, width = _crop_dims(ind_block[block_idx], original_shape)
                crop_sizes.append((depth, height, width))
                sx, ex, sy, ey, sz, ez = map(int, ind_block[block_idx])
                slices = (slice(sx, sx + depth), slice(sy, sy + height), slice(sz, sz + width))
                pseudo_blocks.append(torch.from_numpy(pseudo_labels[slices]))
                conf_blocks.append(torch.from_numpy(high_conf_mask[slices]).float())
                distance_blocks.append(torch.from_numpy(distance_weights[slices]).float())
                shape_blocks.append(torch.from_numpy(shape_weights[slices]).float())
                prior_blocks.append(torch.from_numpy(lumen_prior[slices].astype(np.float32)))
            pseudo_blocks = torch.stack(pseudo_blocks).to(device)
            conf_blocks = torch.stack(conf_blocks).to(device)
            distance_blocks = torch.stack(distance_blocks).to(device)
            shape_blocks = torch.stack(shape_blocks).to(device)
            prior_blocks = torch.stack(prior_blocks).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = t_model(images)
            if not torch.isfinite(logits).all():
                return f"logits contains NaN/Inf for {case_id} at step {step}"
            probs = torch.softmax(logits, dim=1)
            if not torch.isfinite(probs).all():
                return f"probs contains NaN/Inf for {case_id} at step {step}"
            batch_loss = torch.zeros((), dtype=logits.dtype, device=device)
            neg_indices = [index for index in range(logits.shape[1]) if index not in (TL_CLASS, FL_CLASS)]
            for idx, (depth, height, width) in enumerate(crop_sizes):
                logits_i = logits[idx : idx + 1, :, :depth, :height, :width]
                probs_i = probs[idx : idx + 1, :, :depth, :height, :width]
                pseudo_i = pseudo_blocks[idx : idx + 1, :depth, :height, :width]
                conf_i = conf_blocks[idx, :depth, :height, :width]
                distance_i = distance_blocks[idx, :depth, :height, :width]
                shape_i = shape_blocks[idx, :depth, :height, :width]
                prior_i = prior_blocks[idx : idx + 1, :depth, :height, :width]
                ce = F.cross_entropy(logits_i, pseudo_i, reduction="none").squeeze(0)
                bg_mask = (pseudo_i.squeeze(0) == BG_CLASS).float()
                weight_map = conf_i * torch.where(bg_mask > 0, distance_i, torch.ones_like(distance_i))
                if float(weight_map.sum().item()) > 0.0:
                    ce_loss = (ce * weight_map).sum() / (weight_map.sum() + 1e-6)
                else:
                    ce_loss = torch.zeros((), dtype=logits.dtype, device=device)
                pos_logits = torch.logsumexp(logits_i[:, TL_CLASS : FL_CLASS + 1], dim=1)
                if neg_indices:
                    neg_logits = torch.logsumexp(logits_i[:, neg_indices], dim=1)
                else:
                    neg_logits = torch.zeros_like(pos_logits)
                lumen_logits = pos_logits - neg_logits
                bce = F.binary_cross_entropy_with_logits(lumen_logits, prior_i, reduction="none").squeeze(0)
                if float(shape_i.sum().item()) > 0.0:
                    shape_loss = (bce * shape_i).sum() / (shape_i.sum() + 1e-6)
                else:
                    shape_loss = torch.zeros((), dtype=logits.dtype, device=device)
                entropy_loss = -(probs_i * torch.log(torch.clamp(probs_i, min=1e-6))).sum(dim=1).mean()
                block_loss = (
                    float(args.pseudo_loss_weight) * ce_loss
                    + float(args.shape_loss_weight) * shape_loss
                    + float(args.entropy_weight) * entropy_loss
                )
                if not torch.isfinite(block_loss):
                    return f"block loss contains NaN/Inf for {case_id} at step {step}"
                batch_loss = batch_loss + block_loss
            batch_loss = batch_loss / max(1, len(batch_entries))
            batch_loss.backward()
            optimizer.step()
    return None


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device = torch.device(args.device if use_cuda else "cpu")
    print(f"Using device: {device}")
    case_list = verify_s_t_alignment(Path(args.s_data_dir), Path(args.t_data_dir))
    skip_cases = {case_id.strip() for case_id in args.skip_cases.split(",") if case_id.strip()}
    if skip_cases:
        case_list = [case_id for case_id in case_list if case_id not in skip_cases]
    total_cases = len(case_list)
    if args.max_cases and args.max_cases > 0:
        total_cases = min(total_cases, int(args.max_cases))
    case_pbar = tqdm(total=total_cases, desc=f"{MODEL_NAME} TTA")
    s_model = build_model(device)
    checkpoint = torch.load(str(Path(args.checkpoint)), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    s_model.load_state_dict(state_dict, strict=False)
    for parameter in s_model.parameters():
        parameter.requires_grad = False
    s_model.eval()
    dataloader_s = create_dataloader(args.s_data_dir, args)
    dataloader_t = create_dataloader(args.t_data_dir, args)
    metadata_cache: Dict[str, Dict[str, Any]] = {}
    current_case = None
    current_entries_s: List[Dict[str, Any]] = []
    current_entries_t: List[Dict[str, Any]] = []
    processed_cases = 0
    pre_metrics_list: List[Dict[str, float]] = []
    post_metrics_list: List[Dict[str, float]] = []
    stat_keys = [key.strip() for key in args.stat_metrics.split(",") if key.strip()]

    def flush_case() -> bool:
        nonlocal processed_cases, current_case, current_entries_s, current_entries_t
        if current_case is None or not current_entries_s or not current_entries_t:
            return True
        metadata = _load_case_metadata(current_case, Path(args.t_data_dir), metadata_cache)
        ind_block = metadata["ind_block"]
        original_shape = metadata["original_shape"]
        spacing_s = tuple(current_entries_s[0]["spacing"])
        spacing_t = tuple(current_entries_t[0]["spacing"])
        s_prob_blocks = infer_blocks(s_model, current_entries_s, ind_block, original_shape, device, batch_size=args.batch_size)
        s_prob_volume = _reconstruct_prob_volume(s_prob_blocks, ind_block, original_shape)
        error = _check_numpy_prob_volume("s_prob_volume", s_prob_volume, current_case)
        if error is not None:
            print(f"[Error] {error}")
            return False
        pseudo_raw = np.argmax(s_prob_volume, axis=0).astype(np.int64)
        lumen_prior, _, gap_mask = generate_lumen_prior(
            prob_volume=s_prob_volume,
            topk=args.prior_topk,
            closing_shape=args.closing_shape,
            closing_radius=args.closing_radius,
            closing_iterations=args.closing_iterations,
        )
        pseudo_labels = refine_pseudo_labels_with_prior(pseudo_raw, s_prob_volume, lumen_prior)
        high_conf_mask = compute_high_confidence_mask(s_prob_volume, args.pseudo_conf_threshold)
        distance_weights = distance_weight_map_from_prior(lumen_prior, args.distance_scale, args.distance_decay)
        s_pred = _finalize_prediction(s_prob_volume, lumen_prior, args)
        true_blocks_s = []
        for entry in current_entries_s:
            block_idx = entry["block_idx"]
            depth, height, width = _crop_dims(ind_block[block_idx], original_shape)
            true_blocks_s.append((block_idx, entry["label"].cpu().numpy()[:depth, :height, :width]))
        true_volume_s = np.round(_reconstruct_volume(true_blocks_s, ind_block, original_shape)).astype(np.int64)
        pre_metrics = _compute_case_metrics(s_pred, true_volume_s, spacing_s)
        pre_metrics_list.append(pre_metrics)
        t_model = build_model(device)
        t_model.load_state_dict(s_model.state_dict(), strict=False)
        error = adapt_target_model(
            t_model=t_model,
            entries=current_entries_t,
            ind_block=ind_block,
            original_shape=original_shape,
            pseudo_labels=pseudo_labels,
            high_conf_mask=high_conf_mask,
            lumen_prior=lumen_prior,
            gap_mask=gap_mask,
            distance_weights=distance_weights,
            args=args,
            device=device,
            case_id=current_case,
        )
        if error is not None:
            print(f"[Error] {error}")
            return False
        t_prob_blocks = infer_blocks(t_model, current_entries_t, ind_block, original_shape, device, batch_size=args.batch_size)
        t_prob_volume = _reconstruct_prob_volume(t_prob_blocks, ind_block, original_shape)
        error = _check_numpy_prob_volume("t_prob_volume", t_prob_volume, current_case)
        if error is not None:
            print(f"[Error] {error}")
            return False
        alpha_case, source_entropy, target_entropy = _case_alpha_scheme2_softmax(
            s_prob=s_prob_volume,
            t_prob=t_prob_volume,
            lumen_prior=lumen_prior,
            roi_mode=args.case_fusion_roi,
            eps=args.case_fusion_eps,
        )
        ensemble_prob = (1.0 - alpha_case) * s_prob_volume + alpha_case * t_prob_volume
        if args.debug_case_alpha:
            print(
                f"[Case Alpha] {current_case} roi={args.case_fusion_roi} "
                f"Hs={source_entropy:.6f} Ht={target_entropy:.6f} alpha={alpha_case:.4f}"
            )
        t_pred = _finalize_prediction(ensemble_prob, lumen_prior, args)
        if args.save_post_tta_dir:
            _save_post_tta_prediction(current_case, t_pred, metadata, args.save_post_tta_dir)
        true_blocks_t = []
        for entry in current_entries_t:
            block_idx = entry["block_idx"]
            depth, height, width = _crop_dims(ind_block[block_idx], original_shape)
            true_blocks_t.append((block_idx, entry["label"].cpu().numpy()[:depth, :height, :width]))
        true_volume_t = np.round(_reconstruct_volume(true_blocks_t, ind_block, original_shape)).astype(np.int64)
        post_metrics = _compute_case_metrics(t_pred, true_volume_t, spacing_t)
        post_metrics_list.append(post_metrics)
        processed_cases += 1
        case_pbar.update(1)
        delta = post_metrics["dice_mean"] - pre_metrics["dice_mean"]
        print(f"{current_case}: {pre_metrics['dice_mean']:.4f} -> {post_metrics['dice_mean']:.4f} (delta {delta:+.4f})")
        if args.max_cases and processed_cases >= args.max_cases:
            return False
        return True

    for batch_s, batch_t in zip(dataloader_s, dataloader_t):
        images_s, masks_s, filenames_s, spacings_s = batch_s
        images_t, masks_t, filenames_t, spacings_t = batch_t
        if list(filenames_s) != list(filenames_t):
            raise ValueError("Source-like and target dataloader order mismatch")
        images_s_cpu = images_s.cpu()
        images_t_cpu = images_t.cpu()
        masks_s_cpu = masks_s.cpu()
        masks_t_cpu = masks_t.cpu()
        spacings_s_np = spacings_s.cpu().numpy()
        spacings_t_np = spacings_t.cpu().numpy()
        for idx, filename in enumerate(filenames_t):
            case_id, block_idx = _parse_case_block(filename)
            if case_id in skip_cases:
                continue
            if current_case is None:
                current_case = case_id
            if case_id != current_case:
                if not flush_case():
                    case_pbar.close()
                    return
                current_entries_s = []
                current_entries_t = []
                current_case = case_id
            current_entries_s.append(
                {
                    "image": images_s_cpu[idx],
                    "label": masks_s_cpu[idx],
                    "block_idx": block_idx,
                    "spacing": tuple(float(v) for v in spacings_s_np[idx].tolist()),
                }
            )
            current_entries_t.append(
                {
                    "image": images_t_cpu[idx],
                    "label": masks_t_cpu[idx],
                    "block_idx": block_idx,
                    "spacing": tuple(float(v) for v in spacings_t_np[idx].tolist()),
                }
            )
    if not flush_case():
        case_pbar.close()
        return
    case_pbar.close()
    pre_summary = _summarize_metrics(pre_metrics_list)
    post_summary = _summarize_metrics(post_metrics_list)
    if pre_summary and post_summary:
        print("\n===== AortaSeg Pre-TTA vs Post-TTA =====")
        for key in stat_keys:
            pre_val = pre_summary.get(key, float("nan"))
            post_val = post_summary.get(key, float("nan"))
            print(f"{key}: {pre_val:.4f} -> {post_val:.4f} (delta {post_val - pre_val:+.4f})")
        stats = _compute_ci_pvalue(
            pre_list=pre_metrics_list,
            post_list=post_metrics_list,
            keys=stat_keys,
            bootstrap_iters=args.bootstrap_iters,
            perm_iters=args.perm_iters,
            seed=args.seed,
        )
        if stats:
            print("\n===== Paired Statistics =====")
            for key in stat_keys:
                if key not in stats:
                    continue
                stat = stats[key]
                print(
                    f"{key}: {stat['pre_mean']:.4f} -> {stat['post_mean']:.4f} "
                    f"(delta {stat['delta_mean']:+.4f}) "
                    f"95%CI [{stat['ci_low']:+.4f}, {stat['ci_high']:+.4f}] "
                    f"p={stat['p_value']:.6f} n={int(stat['n'])}"
                )
    print(f"{MODEL_NAME} TTA finished")


if __name__ == "__main__":
    main()
