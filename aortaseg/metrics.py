from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


EMPTY_DISTANCE = 373.1287
FOREGROUND_CLASSES = (1, 2)


def _surface_metrics(pred_mask: np.ndarray, target_mask: np.ndarray, spacing: tuple[float, float, float]) -> tuple[float, float]:
    pred = pred_mask.astype(bool)
    target = target_mask.astype(bool)
    if not pred.any() and not target.any():
        return 0.0, 0.0
    if not pred.any() or not target.any():
        return EMPTY_DISTANCE, EMPTY_DISTANCE
    structure = np.ones((3, 3, 3), dtype=bool)
    pred_surface = pred ^ binary_erosion(pred, structure=structure, border_value=0)
    target_surface = target ^ binary_erosion(target, structure=structure, border_value=0)
    pred_distance = distance_transform_edt(~pred, sampling=spacing)
    target_distance = distance_transform_edt(~target, sampling=spacing)
    target_to_pred = pred_distance[target_surface]
    pred_to_target = target_distance[pred_surface]
    hd95 = max(np.percentile(target_to_pred, 95), np.percentile(pred_to_target, 95))
    assd = (float(np.mean(target_to_pred)) + float(np.mean(pred_to_target))) / 2.0
    return float(hd95), float(assd)


def _spacings_to_numpy(spacings, batch_size: int) -> np.ndarray:
    if spacings is None:
        return np.ones((batch_size, 3), dtype=np.float32)
    if isinstance(spacings, torch.Tensor):
        spacings_np = spacings.detach().cpu().numpy()
    else:
        spacings_np = np.asarray(spacings)
    if spacings_np.ndim == 1:
        return np.tile(spacings_np.astype(np.float32), (batch_size, 1))
    return spacings_np.astype(np.float32)


def _class_metrics(pred: torch.Tensor, target: torch.Tensor, class_idx: int, spacings: np.ndarray) -> Dict[str, float]:
    dice_values = []
    iou_values = []
    sensitivity_values = []
    ppv_values = []
    hd95_values = []
    assd_values = []
    for batch_idx in range(pred.shape[0]):
        pred_mask = pred[batch_idx] == class_idx
        target_mask = target[batch_idx] == class_idx
        pred_sum = int(pred_mask.sum().item())
        target_sum = int(target_mask.sum().item())
        intersection = int((pred_mask & target_mask).sum().item())
        union = pred_sum + target_sum
        if union == 0:
            dice = 1.0
        else:
            dice = (2.0 * intersection) / (union + 1e-8)
        denom_iou = pred_sum + target_sum - intersection
        if denom_iou == 0:
            iou = 1.0
        else:
            iou = intersection / (denom_iou + 1e-8)
        if target_sum == 0:
            sensitivity = 1.0
        else:
            sensitivity = intersection / (target_sum + 1e-8)
        if pred_sum == 0:
            ppv = 1.0
        else:
            ppv = intersection / (pred_sum + 1e-8)
        spacing = tuple(float(v) for v in spacings[batch_idx].tolist())
        hd95, assd = _surface_metrics(pred_mask.cpu().numpy(), target_mask.cpu().numpy(), spacing)
        dice_values.append(float(dice))
        iou_values.append(float(iou))
        sensitivity_values.append(float(sensitivity))
        ppv_values.append(float(ppv))
        hd95_values.append(hd95)
        assd_values.append(assd)
    return {
        "dice": float(np.mean(dice_values)),
        "dice_std": float(np.std(dice_values)),
        "iou": float(np.mean(iou_values)),
        "iou_std": float(np.std(iou_values)),
        "sensitivity": float(np.mean(sensitivity_values)),
        "sensitivity_std": float(np.std(sensitivity_values)),
        "ppv": float(np.mean(ppv_values)),
        "ppv_std": float(np.std(ppv_values)),
        "hd95": float(np.mean(hd95_values)),
        "hd95_std": float(np.std(hd95_values)),
        "assd": float(np.mean(assd_values)),
        "assd_std": float(np.std(assd_values)),
    }


def calculate_all_metrics_3d_multiclass_mm(
    logits: torch.Tensor,
    target: torch.Tensor,
    spacings=None,
    num_classes: int = 3,
) -> Dict[str, float]:
    if target.dim() == 5:
        target = target[:, 0]
    if logits.dim() == 5:
        pred = torch.argmax(logits, dim=1)
    elif logits.dim() == 4:
        pred = logits.long()
    else:
        raise ValueError(f"Unsupported logits shape {tuple(logits.shape)}")
    target = target.long()
    if num_classes < 3:
        raise ValueError("num_classes must be at least 3")
    spacings_np = _spacings_to_numpy(spacings, pred.shape[0])
    metrics: Dict[str, float] = {}
    dice_mean = []
    iou_mean = []
    sensitivity_mean = []
    ppv_mean = []
    hd95_mean = []
    assd_mean = []
    for class_idx in FOREGROUND_CLASSES:
        suffix = "TL" if class_idx == 1 else "FL"
        class_metrics = _class_metrics(pred, target, class_idx, spacings_np)
        metrics[f"dice_{suffix}"] = class_metrics["dice"]
        metrics[f"dice_{suffix}_std"] = class_metrics["dice_std"]
        metrics[f"iou_{suffix}"] = class_metrics["iou"]
        metrics[f"iou_{suffix}_std"] = class_metrics["iou_std"]
        metrics[f"sensitivity_{suffix}"] = class_metrics["sensitivity"]
        metrics[f"sensitivity_{suffix}_std"] = class_metrics["sensitivity_std"]
        metrics[f"ppv_{suffix}"] = class_metrics["ppv"]
        metrics[f"ppv_{suffix}_std"] = class_metrics["ppv_std"]
        metrics[f"hd95_{suffix}"] = class_metrics["hd95"]
        metrics[f"hd95_{suffix}_std"] = class_metrics["hd95_std"]
        metrics[f"assd_{suffix}"] = class_metrics["assd"]
        metrics[f"assd_{suffix}_std"] = class_metrics["assd_std"]
        dice_mean.append(class_metrics["dice"])
        iou_mean.append(class_metrics["iou"])
        sensitivity_mean.append(class_metrics["sensitivity"])
        ppv_mean.append(class_metrics["ppv"])
        hd95_mean.append(class_metrics["hd95"])
        assd_mean.append(class_metrics["assd"])
    metrics["dice_mean"] = float(np.mean(dice_mean))
    metrics["iou_mean"] = float(np.mean(iou_mean))
    metrics["sensitivity_mean"] = float(np.mean(sensitivity_mean))
    metrics["ppv_mean"] = float(np.mean(ppv_mean))
    metrics["hd95_mean"] = float(np.mean(hd95_mean))
    metrics["assd_mean"] = float(np.mean(assd_mean))
    return metrics
