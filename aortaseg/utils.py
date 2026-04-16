from __future__ import annotations

import subprocess
import time
from typing import Optional

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, generate_binary_structure, label as cc_label


TL = 1
FL = 2


def wait_for_gpu(required_gb: float = 8.0, gpu_id: int = 0, check_interval: int = 60) -> bool:
    while True:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader", f"--id={gpu_id}"],
                capture_output=True,
                text=True,
                check=True,
            )
            free_gb = int(result.stdout.strip()) / 1024.0
            print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id} free memory: {free_gb:.1f}GB")
            if free_gb >= required_gb:
                return True
        except FileNotFoundError:
            print("nvidia-smi not found, skip GPU wait")
            return False
        except Exception as exc:
            print(f"GPU check failed: {exc}")
        time.sleep(check_interval)


def filter_topk_lumen_components(pred_volume: np.ndarray, topk: int = 1) -> np.ndarray:
    if topk <= 0:
        return pred_volume
    lumen_mask = (pred_volume == TL) | (pred_volume == FL)
    if not lumen_mask.any():
        return pred_volume
    cc_map, num_components = cc_label(lumen_mask.astype(bool), structure=generate_binary_structure(3, 2))
    if num_components == 0:
        return pred_volume
    component_sizes = np.bincount(cc_map.ravel())
    keep_ids = np.argsort(component_sizes[1:])[::-1][: min(int(topk), num_components)] + 1
    keep_mask = np.isin(cc_map, keep_ids)
    filtered = pred_volume.copy()
    filtered[~keep_mask] = 0
    return filtered


def postprocess_lumen_and_refine_channels(
    pred_volume: np.ndarray,
    prob_volume: Optional[np.ndarray],
    topk: int = 1,
    closing_structure: Optional[np.ndarray] = None,
    closing_shape: str = "ball",
    closing_radius: int = 1,
    closing_iterations: int = 1,
) -> np.ndarray:
    def build_structure() -> np.ndarray:
        if closing_structure is not None:
            return closing_structure.astype(bool)
        radius = max(int(closing_radius), 1)
        if closing_shape == "ball":
            structure = generate_binary_structure(3, 2)
            if radius > 1:
                structure = binary_dilation(structure, iterations=radius - 1)
            return structure.astype(bool)
        if closing_shape == "ellipsoid":
            z_radius = max(1, radius // 2)
            return np.ones((2 * z_radius + 1, 2 * radius + 1, 2 * radius + 1), dtype=bool)
        return np.ones((2 * radius + 1, 2 * radius + 1, 2 * radius + 1), dtype=bool)

    lumen_mask = (pred_volume == TL) | (pred_volume == FL)
    if int(closing_iterations) > 0:
        lumen_mask = binary_closing(lumen_mask.astype(bool), structure=build_structure(), iterations=int(closing_iterations))
    refined = np.zeros_like(pred_volume, dtype=np.int64)
    if prob_volume is None:
        refined[lumen_mask] = pred_volume[lumen_mask]
    else:
        probs = np.asarray(prob_volume)
        if probs.ndim != 4 or probs.shape[0] < 3:
            raise ValueError("prob_volume must have shape (C, D, H, W) with C >= 3")
        tl_prob = probs[TL]
        fl_prob = probs[FL]
        inside = lumen_mask.astype(bool)
        refined_inside = np.where(tl_prob >= fl_prob, TL, FL)
        refined[inside] = refined_inside[inside]
    return filter_topk_lumen_components(refined, topk=int(topk)) if topk > 0 else refined
