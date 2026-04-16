from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def compute_blocks_for_volume(volume_shape, block_size: int = 64, block_pad: int = 0):
    xlen, ylen, zlen = volume_shape

    def axis_starts(length: int):
        if length <= block_size:
            return np.array([0], dtype=int)
        steps = int(np.ceil(length / block_size)) + block_pad
        start = 0
        end = length - block_size
        return np.round(np.linspace(start, end, steps)).astype(int)

    xstarts = axis_starts(xlen)
    ystarts = axis_starts(ylen)
    zstarts = axis_starts(zlen)
    blocks = []
    for xs in xstarts:
        for ys in ystarts:
            for zs in zstarts:
                xe = min(xs + block_size - 1, xlen - 1)
                ye = min(ys + block_size - 1, ylen - 1)
                ze = min(zs + block_size - 1, zlen - 1)
                blocks.append([xs, xe, ys, ye, zs, ze])
    ind_block = np.asarray(blocks, dtype=int)
    ind_brain = [0, xlen - 1, 0, ylen - 1, 0, zlen - 1]
    return ind_block, ind_brain


def get_aorta_bbox(segmentation: np.ndarray):
    coords = np.where(segmentation > 0)
    if len(coords[0]) == 0:
        raise ValueError("No aorta region found")
    return (
        int(coords[0].min()),
        int(coords[0].max()),
        int(coords[1].min()),
        int(coords[1].max()),
        int(coords[2].min()),
        int(coords[2].max()),
    )


def expand_bbox(bbox, original_shape, expansion_ratio: float):
    min_x, max_x, min_y, max_y, min_z, max_z = bbox
    x_pad = int((max_x - min_x + 1) * expansion_ratio / 2.0)
    y_pad = int((max_y - min_y + 1) * expansion_ratio / 2.0)
    z_pad = int((max_z - min_z + 1) * expansion_ratio / 2.0)
    start_x = max(min_x - x_pad, 0)
    end_x = min(max_x + x_pad + 1, original_shape[0])
    start_y = max(min_y - y_pad, 0)
    end_y = min(max_y + y_pad + 1, original_shape[1])
    start_z = max(min_z - z_pad, 0)
    end_z = min(max_z + z_pad + 1, original_shape[2])
    return start_x, end_x, start_y, end_y, start_z, end_z


def crop_volume(volume: np.ndarray, crop_bbox):
    start_x, end_x, start_y, end_y, start_z, end_z = crop_bbox
    return volume[start_x:end_x, start_y:end_y, start_z:end_z]


def run_totalsegmentator(image_path: str, roi_subset: list[str], fast: bool) -> np.ndarray:
    image = nib.load(image_path).get_fdata()
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "segmentation.nii.gz")
        try:
            from totalsegmentator.python_api import totalsegmentator

            totalsegmentator(
                input=image_path,
                output=output_path,
                roi_subset=roi_subset,
                ml=True,
                nr_thr_resamp=1,
                nr_thr_saving=1,
                fast=fast,
                quiet=False,
            )
            return nib.load(output_path).get_fdata()
        except ModuleNotFoundError:
            print("[Warning] totalsegmentator is not installed, using center fallback mask")
            fallback = np.zeros_like(image)
            center = np.array(image.shape) // 2
            fallback[tuple(center.tolist())] = 1
            return fallback
        except Exception as exc:
            print(f"[Warning] TotalSegmentator failed for {image_path}: {exc}")
            fallback = np.zeros_like(image)
            center = np.array(image.shape) // 2
            fallback[tuple(center.tolist())] = 1
            return fallback


def process_single_case(image_path: str, output_dir: Path, block_size: int, block_pad: int, expansion_ratio: float, roi_subset: list[str], fast: bool):
    image_nib = nib.load(image_path)
    image = image_nib.get_fdata()
    aorta_seg = run_totalsegmentator(image_path, roi_subset=roi_subset, fast=fast)
    crop_bbox = expand_bbox(get_aorta_bbox(aorta_seg), image.shape, expansion_ratio)
    cropped_image = crop_volume(image, crop_bbox)
    cropped_label = crop_volume(aorta_seg, crop_bbox)
    ind_block, ind_brain = compute_blocks_for_volume(cropped_image.shape, block_size=block_size, block_pad=block_pad)
    blocks_img = []
    blocks_label = []
    for ind in ind_block:
        xs, xe, ys, ye, zs, ze = map(int, ind)
        blocks_img.append(cropped_image[xs : xe + 1, ys : ye + 1, zs : ze + 1])
        blocks_label.append(cropped_label[xs : xe + 1, ys : ye + 1, zs : ze + 1])
    case_id = Path(image_path).name.replace("_image.nii.gz", "")
    output_path = output_dir / f"{case_id}_blocks.npz"
    np.savez_compressed(
        output_path,
        blocks_img=np.asarray(blocks_img),
        blocks_label=np.asarray(blocks_label),
        ind_block=ind_block,
        ind_brain=ind_brain,
        original_shape=cropped_image.shape,
        original_image_shape=image.shape,
        crop_bbox=crop_bbox,
        affine=image_nib.affine,
        case_id=case_id,
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AortaSeg VOI crop and block builder")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--block_pad", type=int, default=0)
    parser.add_argument("--expansion_ratio", type=float, default=0.1)
    parser.add_argument("--roi_subset", nargs="+", default=["aorta"])
    parser.add_argument("--fast", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted(path for path in data_dir.iterdir() if path.name.endswith("_image.nii.gz"))
    failed_cases = []
    for image_path in tqdm(image_files, desc="AortaSeg VOI"):
        try:
            output_path = process_single_case(
                str(image_path),
                output_dir=output_dir,
                block_size=args.block_size,
                block_pad=args.block_pad,
                expansion_ratio=args.expansion_ratio,
                roi_subset=list(args.roi_subset),
                fast=args.fast,
            )
            print(f"Saved {output_path}")
        except Exception as exc:
            print(f"[Error] Failed to process {image_path.name}: {exc}")
            failed_cases.append(image_path.name)
    print(f"Processed {len(image_files) - len(failed_cases)}/{len(image_files)} cases")
    if failed_cases:
        print(f"Failed cases: {failed_cases}")


if __name__ == "__main__":
    main()
