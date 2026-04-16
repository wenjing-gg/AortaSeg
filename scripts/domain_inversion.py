from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aortaseg import build_aortaseg
from aortaseg.data import CTABlockDataset3D
from aortaseg.utils import wait_for_gpu


class DeepInversionFeatureHook:
    def __init__(self, module: nn.Module) -> None:
        self.hook = module.register_forward_hook(self._hook_fn)
        self.r_feature = torch.tensor(0.0)

    def _hook_fn(self, module: nn.Module, inputs, outputs) -> None:
        features = inputs[0]
        running_mean = getattr(module, "running_mean", None)
        running_var = getattr(module, "running_var", None)
        if running_mean is None or running_var is None:
            self.r_feature = torch.tensor(0.0, device=features.device)
            return
        channels = features.shape[1]
        mean = features.mean(dim=(0, 2, 3, 4))
        var = features.permute(1, 0, 2, 3, 4).contiguous().view(channels, -1).var(dim=1, unbiased=False)
        self.r_feature = torch.norm(running_var.data - var, p=2) + torch.norm(running_mean.data - mean, p=2)

    def close(self) -> None:
        self.hook.remove()


class StyleLoss:
    def __init__(self, hooks) -> None:
        self.hooks = hooks

    def __call__(self) -> torch.Tensor:
        if not self.hooks:
            return torch.tensor(0.0)
        return sum(hook.r_feature for hook in self.hooks) / len(self.hooks)


class ContentLoss(nn.Module):
    def __init__(self, target_features, weight: float = 1.0) -> None:
        super().__init__()
        self.targets = [feature.detach() for feature in target_features]
        self.weight = float(weight)
        self.criterion = nn.MSELoss()

    def forward(self, outputs) -> torch.Tensor:
        losses = [self.criterion(output * self.weight, target * self.weight) for output, target in zip(outputs, self.targets)]
        if not losses:
            device = outputs[0].device if outputs else "cpu"
            return torch.zeros((), dtype=torch.float32, device=device)
        return sum(losses) / len(losses)


class FSMGenerator3D:
    def __init__(self, low_freq_ratio: float = 0.1) -> None:
        if not 0.0 < low_freq_ratio < 0.5:
            raise ValueError("low_freq_ratio must be in (0, 0.5)")
        self.low_freq_ratio = float(low_freq_ratio)

    @staticmethod
    def _fftshift(x: torch.Tensor) -> torch.Tensor:
        shifts = [dim // 2 for dim in x.shape[-3:]]
        return torch.roll(x, shifts=shifts, dims=(-3, -2, -1))

    @staticmethod
    def _ifftshift(x: torch.Tensor) -> torch.Tensor:
        shifts = [-(dim // 2) for dim in x.shape[-3:]]
        return torch.roll(x, shifts=shifts, dims=(-3, -2, -1))

    def _fourier_domain_adaptation(self, source_like: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if source_like.shape != target.shape:
            raise ValueError(f"Shape mismatch: {tuple(source_like.shape)} vs {tuple(target.shape)}")
        src_fft = self._fftshift(torch.fft.fftn(source_like, dim=(-3, -2, -1)))
        tgt_fft = self._fftshift(torch.fft.fftn(target, dim=(-3, -2, -1)))
        src_amp = torch.abs(src_fft)
        tgt_amp = torch.abs(tgt_fft)
        tgt_phase = torch.angle(tgt_fft)
        depth, height, width = source_like.shape[-3:]
        band = int(min(depth, height, width) * self.low_freq_ratio)
        if band < 1:
            return target, source_like
        center_d, center_h, center_w = depth // 2, height // 2, width // 2
        d1, d2 = max(center_d - band, 0), min(center_d + band, depth)
        h1, h2 = max(center_h - band, 0), min(center_h + band, height)
        w1, w2 = max(center_w - band, 0), min(center_w + band, width)
        mixed_amp = tgt_amp.clone()
        mixed_amp[..., d1:d2, h1:h2, w1:w2] = src_amp[..., d1:d2, h1:h2, w1:w2]
        mixed_fft = self._ifftshift(mixed_amp * torch.exp(1j * tgt_phase))
        fst_image = torch.fft.ifftn(mixed_fft, dim=(-3, -2, -1)).real
        return fst_image, source_like


def collect_features(model: nn.Module, image: torch.Tensor):
    logits = model(image, return_all=True)
    if isinstance(logits, (list, tuple)):
        return [logits[0]]
    return [logits]


def generate_source_like(
    model: nn.Module,
    fsm_generator: FSMGenerator3D,
    target_image: torch.Tensor,
    bn_hooks,
    steps: int,
    lr: float,
    noise_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        target_features = [feature.detach() for feature in collect_features(model, target_image)]
    content_loss = ContentLoss(target_features)
    style_loss = StyleLoss(bn_hooks)
    optimized = (target_image + torch.randn_like(target_image) * float(noise_lambda)).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([optimized], lr=float(lr))
    min_val = float(target_image.min().item())
    max_val = float(target_image.max().item())
    for step_idx in range(int(steps)):
        optimizer.zero_grad()
        outputs = collect_features(model, optimized)
        content = content_loss(outputs)
        style = style_loss()
        loss = content + style
        loss.backward()
        optimizer.step()
        optimized.data.clamp_(min_val, max_val)
        if (step_idx + 1) % max(1, steps // 5) == 0 or step_idx == 0:
            print(
                f"DeepInversion {step_idx + 1}/{steps} "
                f"style={style.item():.4f} content={content.item():.4f} total={loss.item():.4f}"
            )
    with torch.no_grad():
        source_like, target_style = fsm_generator._fourier_domain_adaptation(optimized, target_image)
    return source_like, target_style


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AortaSeg domain inversion cache builder")
    parser.add_argument("--blocks_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_blocks_dir", type=str, required=True)
    parser.add_argument("--window_level", type=float, default=1024.0)
    parser.add_argument("--window_width", type=float, default=4096.0)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--if_flt", type=int, default=1)
    parser.add_argument("--excel_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--wait_for_gpu", action="store_true")
    parser.add_argument("--required_gb", type=float, default=8.0)
    parser.add_argument("--inv_steps", type=int, default=300)
    parser.add_argument("--inv_lr", type=float, default=0.01)
    parser.add_argument("--noise_lambda", type=float, default=0.3)
    parser.add_argument("--fft_low_ratio", type=float, default=0.1)
    return parser


def prepare_loader_and_model(args, device: torch.device):
    dataset = CTABlockDataset3D(
        data_dir=args.blocks_dir,
        phase="test",
        window_level=args.window_level,
        window_width=args.window_width,
        normalize=args.normalize,
        if_flt=bool(args.if_flt),
        augmentation=False,
        excel_path=args.excel_path or None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and device.type == "cuda",
        drop_last=False,
    )
    model = build_aortaseg(device=device, deep_supervision=True)
    checkpoint = torch.load(str(Path(args.checkpoint)), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return loader, model


def parse_case_and_index(filename: str) -> tuple[str, int]:
    base = Path(filename).name
    if base.endswith(".nii.gz"):
        base = base[:-7]
    if base.endswith(".npy"):
        base = base[:-4]
    if "_block" not in base:
        raise ValueError(f"Cannot parse block filename: {filename}")
    case_id, block_part = base.split("_block", 1)
    index_str = "".join(ch for ch in block_part if ch.isdigit())
    return case_id, int(index_str or 0)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    if args.wait_for_gpu and device.type == "cuda":
        wait_for_gpu(required_gb=args.required_gb, gpu_id=args.gpu)
    output_dir = Path(args.output_blocks_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader, model = prepare_loader_and_model(args, device)
    fsm_generator = FSMGenerator3D(low_freq_ratio=args.fft_low_ratio)
    bn_modules = [
        module
        for module in model.modules()
        if isinstance(module, (nn.BatchNorm3d, nn.InstanceNorm3d)) and getattr(module, "running_mean", None) is not None
    ]
    bn_hooks = [DeepInversionFeatureHook(module) for module in bn_modules[1:9]]
    case_buffers = defaultdict(list)
    try:
        for images, _, filenames, _ in tqdm(loader, desc="AortaSeg Inversion"):
            images = images.to(device, non_blocking=True)
            source_like, _ = generate_source_like(
                model,
                fsm_generator,
                images,
                bn_hooks,
                steps=args.inv_steps,
                lr=args.inv_lr,
                noise_lambda=args.noise_lambda,
            )
            for idx, filename in enumerate(filenames):
                case_id, block_idx = parse_case_and_index(filename)
                block_array = source_like[idx].detach().cpu().numpy().astype(np.float32)
                case_buffers[case_id].append((block_idx, block_array))
    finally:
        for hook in bn_hooks:
            hook.close()
    blocks_dir = Path(args.blocks_dir)
    for case_id, blocks in sorted(case_buffers.items()):
        src_path = blocks_dir / f"{case_id}_blocks.npz"
        if not src_path.exists():
            print(f"[Warning] Missing source NPZ: {src_path}")
            continue
        with np.load(src_path, allow_pickle=True) as data:
            payload = {key: data[key] for key in data.files}
        original_blocks = payload["blocks_img"]
        inverted_blocks = np.zeros_like(original_blocks, dtype=np.float32)
        for block_idx, array in sorted(blocks, key=lambda item: item[0]):
            if array.ndim == 4:
                array = np.squeeze(array, axis=0)
            target_shape = original_blocks[block_idx].shape
            cropped = array[: target_shape[0], : target_shape[1], : target_shape[2]]
            inverted_blocks[block_idx, : cropped.shape[0], : cropped.shape[1], : cropped.shape[2]] = cropped
        payload["blocks_img"] = inverted_blocks
        out_path = output_dir / f"{case_id}_blocks.npz"
        np.savez_compressed(out_path, **payload)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
