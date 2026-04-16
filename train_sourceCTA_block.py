from __future__ import annotations

import argparse
import datetime
import math
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from aortaseg import build_aortaseg
from dataset_CTA_block import get_cta_block_data_loaders
from evaluate_case_level import evaluate_case_level_with_loss, print_case_level_results
from loss import SegmentationLoss3D
from utils import wait_for_gpu


MODEL_NAME = "AortaSeg"


def summarize_case_metrics(all_metrics):
    if not all_metrics:
        raise ValueError("Empty metrics list")
    summary = {}
    for key in all_metrics[0].keys():
        try:
            values = np.asarray([float(item[key]) for item in all_metrics], dtype=np.float64)
        except Exception:
            continue
        summary[key] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1) if values.size > 1 else 0.0)
    summary["num_cases"] = len(all_metrics)
    return summary


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    progress = tqdm(train_loader, desc=f"{MODEL_NAME} Train {epoch + 1}/{total_epochs}")
    for step, (images, masks, _, _) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images, return_all=True)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += float(loss.item())
        lr = optimizer.param_groups[0]["lr"]
        progress.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss / step:.4f}", lr=f"{lr:.2e}")
    return total_loss / max(1, len(train_loader))


def validate_epoch(model, val_loader, criterion, device, data_dir, epoch, total_epochs):
    print(f"\n{MODEL_NAME} validation {epoch + 1}/{total_epochs}")
    avg_loss, _, _, all_metrics = evaluate_case_level_with_loss(val_loader, model, device, data_dir, criterion)
    return avg_loss, summarize_case_metrics(all_metrics)


def test_model(model, test_loader, criterion, device, data_dir, dataset_type: str):
    avg_loss, avg_metrics, _, _ = evaluate_case_level_with_loss(test_loader, model, device, data_dir, criterion)
    print(f"\n{dataset_type} loss: {avg_loss:.4f}")
    print_case_level_results(avg_metrics, dataset_type)
    return avg_metrics


def save_training_log(log_path: Path, epoch: int, train_loss: float, val_loss: float, val_metrics) -> None:
    content = (
        f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}]\n"
        f"Model: {MODEL_NAME}\n"
        f"Epoch: {epoch}\n"
        f"Train Loss: {train_loss:.4f}\n"
        f"Val Loss: {val_loss:.4f}\n"
        f"Dice TL: {val_metrics['dice_TL']:.4f} ± {val_metrics['dice_TL_std']:.4f}\n"
        f"Dice FL: {val_metrics['dice_FL']:.4f} ± {val_metrics['dice_FL_std']:.4f}\n"
        f"IoU TL: {val_metrics['iou_TL']:.4f} ± {val_metrics['iou_TL_std']:.4f}\n"
        f"IoU FL: {val_metrics['iou_FL']:.4f} ± {val_metrics['iou_FL_std']:.4f}\n"
        f"HD95 TL: {val_metrics['hd95_TL']:.2f} ± {val_metrics['hd95_TL_std']:.2f}\n"
        f"HD95 FL: {val_metrics['hd95_FL']:.2f} ± {val_metrics['hd95_FL_std']:.2f}\n"
        f"ASSD TL: {val_metrics['assd_TL']:.2f} ± {val_metrics['assd_TL_std']:.2f}\n"
        f"ASSD FL: {val_metrics['assd_FL']:.2f} ± {val_metrics['assd_FL_std']:.2f}\n"
        f"{'-' * 60}\n"
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(content)


def train(args) -> None:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader, dataset_type, _, _ = get_cta_block_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
        window_level=args.window_level,
        window_width=args.window_width,
        normalize=True,
        if_flt=bool(args.if_flt),
        excel_path=args.excel_path or None,
        return_case_splits=True,
        seed=args.seed,
    )
    model = build_aortaseg(device=device, deep_supervision=True)
    criterion = SegmentationLoss3D(
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        class_weights=torch.tensor(args.class_weights, dtype=torch.float32, device=device) if args.class_weights else None,
        ignore_background=True,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(args.min_lr / args.lr, cosine)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "aortaseg_best.pth"
    log_path = checkpoint_dir / "training_log_aortaseg.txt"
    best_dice = -float("inf")
    best_epoch = 0
    patience_counter = 0
    for epoch in range(args.epochs):
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, args.epochs)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, args.data_dir, epoch, args.epochs)
        current_dice = (val_metrics["dice_TL"] + val_metrics["dice_FL"]) / 2.0
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Mean Dice: {current_dice:.4f}")
        if current_dice > best_dice:
            best_dice = current_dice
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(
                {
                    "model_name": MODEL_NAME,
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            save_training_log(log_path, epoch + 1, train_loss, val_loss, val_metrics)
            print(f"Saved best checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{args.patience}")
    print(f"\nBest epoch: {best_epoch}")
    print(f"Best mean Dice: {best_dice:.4f}")
    if checkpoint_path.exists():
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_model(model, val_loader, criterion, device, args.data_dir, dataset_type)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AortaSeg source-domain trainer")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./artifacts/AortaSeg")
    parser.add_argument("--excel_path", type=str, default="")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=5e-7)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--window_level", type=float, default=1024.0)
    parser.add_argument("--window_width", type=float, default=4096.0)
    parser.add_argument("--if_flt", type=int, default=1)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--class_weights", type=float, nargs="+", default=[1.0, 1.0, 1.0])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--wait_for_gpu", action="store_true")
    parser.add_argument("--required_gb", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.wait_for_gpu and torch.cuda.is_available() and args.gpu >= 0:
        wait_for_gpu(required_gb=args.required_gb, gpu_id=args.gpu)
    train(args)


if __name__ == "__main__":
    main()
