from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss3D(nn.Module):
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: torch.Tensor | None = None,
        smooth: float = 1e-6,
        ignore_background: bool = True,
        deep_supervision_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.ignore_background = bool(ignore_background)
        self.deep_supervision_weights = list(deep_supervision_weights) if deep_supervision_weights is not None else None
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

    def _prepare_targets(self, targets: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        if targets.dim() == 5:
            targets = targets.squeeze(1)
        if targets.shape[-3:] != target_shape:
            targets = F.interpolate(targets.unsqueeze(1).float(), size=target_shape, mode="nearest").squeeze(1).long()
        return targets

    def _dice_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, targets.unsqueeze(1), 1)
        start_idx = 1 if self.ignore_background else 0
        losses = []
        for class_idx in range(start_idx, probs.shape[1]):
            pred_class = probs[:, class_idx]
            target_class = one_hot[:, class_idx]
            intersection = torch.sum(pred_class * target_class, dim=(1, 2, 3))
            pred_sum = torch.sum(pred_class, dim=(1, 2, 3))
            target_sum = torch.sum(target_class, dim=(1, 2, 3))
            dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            losses.append(1.0 - dice)
        if not losses:
            return torch.zeros((), dtype=probs.dtype, device=probs.device)
        return torch.mean(torch.stack(losses, dim=0))

    def _single_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(logits, targets.long())
        dice = self._dice_loss(torch.softmax(logits, dim=1), targets)
        return self.ce_weight * ce + self.dice_weight * dice

    def _deep_supervision_weights(self, n_outputs: int, device: torch.device) -> torch.Tensor:
        if self.deep_supervision_weights is not None:
            weights = torch.tensor(self.deep_supervision_weights, dtype=torch.float32, device=device)
            if weights.numel() != n_outputs:
                raise ValueError(f"Expected {n_outputs} deep supervision weights, got {weights.numel()}")
            if float(weights.sum().item()) == 0.0:
                raise ValueError("Deep supervision weights cannot sum to zero")
            return weights / weights.sum()
        if n_outputs == 1:
            return torch.tensor([1.0], dtype=torch.float32, device=device)
        weights = torch.tensor([1.0 / (2 ** idx) for idx in range(n_outputs)], dtype=torch.float32, device=device)
        weights[-1] = 0.0
        if float(weights.sum().item()) == 0.0:
            weights[0] = 1.0
        return weights / weights.sum()

    def forward(self, logits, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, (list, tuple)):
            if not logits:
                raise ValueError("Received empty logits list")
            weights = self._deep_supervision_weights(len(logits), logits[0].device)
            total_loss = torch.zeros((), dtype=logits[0].dtype, device=logits[0].device)
            for idx, logit in enumerate(logits):
                if float(weights[idx].item()) == 0.0:
                    continue
                target_i = targets[idx] if isinstance(targets, (list, tuple)) else targets
                target_i = self._prepare_targets(target_i, logit.shape[2:])
                total_loss = total_loss + weights[idx] * self._single_loss(logit, target_i)
            return total_loss
        targets = self._prepare_targets(targets, logits.shape[2:])
        return self._single_loss(logits, targets)
