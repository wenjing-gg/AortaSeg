from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


DEFAULT_FEATURES = (32, 64, 125, 256, 320, 320)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-3:] == ref.shape[-3:]:
            return x
        diff = [ref.shape[-3 + idx] - x.shape[-3 + idx] for idx in range(3)]
        pad = []
        for delta in reversed(diff):
            before = max(delta // 2, 0)
            after = max(delta - before, 0)
            pad.extend([before, after])
        if any(pad):
            x = nn.functional.pad(x, pad)
        if x.shape[-3:] != ref.shape[-3:]:
            x = x[..., : ref.shape[-3], : ref.shape[-2], : ref.shape[-1]]
        return x

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self._match_size(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class AortaSeg(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 3,
        features: Sequence[int] = DEFAULT_FEATURES,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        if len(features) < 2:
            raise ValueError("AortaSeg requires at least two feature stages")
        self.deep_supervision = bool(deep_supervision)
        self.stem = ConvBlock(input_channels, int(features[0]))
        self.downsamples = nn.ModuleList()
        self.encoders = nn.ModuleList()
        for idx in range(len(features) - 1):
            self.downsamples.append(nn.Conv3d(int(features[idx]), int(features[idx + 1]), kernel_size=2, stride=2))
            self.encoders.append(ConvBlock(int(features[idx + 1]), int(features[idx + 1])))
        decoder_in = list(features)[::-1]
        self.decoders = nn.ModuleList(
            [
                UpBlock(int(decoder_in[idx]), int(decoder_in[idx + 1]), int(decoder_in[idx + 1]))
                for idx in range(len(decoder_in) - 1)
            ]
        )
        self.head = nn.Conv3d(int(features[0]), num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, return_all: bool = False):
        skips = [self.stem(x)]
        x = skips[0]
        for downsample, encoder in zip(self.downsamples, self.encoders):
            x = encoder(downsample(x))
            skips.append(x)
        x = skips[-1]
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = decoder(x, skip)
        logits = self.head(x)
        return [logits] if return_all else logits


def build_aortaseg(device: torch.device | None = None, deep_supervision: bool = True) -> AortaSeg:
    model = AortaSeg(deep_supervision=deep_supervision)
    return model.to(device) if device is not None else model
