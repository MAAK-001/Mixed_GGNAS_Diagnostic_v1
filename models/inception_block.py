"""Inception-style manual block using multi-branch strip convolutions at three scales."""

from __future__ import annotations

import torch
from torch import nn

from .scale_selection import ScaleMixture


class _StripBranch(nn.Module):
    def __init__(self, channels: int, width: int) -> None:
        super().__init__()
        pad = width // 2
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, (1, width), padding=(0, pad), bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, (width, 1), padding=(pad, 0), bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class _InceptionVariant(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, widths: tuple[int, int, int]) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.branches = nn.ModuleList([_StripBranch(out_channels, width) for width in widths])
        self.project = nn.Conv2d(out_channels * 3, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class InceptionBlock(nn.Module):
    """Three paper-inspired strip-convolution branches, blended across three scale choices."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.scales = ScaleMixture(
            _InceptionVariant(in_channels, out_channels, (3, 5, 7)),
            _InceptionVariant(in_channels, out_channels, (5, 7, 9)),
            _InceptionVariant(in_channels, out_channels, (3, 7, 11)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scales(x)
