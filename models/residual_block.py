"""Residual manual block with the paper's three receptive-field variants."""

from __future__ import annotations

import torch
from torch import nn

from .scale_selection import ScaleMixture


class _ResidualVariant(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(x) + self.shortcut(x))


class ResidualBlock(nn.Module):
    """Blend 3x3, dilated-3x3, and dilated-5x5 residual variants by gradient descent."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.scales = ScaleMixture(
            _ResidualVariant(in_channels, out_channels, 3, 1),
            _ResidualVariant(in_channels, out_channels, 3, 3),
            _ResidualVariant(in_channels, out_channels, 5, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scales(x)
