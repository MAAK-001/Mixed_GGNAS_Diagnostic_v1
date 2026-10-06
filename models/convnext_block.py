"""ConvNeXt-style manual block with differentiable 5/7/9 depthwise scales."""

from __future__ import annotations

import torch
from torch import nn

from .scale_selection import ScaleMixture


class _ConvNeXtVariant(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depthwise_kernel: int) -> None:
        super().__init__()
        self.input_project = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.depthwise = nn.Conv2d(out_channels, out_channels, depthwise_kernel, padding=depthwise_kernel // 2, groups=out_channels)
        self.norm = nn.BatchNorm2d(out_channels)
        self.pointwise = nn.Sequential(nn.Conv2d(out_channels, 4 * out_channels, 1), nn.GELU(), nn.Conv2d(4 * out_channels, out_channels, 1))
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.input_project(x)
        output = self.pointwise(self.norm(self.depthwise(residual)))
        return self.activation(residual + output)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt pattern whose 5, 7 and 9 kernels map to the 3, 5 and 7 scale labels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.scales = ScaleMixture(
            _ConvNeXtVariant(in_channels, out_channels, 5),
            _ConvNeXtVariant(in_channels, out_channels, 7),
            _ConvNeXtVariant(in_channels, out_channels, 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scales(x)
