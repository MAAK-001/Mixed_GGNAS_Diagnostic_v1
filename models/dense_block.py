"""Dense manual block with three learned receptive-field alternatives."""

from __future__ import annotations

import torch
from torch import nn

from .scale_selection import ScaleMixture


class _DenseVariant(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        growth = max(8, out_channels // 2)
        self.layers = nn.ModuleList()
        for layer_index in range(3):
            channels = in_channels + layer_index * growth
            padding = dilation * (kernel_size // 2)
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                nn.Conv2d(channels, growth, kernel_size, padding=padding, dilation=dilation, bias=False),
            ))
        self.project = nn.Conv2d(in_channels + 3 * growth, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        return self.project(torch.cat(features, dim=1))


class DenseBlock(nn.Module):
    """Dense connectivity with the same 3/5/7 scale encoding as the Mixed-GGNAS paper."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.scales = ScaleMixture(
            _DenseVariant(in_channels, out_channels, 3, 1),
            _DenseVariant(in_channels, out_channels, 3, 3),
            _DenseVariant(in_channels, out_channels, 5, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scales(x)
