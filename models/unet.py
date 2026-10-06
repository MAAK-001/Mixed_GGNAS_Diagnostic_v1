"""U-shaped segmentation network whose eight block positions come from a chromosome."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from collections.abc import Sequence


class _Bridge(nn.Module):
    """Fixed bottleneck between searchable encoder and decoder block positions."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MixedUNet(nn.Module):
    """Four encoder + bridge + four decoder architecture with standard skip connections."""

    def __init__(self, chromosome: list[int], base_channels: int = 16, selected_scales: Sequence[int] | None = None) -> None:
        super().__init__()
        # Imported here to avoid an architecture_builder <-> unet import cycle.
        from .architecture_builder import build_block

        self.chromosome = chromosome
        self.selected_scales = None if selected_scales is None else [int(scale) for scale in selected_scales]
        channels = [base_channels * (2 ** level) for level in range(4)]
        self.stem = nn.Sequential(nn.Conv2d(3, channels[0], 3, padding=1, bias=False), nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(4)])
        current = channels[0]
        for block_id, out_channels in zip(chromosome[:4], channels):
            self.encoders.append(build_block(block_id, current, out_channels, direction="down"))
            current = out_channels
        self.bridge = _Bridge(channels[-1], channels[-1] * 2)
        decoder_channels = list(reversed(channels))
        self.upconvs = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current = channels[-1] * 2
        for block_id, skip_channels in zip(chromosome[4:], decoder_channels):
            self.upconvs.append(nn.ConvTranspose2d(current, skip_channels, 2, stride=2))
            self.fusions.append(nn.Sequential(nn.Conv2d(skip_channels * 2, skip_channels, 1, bias=False), nn.BatchNorm2d(skip_channels), nn.ReLU(inplace=True)))
            self.decoders.append(build_block(block_id, skip_channels, skip_channels, direction="up"))
            current = skip_channels
        self.head = nn.Conv2d(channels[0], 1, 1)

        # During NSGA-II and scale-selection training selected_scales is None,
        # so every manual block keeps all three scale branches. For final test
        # reconstruction, a saved scale list collapses each block identically
        # to the already-trained final architecture.
        if self.selected_scales is not None:
            from .scale_selection import collapse_model_to_scales
            collapse_model_to_scales(self, self.selected_scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        x = self.stem(x)
        skips: list[torch.Tensor] = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)
        x = self.bridge(x)
        for upconv, fusion, decoder, skip in zip(self.upconvs, self.fusions, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(fusion(torch.cat((skip, x), dim=1)))
        output = self.head(x)
        if output.shape[-2:] != original_size:
            output = functional.interpolate(output, size=original_size, mode="bilinear", align_corners=False)
        return output
