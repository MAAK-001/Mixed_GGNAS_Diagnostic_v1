"""Decode the eight-gene chromosome into a mixed-block U-shaped segmentation model."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from .convnext_block import ConvNeXtBlock
from .darts_block import FixedDownsampleDARTSBlock, FixedUpsampleDARTSBlock
from .dense_block import DenseBlock
from .inception_block import InceptionBlock
from .residual_block import ResidualBlock
from .unet import MixedUNet

BLOCK_NAMES = {0: "Residual", 1: "Dense", 2: "Inception", 3: "ConvNeXt", 4: "DARTS"}


def validate_chromosome(chromosome: Sequence[int]) -> list[int]:
    """Validate the fixed eight-gene, five-block-type representation."""
    values = [int(gene) for gene in chromosome]
    if len(values) != 8 or any(gene not in BLOCK_NAMES for gene in values):
        raise ValueError("A chromosome must contain exactly eight integer genes in {0, 1, 2, 3, 4}.")
    return values


def build_block(block_id: int, in_channels: int, out_channels: int, direction: str) -> nn.Module:
    """Construct a chosen manual block or the fixed DARTS cell for its U-Net position."""
    if block_id == 0:
        return ResidualBlock(in_channels, out_channels)
    if block_id == 1:
        return DenseBlock(in_channels, out_channels)
    if block_id == 2:
        return InceptionBlock(in_channels, out_channels)
    if block_id == 3:
        return ConvNeXtBlock(in_channels, out_channels)
    return FixedDownsampleDARTSBlock(in_channels, out_channels) if direction == "down" else FixedUpsampleDARTSBlock(in_channels, out_channels)


def build_model_from_chromosome(
    chromosome: Sequence[int],
    base_channels: int = 16,
    selected_scales: Sequence[int] | None = None,
) -> MixedUNet:
    """Build a chromosome, optionally reconstructed with saved best scales."""
    return MixedUNet(
        validate_chromosome(chromosome),
        base_channels=base_channels,
        selected_scales=selected_scales,
    )


def describe_architecture(chromosome: Sequence[int]) -> dict[str, list[str] | list[int]]:
    """Return a readable encoder/decoder view for logs, CSV rows and final JSON."""
    values = validate_chromosome(chromosome)
    return {"chromosome": values, "encoder": [BLOCK_NAMES[gene] for gene in values[:4]], "decoder": [BLOCK_NAMES[gene] for gene in values[4:]]}
