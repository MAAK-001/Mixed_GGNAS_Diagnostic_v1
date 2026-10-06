"""Binary segmentation metrics with explicit and safe empty-mask behavior."""

from __future__ import annotations

import torch


def _binary_predictions(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).float()


def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, epsilon: float = 1e-7) -> torch.Tensor:
    """Return mean per-image Dice; exact empty prediction/empty target pairs receive Dice 1."""
    predictions = _binary_predictions(logits, threshold).flatten(1)
    targets = targets.float().flatten(1)
    intersection = (predictions * targets).sum(dim=1)
    denominator = predictions.sum(dim=1) + targets.sum(dim=1)
    scores = (2 * intersection + epsilon) / (denominator + epsilon)
    return scores.mean()


def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, epsilon: float = 1e-7) -> torch.Tensor:
    """Return mean per-image IoU; exact empty prediction/empty target pairs receive IoU 1."""
    predictions = _binary_predictions(logits, threshold).flatten(1)
    targets = targets.float().flatten(1)
    intersection = (predictions * targets).sum(dim=1)
    union = predictions.sum(dim=1) + targets.sum(dim=1) - intersection
    return ((intersection + epsilon) / (union + epsilon)).mean()
