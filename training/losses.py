"""Final-training BCE + soft-mIoU loss and constrained coefficient utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


def soft_miou_loss(logits: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Differentiable mean IoU loss (1 - soft IoU), averaged per image."""
    probabilities = torch.sigmoid(logits).flatten(1)
    targets = targets.float().flatten(1)
    intersection = (probabilities * targets).sum(dim=1)
    union = probabilities.sum(dim=1) + targets.sum(dim=1) - intersection
    soft_iou = (intersection + epsilon) / (union + epsilon)
    return 1.0 - soft_iou.mean()


class BCE_mIoULoss(nn.Module):
    """Weighted BCE + soft-mIoU loss with simplex-constrained coefficients."""

    def __init__(self, a1: float = 0.5, a2: float = 0.5) -> None:
        super().__init__()
        if a1 < 0 or a2 < 0 or abs((a1 + a2) - 1.0) > 1e-8:
            raise ValueError("a1 and a2 must be non-negative and sum to one.")
        self.a1 = float(a1)
        self.a2 = float(a2)

    def components(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bce = functional.binary_cross_entropy_with_logits(logits, targets)
        miou = soft_miou_loss(logits, targets)
        return bce, miou

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce, miou = self.components(logits, targets)
        return self.a1 * bce + self.a2 * miou


def coefficients_from_parameter(raw_a1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map one unconstrained scalar to a1,a2 on the probability simplex."""
    a1 = torch.sigmoid(raw_a1)
    return a1, 1.0 - a1
