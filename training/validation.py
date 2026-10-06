"""Validation and test evaluation for the final BCE + mIoU objective."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from evaluation.metrics import dice_coefficient, iou_score
from training.losses import BCE_mIoULoss


@torch.no_grad()
def validate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    a1: float = 0.5,
    a2: float = 0.5,
) -> dict[str, float]:
    """Evaluate loss, Dice and mIoU without modifying model parameters."""
    if len(loader.dataset) == 0:
        raise ValueError("Validation/test loader is empty")
    if a1 < 0 or a2 < 0 or abs((a1 + a2) - 1.0) > 1e-6:
        raise ValueError("Loss coefficients must be non-negative and sum to one.")

    model.eval()
    criterion = BCE_mIoULoss(a1=a1, a2=a2)
    total_loss = total_dice = total_iou = 0.0
    total_items = 0
    amp_context = torch.autocast("cuda", enabled=use_amp and device.type == "cuda") if device.type == "cuda" else nullcontext()

    for images, masks in loader:
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        masks = masks.to(device, dtype=torch.float32, non_blocking=True)
        with amp_context:
            logits = model(images)
            loss = criterion(logits, masks)
        count = images.shape[0]
        total_loss += float(loss.item()) * count
        total_dice += float(dice_coefficient(logits, masks).item()) * count
        total_iou += float(iou_score(logits, masks).item()) * count
        total_items += count

    return {
        "loss": total_loss / total_items,
        "dice": total_dice / total_items,
        "miou": total_iou / total_items,
    }
