"""Build one fixed calibration batch shared by both training-free NAS paths."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


def build_fixed_proxy_batch(
    loader: DataLoader,
    device: torch.device,
    image_size: tuple[int, int] = (64, 64),
    batch_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the first fixed, non-augmented calibration batch.

    The caller supplies the validation loader, which is deterministic and has
    augmentation disabled in the existing pipeline. The same tensors are then
    reused by both NSGA-II branches, so the two third objectives are compared
    under exactly identical inputs.
    """
    if batch_size < 2:
        raise ValueError("The proxy batch must contain at least two images.")

    iterator = iter(loader)
    images, masks = next(iterator)
    images = images[:batch_size]
    masks = masks[:batch_size]

    if images.shape[0] < 2:
        raise ValueError("The validation split contains fewer than two proxy images.")

    images = F.interpolate(images, size=image_size, mode="bilinear", align_corners=False)
    masks = F.interpolate(masks, size=image_size, mode="nearest")

    return images.to(device=device, dtype=torch.float32), masks.to(device=device, dtype=torch.float32)
