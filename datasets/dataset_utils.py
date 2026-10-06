"""Common manifest-backed image/mask loading and preprocessing for binary segmentation."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset


IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def load_manifest(
    path: Path,
    split: str | None = None,
) -> list[dict[str, str]]:
    """Read a persisted split manifest and optionally keep one named split."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    return [
        row
        for row in rows
        if split is None or row["split"] == split
    ]


class SegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Loads RGB images and one or more binary masks from a split manifest."""

    def __init__(
        self,
        rows: Iterable[dict[str, str]],
        image_size: tuple[int, int],
        training: bool = False,
    ) -> None:
        self.rows = list(rows)
        self.image_size = image_size
        self.training = training

    def __len__(self) -> int:
        return len(self.rows)

    def _load_union_mask(
        self,
        raw_paths: str,
        original_size: tuple[int, int],
    ) -> Image.Image:
        """
        Union all masks listed in the manifest.

        Multiple mask paths are separated by ';'.
        Missing optional masks contribute only background.
        """
        union = np.zeros(
            (original_size[1], original_size[0]),
            dtype=np.uint8,
        )

        for raw_path in filter(None, raw_paths.split(";")):
            path = Path(raw_path)

            if not path.exists():
                continue

            with Image.open(path) as mask_image:
                mask = np.asarray(
                    mask_image.convert("L"),
                    dtype=np.uint8,
                )

            if mask.shape != union.shape:
                raise ValueError(
                    f"Mask shape differs from image: {path}. "
                    f"Expected {union.shape}, got {mask.shape}."
                )

            union = np.maximum(
                union,
                (mask > 0).astype(np.uint8) * 255,
            )

        return Image.fromarray(union, mode="L")

    def _augment(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        """Apply paired spatial augmentation and image-only brightness jitter."""

        if random.random() < 0.5:
            image = image.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )
            mask = mask.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )

        if random.random() < 0.5:
            image = image.transpose(
                Image.Transpose.FLIP_TOP_BOTTOM
            )
            mask = mask.transpose(
                Image.Transpose.FLIP_TOP_BOTTOM
            )

        if random.random() < 0.25:
            image = ImageEnhance.Brightness(image).enhance(
                random.uniform(0.9, 1.1)
            )

        return image, mask

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        row = self.rows[index]

        # ---------------------------------------------------------
        # Load image
        # ---------------------------------------------------------
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")

        # ---------------------------------------------------------
        # Load and union masks
        # ---------------------------------------------------------
        mask = self._load_union_mask(
            row["mask_path"],
            image.size,
        )

        # ---------------------------------------------------------
        # Training augmentation
        # ---------------------------------------------------------
        if self.training:
            image, mask = self._augment(
                image,
                mask,
            )

        # ---------------------------------------------------------
        # Resize
        # ---------------------------------------------------------
        height, width = self.image_size

        image = image.resize(
            (width, height),
            Image.Resampling.BILINEAR,
        )

        mask = mask.resize(
            (width, height),
            Image.Resampling.NEAREST,
        )

        # ---------------------------------------------------------
        # Image → NumPy float32
        # ---------------------------------------------------------
        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

        image_array /= np.float32(255.0)

        # Explicitly construct float32 normalization arrays.
        mean = np.asarray(
            IMAGE_MEAN,
            dtype=np.float32,
        ).reshape(1, 1, 3)

        std = np.asarray(
            IMAGE_STD,
            dtype=np.float32,
        ).reshape(1, 1, 3)

        image_array = (
            image_array - mean
        ) / std

        # ---------------------------------------------------------
        # Mask → NumPy float32
        # ---------------------------------------------------------
        mask_array = np.asarray(
            mask,
            dtype=np.float32,
        )

        mask_array = (
            mask_array > np.float32(0.0)
        ).astype(np.float32)

        # ---------------------------------------------------------
        # NumPy → PyTorch
        #
        # Explicit dtype enforcement is intentional.
        # This guarantees that every dataset returns:
        #
        #     image.dtype == torch.float32
        #     mask.dtype  == torch.float32
        # ---------------------------------------------------------
        image_tensor = torch.from_numpy(
            image_array.transpose(2, 0, 1)
        ).to(dtype=torch.float32)

        mask_tensor = torch.from_numpy(
            mask_array[None, ...]
        ).to(dtype=torch.float32)

        return image_tensor, mask_tensor


def create_dataloader(
    rows: list[dict[str, str]],
    image_size: tuple[int, int],
    batch_size: int,
    num_workers: int,
    training: bool,
) -> DataLoader:
    """Create a consistently configured loader for one manifest split."""

    return DataLoader(
        SegmentationDataset(
            rows,
            image_size=image_size,
            training=training,
        ),
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )