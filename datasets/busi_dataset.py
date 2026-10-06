"""BUSI image/mask discovery with correct multi-mask union handling."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def discover_busi(
    root: Path,
    include_normal: bool = False,
) -> tuple[list[dict[str, str]], dict]:
    """Discover BUSI cases and retain every mask annotation.

    Multiple masks belonging to one image are stored in mask_path separated
    by ';'. SegmentationDataset later performs their pixelwise OR.

    Normal images are excluded by default because this project's main BUSI
    population is the 647 benign + malignant cases.
    """

    dataset_root = (
        root
        / "BUSI (Breast Ultrasound Image)"
    )

    labels = ["benign", "malignant"]

    if include_normal:
        labels.append("normal")

    records: list[dict[str, str]] = []
    missing_masks: list[str] = []
    duplicate_masks: dict[
        str,
        list[str],
    ] = {}

    for label in labels:
        directory = dataset_root / label

        if not directory.is_dir():
            raise FileNotFoundError(
                f"BUSI class directory not found: {directory}"
            )

        images = sorted(
            path
            for path in directory.glob("*.png")
            if "_mask" not in path.stem.lower()
        )

        masks_by_base: dict[
            str,
            list[Path],
        ] = defaultdict(list)

        for mask in directory.glob(
            "*_mask*.png"
        ):
            base = mask.stem.split(
                "_mask",
                1,
            )[0]

            masks_by_base[base].append(mask)

        for image in images:
            candidates = sorted(
                masks_by_base.get(
                    image.stem,
                    [],
                ),
                key=lambda item: item.name,
            )

            if not candidates:
                if label == "normal":
                    records.append(
                        {
                            "image_path": str(image),
                            "mask_path": "",
                            "group": label,
                        }
                    )
                    continue

                missing_masks.append(
                    str(image)
                )
                continue

            if len(candidates) > 1:
                duplicate_masks[
                    str(image)
                ] = [
                    str(item)
                    for item in candidates
                ]

            records.append(
                {
                    "image_path": str(image),
                    "mask_path": ";".join(
                        str(item)
                        for item in candidates
                    ),
                    "group": label,
                }
            )

    report = {
        "dataset_root": str(dataset_root),
        "pairs": len(records),
        "missing_masks": missing_masks,
        "duplicate_masks": duplicate_masks,
        "included_labels": labels,
        "normal_excluded": not include_normal,
        "multiple_masks_union": True,
    }

    return records, report