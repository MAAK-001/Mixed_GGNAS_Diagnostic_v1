"""IDRiD official train/test discovery with four-lesion union masks."""

from __future__ import annotations

from pathlib import Path


LESIONS = (
    ("1. Microaneurysms", "MA"),
    ("2. Haemorrhages", "HE"),
    ("3. Hard Exudates", "EX"),
    ("4. Soft Exudates", "SE"),
)


def _find_segmentation_root(root: Path) -> Path:
    """Return the exact IDRiD segmentation directory from the supplied layout."""

    idrid_root = root / "IDRID"
    candidates = [
        idrid_root / "A. Segmentation" / "A. Segmentation",
        idrid_root / "A.%20Segmentation" / "A. Segmentation",
        idrid_root / "A.%20Segmentation" / "A.%20Segmentation",
    ]
    for base in candidates:
        image_dir = base / "1. Original Images"
        gt_dir = base / "2. All Segmentation Groundtruths"
        if image_dir.is_dir() and gt_dir.is_dir():
            return base

    raise FileNotFoundError(
        "Expected IDRiD segmentation layout. Tried: "
        + "; ".join(str(path) for path in candidates)
    )


def _find_mask(
    mask_dir: Path,
    image_stem: str,
    suffix: str,
) -> Path | None:
    """Find one lesion mask for an IDRiD image."""

    candidates = sorted(
        mask_dir.glob(
            f"{image_stem}_{suffix}.*"
        )
    )

    return (
        candidates[0]
        if candidates
        else None
    )


def discover_idrid(
    root: Path,
) -> tuple[list[dict[str, str]], dict]:
    """Preserve official IDRiD train/test and union four lesion masks.

    Optic disc is deliberately excluded because this project treats the
    target as retinal lesion segmentation rather than optic-disc segmentation.
    """

    base = _find_segmentation_root(root)

    image_root = (
        base / "1. Original Images"
    )

    mask_root = (
        base
        / "2. All Segmentation Groundtruths"
    )

    records: list[
        dict[str, str]
    ] = []

    missing_optional_masks: dict[
        str,
        list[str],
    ] = {}

    official_counts: dict[
        str,
        int,
    ] = {}

    split_info = (
        (
            "official_train",
            "a. Training Set",
            "a. Training Set",
        ),
        (
            "official_test",
            "b. Testing Set",
            "b. Testing Set",
        ),
    )

    for (
        official_name,
        image_folder,
        mask_folder,
    ) in split_info:

        image_dir = (
            image_root / image_folder
        )

        gt_root = (
            mask_root / mask_folder
        )

        if (
            not image_dir.is_dir()
            or not gt_root.is_dir()
        ):
            raise FileNotFoundError(
                "Missing IDRiD image/ground-truth "
                f"folder for {official_name}: "
                f"{image_dir}"
            )

        images = sorted(
            image_dir.glob("*.jpg")
        )

        official_counts[
            official_name
        ] = len(images)

        for image in images:
            masks: list[str] = []
            missing: list[str] = []

            for (
                lesion_folder,
                suffix,
            ) in LESIONS:

                mask = _find_mask(
                    gt_root / lesion_folder,
                    image.stem,
                    suffix,
                )

                if mask is not None:
                    masks.append(
                        str(mask)
                    )
                else:
                    missing.append(
                        suffix
                    )

            if missing:
                missing_optional_masks[
                    str(image)
                ] = missing

            records.append(
                {
                    "image_path": str(
                        image
                    ),
                    "mask_path": ";".join(
                        masks
                    ),
                    "group": image.stem,
                    "official_split": (
                        official_name
                    ),
                }
            )

    report = {
        "dataset_root": str(base),
        "pairs": len(records),
        "official_counts": official_counts,
        "missing_optional_lesion_masks": (
            missing_optional_masks
        ),
        "target": (
            "Union of Microaneurysms, "
            "Haemorrhages, Hard Exudates "
            "and Soft Exudates; "
            "optic disc excluded."
        ),
        "official_test_preserved": True,
    }

    return records, report