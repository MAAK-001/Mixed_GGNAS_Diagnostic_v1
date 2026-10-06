"""CVC-ClinicDB discovery using PNG images and sequence-aware grouping."""

from __future__ import annotations

import csv
from pathlib import Path


# Official CVC-ClinicDB frame-to-sequence boundaries.
_SEQUENCE_ENDS = [
    25,
    50,
    67,
    78,
    103,
    126,
    151,
    177,
    199,
    205,
    227,
    252,
    277,
    297,
    317,
    342,
    363,
    383,
    408,
    428,
    447,
    466,
    478,
    503,
    528,
    546,
    571,
    591,
    612,
]


def _sequence_from_frame(
    frame_number: int,
) -> str:
    """Map a CVC frame number to its video sequence."""

    for sequence_id, end in enumerate(
        _SEQUENCE_ENDS,
        start=1,
    ):
        if frame_number <= end:
            return f"sequence_{sequence_id:02d}"

    raise ValueError(
        f"CVC frame number {frame_number} "
        "is outside the expected 1..612 range"
    )


def _metadata_sequence_map(
    dataset_root: Path,
) -> dict[str, str]:
    """Read sequence IDs from an extracted metadata.csv when available."""

    candidates = list(
        dataset_root.rglob("metadata.csv")
    )

    candidates += list(
        dataset_root.rglob(
            "metadata*.csv"
        )
    )

    for metadata_path in candidates:
        try:
            with metadata_path.open(
                newline="",
                encoding="utf-8-sig",
            ) as handle:
                reader = csv.DictReader(
                    handle
                )

                fields = {
                    field.lower().strip()
                    for field in (
                        reader.fieldnames
                        or []
                    )
                }

                if "sequence_id" not in fields:
                    continue

                mapping: dict[
                    str,
                    str,
                ] = {}

                for row in reader:
                    name = (
                        row.get("frame_id")
                        or row.get("image")
                        or row.get("image_path")
                        or row.get("filename")
                    )

                    sequence = row.get(
                        "sequence_id"
                    )

                    if name and sequence:
                        stem = Path(
                            str(name)
                        ).stem

                        mapping[stem] = str(
                            sequence
                        )

                if mapping:
                    return mapping

        except (
            OSError,
            UnicodeError,
            csv.Error,
        ):
            continue

    return {}


def discover_cvc(
    root: Path,
) -> tuple[list[dict[str, str]], dict]:
    """Pair CVC PNG images/masks and attach originating video sequence."""

    dataset_root = (
        root
        / "CVC-ClinicD (Polyp)"
    )

    image_dir = (
        dataset_root
        / "PNG"
        / "Original"
    )

    mask_dir = (
        dataset_root
        / "PNG"
        / "Ground Truth"
    )

    if (
        not image_dir.is_dir()
        or not mask_dir.is_dir()
    ):
        raise FileNotFoundError(
            "CVC PNG Original/Ground Truth "
            f"directories not found under {dataset_root}"
        )

    images = {
        path.name: path
        for path in image_dir.glob("*.png")
    }

    masks = {
        path.name: path
        for path in mask_dir.glob("*.png")
    }

    names = sorted(
        set(images) & set(masks),
        key=lambda name: (
            int(Path(name).stem)
            if Path(name).stem.isdigit()
            else name
        ),
    )

    sequence_map = _metadata_sequence_map(
        dataset_root
    )

    records: list[
        dict[str, str]
    ] = []

    for name in names:
        stem = Path(name).stem

        if stem in sequence_map:
            group = (
                f"sequence_{sequence_map[stem]}"
            )
        elif stem.isdigit():
            group = _sequence_from_frame(
                int(stem)
            )
        else:
            raise ValueError(
                "Cannot determine CVC "
                f"sequence for frame: {name}"
            )

        records.append(
            {
                "image_path": str(
                    images[name]
                ),
                "mask_path": str(
                    masks[name]
                ),
                "group": group,
            }
        )

    report = {
        "dataset_root": str(dataset_root),
        "pairs": len(records),
        "missing_masks": sorted(
            set(images) - set(masks)
        ),
        "orphan_masks": sorted(
            set(masks) - set(images)
        ),
        "format": (
            "PNG "
            "(TIF intentionally excluded "
            "as duplicate copies)"
        ),
        "sequence_aware_grouping": True,
        "sequence_source": (
            "metadata.csv"
            if sequence_map
            else "official frame-number "
            "sequence ranges"
        ),
        "sequence_count": len(
            {
                row["group"]
                for row in records
            }
        ),
    }

    return records, report