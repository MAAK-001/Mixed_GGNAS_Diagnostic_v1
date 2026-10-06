"""Preflight verification for real BUSI, CVC and IDRID layouts before NAS is allowed to start."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from config import ExperimentConfig, make_config
from datasets.busi_dataset import discover_busi
from datasets.cvc_dataset import discover_cvc
from datasets.idrid_dataset import discover_idrid
from datasets.split_dataset import create_or_load_manifest
from utils.reproducibility import save_json


def _dimensions(paths: list[str]) -> dict[str, int]:
    """Count image dimensions and modes, recording every unreadable path as an error at the caller."""
    counts: Counter[str] = Counter()
    for raw_path in paths:
        with Image.open(raw_path) as image:
            counts[f"{image.size[0]}x{image.size[1]} {image.mode}"] += 1
    return dict(counts)


def _save_examples(records: list[dict[str, str]], output_path: Path) -> None:
    """Save three image/mask previews so pairing can be inspected without changing source data."""
    try:
        import matplotlib.pyplot as pyplot
    except ImportError:
        return
    selected = records[: min(3, len(records))]
    if not selected:
        return
    figure, axes = pyplot.subplots(len(selected), 2, figsize=(8, 3 * len(selected)))
    axes = np.atleast_2d(axes)
    for row, pair in zip(selected, axes):
        with Image.open(row["image_path"]) as image:
            pair[0].imshow(image.convert("RGB"))
        union = None
        for raw_mask in filter(None, row["mask_path"].split(";")):
            with Image.open(raw_mask) as mask:
                array = np.asarray(mask.convert("L")) > 0
            union = array if union is None else np.logical_or(union, array)
        pair[1].imshow(union if union is not None else np.zeros((1, 1)), cmap="gray", vmin=0, vmax=1)
        pair[0].set_title(Path(row["image_path"]).name)
        pair[1].set_title("paired target mask")
        pair[0].axis("off")
        pair[1].axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    pyplot.close(figure)


def verify_dataset(config: ExperimentConfig, force_manifest: bool = False, save_examples: bool = True) -> dict:
    """Discover, pair, dimension-check and split-check one source dataset; raise on serious errors."""
    if config.dataset == "BUSI":
        records, discovery = discover_busi(config.data_root, config.include_busi_normal)
    elif config.dataset == "CVC":
        records, discovery = discover_cvc(config.data_root)
    else:
        records, discovery = discover_idrid(config.data_root)
    errors: list[str] = []
    if not records:
        errors.append("No image/mask pairs were discovered.")
    if discovery.get("missing_masks"):
        errors.append(f"{len(discovery['missing_masks'])} images have no mask.")
    try:
        image_dimensions = _dimensions([row["image_path"] for row in records])
        mask_dimensions = _dimensions([path for row in records for path in row["mask_path"].split(";") if path])
    except (OSError, ValueError) as error:
        errors.append(f"Unreadable image or mask: {error}")
        image_dimensions, mask_dimensions = {}, {}
    manifest, split_report = create_or_load_manifest(config, force=force_manifest)
    split_rows = sum(1 for _ in manifest.open(encoding="utf-8")) - 1
    if split_rows != len(records):
        errors.append(f"Manifest has {split_rows} rows but discovery has {len(records)} pairs.")
    report = {
        "dataset": config.dataset, "discovery": discovery, "image_dimensions": image_dimensions,
        "mask_dimensions": mask_dimensions, "manifest": str(manifest), "split": split_report,
        "errors": errors, "ready_for_nas": not errors,
    }
    verification_path = config.experiment_dir / "verification_report.json"
    save_json(verification_path, report)
    if save_examples:
        _save_examples(records, config.experiment_dir / "verification_samples.png")
    print(f"{config.dataset}: {len(records)} usable pairs; report: {verification_path}")
    if errors:
        raise RuntimeError("Dataset verification failed: " + "; ".join(errors))
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Verify one source segmentation dataset and save its split manifest.")
    parser.add_argument("--dataset", required=True, choices=["BUSI", "CVC", "IDRID"])
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--force-split", action="store_true")
    arguments = parser.parse_args()
    config = make_config(arguments.dataset, data_root=arguments.data_root, output_root=arguments.output_root)
    verify_dataset(config, force_manifest=arguments.force_split)


if __name__ == "__main__":
    _main()
