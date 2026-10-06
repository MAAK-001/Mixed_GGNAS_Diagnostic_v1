"""Deterministic, leakage-aware dataset split manifests.

BUSI:
    70% train / 15% validation / 15% test,
    stratified by benign/malignant class.

CVC:
    490-image train pool and 122-image test set,
    followed by 392 train / 98 validation,
    with all splits separated at sequence level.

IDRiD:
    official 54-image training set and 27-image test set,
    with validation carved only from the official training set.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from config import ExperimentConfig
from datasets.busi_dataset import discover_busi
from datasets.cvc_dataset import discover_cvc
from datasets.idrid_dataset import discover_idrid


def _write_manifest(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    """Persist one reproducible dataset split manifest."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "image_path",
        "mask_path",
        "group",
        "split",
        "official_split",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            [
                {
                    field: row.get(
                        field,
                        "",
                    )
                    for field in fields
                }
                for row in rows
            ]
        )


def _stratified_split(
    records: list[dict[str, str]],
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> list[dict[str, str]]:
    """Split BUSI classes independently to preserve class proportions."""

    by_group: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for record in records:
        by_group[
            record["group"]
        ].append(record)

    result: list[
        dict[str, str]
    ] = []

    rng = random.Random(seed)

    for group in sorted(by_group):
        group_records = list(
            by_group[group]
        )

        rng.shuffle(
            group_records
        )

        train_end = round(
            len(group_records)
            * train_fraction
        )

        val_end = (
            train_end
            + round(
                len(group_records)
                * val_fraction
            )
        )

        for index, record in enumerate(
            group_records
        ):
            item = dict(record)

            if index < train_end:
                item["split"] = "train"
            elif index < val_end:
                item["split"] = "val"
            else:
                item["split"] = "test"

            result.append(item)

    return result


def _choose_groups_for_target(
    groups: dict[
        str,
        list[dict[str, str]],
    ],
    target_count: int,
    seed: int,
) -> set[str]:
    """Choose a deterministic group subset with an exact sample count."""

    items = list(groups.items())

    random.Random(seed).shuffle(
        items
    )

    dp: dict[
        int,
        tuple[str, ...],
    ] = {0: ()}

    for group, rows in items:
        size = len(rows)

        for (
            current,
            chosen,
        ) in list(dp.items())[::-1]:

            new_count = (
                current + size
            )

            if (
                new_count <= target_count
                and new_count not in dp
            ):
                dp[new_count] = (
                    chosen + (group,)
                )

        if target_count in dp:
            return set(
                dp[target_count]
            )

    raise ValueError(
        "Could not form an exact "
        f"{target_count}-sample "
        "group split."
    )


def _cvc_split(
    records: list[dict[str, str]],
    seed: int,
) -> list[dict[str, str]]:
    """Create 392/98/122 CVC splits without sequence leakage."""

    if len(records) != 612:
        raise ValueError(
            "CVC expected 612 PNG pairs, "
            f"found {len(records)}."
        )

    groups: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for record in records:
        groups[
            record["group"]
        ].append(record)

    if len(groups) != 29:
        raise ValueError(
            "Expected 29 CVC sequences, "
            f"found {len(groups)}."
        )

    test_groups = (
        _choose_groups_for_target(
            groups,
            122,
            seed,
        )
    )

    train_pool_groups = {
        group
        for group in groups
        if group not in test_groups
    }

    train_pool_by_group = {
        group: groups[group]
        for group in train_pool_groups
    }

    val_groups = (
        _choose_groups_for_target(
            train_pool_by_group,
            98,
            seed + 1,
        )
    )

    result: list[
        dict[str, str]
    ] = []

    for group in sorted(
        train_pool_groups
    ):
        split = (
            "val"
            if group in val_groups
            else "train"
        )

        for record in (
            train_pool_by_group[group]
        ):
            item = dict(record)
            item["split"] = split
            result.append(item)

    for group in sorted(
        test_groups
    ):
        for record in groups[group]:
            item = dict(record)
            item["split"] = "test"
            result.append(item)

    counts = {
        name: sum(
            row["split"] == name
            for row in result
        )
        for name in (
            "train",
            "val",
            "test",
        )
    }

    expected = {
        "train": 392,
        "val": 98,
        "test": 122,
    }

    if counts != expected:
        raise AssertionError(
            f"Unexpected CVC split counts: "
            f"{counts}"
        )

    return result


def _idrid_split(
    records: list[dict[str, str]],
    seed: int,
) -> list[dict[str, str]]:
    """Preserve IDRiD official 54/27 train/test split."""

    train_records = [
        record
        for record in records
        if record.get(
            "official_split"
        )
        == "official_train"
    ]

    test_records = [
        record
        for record in records
        if record.get(
            "official_split"
        )
        == "official_test"
    ]

    if (
        len(train_records) != 54
        or len(test_records) != 27
    ):
        raise ValueError(
            "Expected official IDRiD "
            "54/27 split, found "
            f"{len(train_records)}/"
            f"{len(test_records)}."
        )

    rng = random.Random(seed)

    rng.shuffle(
        train_records
    )

    val_count = round(
        len(train_records) * 0.20
    )

    result: list[
        dict[str, str]
    ] = []

    for index, record in enumerate(
        train_records
    ):
        item = dict(record)

        item["split"] = (
            "val"
            if index < val_count
            else "train"
        )

        result.append(item)

    for record in test_records:
        item = dict(record)
        item["split"] = "test"
        result.append(item)

    return result


def create_or_load_manifest(
    config: ExperimentConfig,
    force: bool = False,
) -> tuple[Path, dict]:
    """Create or reuse one reproducible split manifest."""

    manifest = (
        config.manifests_dir
        / f"{config.dataset.lower()}_split.csv"
    )

    if manifest.exists() and not force:
        with manifest.open(
            newline="",
            encoding="utf-8",
        ) as handle:
            rows = list(
                csv.DictReader(handle)
            )

        return manifest, {
            "reused_manifest": str(
                manifest
            ),
            "split_counts": {
                name: sum(
                    row["split"] == name
                    for row in rows
                )
                for name in (
                    "train",
                    "val",
                    "test",
                )
            },
        }

    if config.dataset == "BUSI":
        records, report = (
            discover_busi(
                config.data_root,
                config.include_busi_normal,
            )
        )

        rows = _stratified_split(
            records,
            config.random_seed,
            0.70,
            0.15,
        )

    elif config.dataset == "CVC":
        records, report = discover_cvc(
            config.data_root
        )

        rows = _cvc_split(
            records,
            config.random_seed,
        )

    else:
        records, report = discover_idrid(
            config.data_root
        )

        rows = _idrid_split(
            records,
            config.random_seed,
        )

    _write_manifest(
        manifest,
        rows,
    )

    report["manifest"] = str(
        manifest
    )

    report["split_counts"] = {
        name: sum(
            row["split"] == name
            for row in rows
        )
        for name in (
            "train",
            "val",
            "test",
        )
    }

    return manifest, report