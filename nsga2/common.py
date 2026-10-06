"""Shared result structures for the two independent Pymoo NSGA-II paths."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


@dataclass
class Individual:
    chromosome: list[int]
    metrics: dict[str, Any]
    rank: int = 0
    crowding_distance: float = 0.0


def make_individuals(X: np.ndarray, F: np.ndarray, records: dict[tuple[int, ...], dict[str, Any]]) -> list[Individual]:
    ranks = NonDominatedSorting().do(F, return_rank=True)[1]
    result: list[Individual] = []
    for i, x in enumerate(X):
        chromosome = [int(v) for v in x.tolist()]
        result.append(Individual(chromosome, dict(records[tuple(chromosome)]), int(ranks[i]) + 1))
    return result


def pareto_front(population: list[Individual]) -> list[Individual]:
    return [item for item in population if item.rank == 1]


def save_population(population: list[Individual], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "chromosome", "synflow", "synflow_log10", "parameters",
        "third_objective", "third_objective_raw", "third_objective_name", "rank",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in population:
            writer.writerow({
                "chromosome": json.dumps(item.chromosome),
                "synflow": item.metrics["synflow"],
                "synflow_log10": item.metrics["synflow_log10"],
                "parameters": item.metrics["parameters"],
                "third_objective": item.metrics["third_objective"],
                "third_objective_raw": item.metrics["third_objective_raw"],
                "third_objective_name": item.metrics["third_objective_name"],
                "rank": item.rank,
            })


def save_pareto_front(front: list[Individual], path: Path) -> None:
    save_population(front, path)
