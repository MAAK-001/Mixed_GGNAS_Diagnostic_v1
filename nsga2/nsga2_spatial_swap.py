"""NSGA-II branch using the Spatially-Aware Sample-Wise Activation Proxy."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from config import ExperimentConfig
from evaluation.candidate_evaluation import evaluate_candidate
from mcdm.topsis import topsis
from models.architecture_builder import describe_architecture
from nsga2.common import Individual, make_individuals, pareto_front, save_pareto_front, save_population
from search.proxy_data import build_fixed_proxy_batch
from utils.reproducibility import save_json, set_seed


class SpatialSwapProblem(Problem):
    """Eight integer genes, five legal values per gene, and three minimization objectives."""

    def __init__(self) -> None:
        super().__init__(
            n_var=8,
            n_obj=3,
            xl=np.zeros(8, dtype=int),
            xu=np.full(8, 4, dtype=int),
            vtype=int,
        )


def _topsis_select(front: list[Individual], config: ExperimentConfig) -> tuple[Individual, list[dict[str, Any]]]:
    matrix = np.asarray([
        [
            float(item.metrics["third_objective_raw"]),
            float(np.log10(item.metrics["parameters"])),
            float(item.metrics["synflow_log10"]),
        ]
        for item in front
    ], dtype=float)
    # Third objective is the primary benefit; parameter count is a cost;
    # SynFlow is a secondary benefit.
    weights = np.asarray([config.topsis_third_objective_weight, config.topsis_parameter_weight, config.topsis_synflow_weight], dtype=float)
    winner, coefficients, _ = topsis(matrix, weights, np.asarray([True, False, True]))

    ranking = []
    for i, item in enumerate(front):
        ranking.append({
            "chromosome": json.dumps(item.chromosome),
            "third_objective": float(item.metrics["third_objective_raw"]),
            "parameters": int(item.metrics["parameters"]),
            "parameters_million": float(item.metrics["parameters"]) / 1e6,
            "synflow_log10": float(item.metrics["synflow_log10"]),
            "topsis_coefficient": float(coefficients[i]),
        })
    ranking.sort(key=lambda row: row["topsis_coefficient"], reverse=True)
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank

    path = config.experiment_dir / "topsis_ranking.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranking[0].keys()))
        writer.writeheader()
        writer.writerows(ranking)
    return front[winner], ranking


def run_search(config: ExperimentConfig, validation_loader) -> Individual:
    """Run the complete Spatial-SWAP NSGA-II path and apply its own TOPSIS."""
    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.random_seed)
    proxy_images, proxy_masks = build_fixed_proxy_batch(
        validation_loader,
        config.device,
        image_size=config.proxy_image_size,
        batch_size=config.proxy_batch_size,
    )
    proxy_images = proxy_images.detach()
    proxy_masks = proxy_masks.detach()

    problem = SpatialSwapProblem()
    algorithm = NSGA2(
        pop_size=config.population_size,
        n_offsprings=config.population_size,
        sampling=IntegerRandomSampling(),
        crossover=UniformCrossover(prob=config.crossover_probability),
        # Pymoo's integer polynomial mutation + rounding repair keeps every
        # gene discrete while avoiding a custom mutation implementation.
        mutation=PolynomialMutation(
            prob=1.0,
            prob_var=config.mutation_probability,
            repair=RoundingRepair(),
        ),
        eliminate_duplicates=True,
    )
    algorithm.setup(problem, termination=("n_gen", config.generations), seed=config.random_seed, verbose=False)

    records: dict[tuple[int, ...], dict[str, Any]] = {}
    generation = 0
    last_population = None

    while algorithm.has_next():
        infills = algorithm.ask()
        for individual_id, solution in enumerate(infills, start=1):
            chromosome = [int(v) for v in np.asarray(solution.X).tolist()]
            result = evaluate_candidate(
                chromosome, generation, individual_id, config,
                "spatial_swap",
                proxy_images,
                proxy_masks,
            )
            records[tuple(chromosome)] = dict(result)
            solution.F = np.asarray([
                -float(result["synflow_log10"]),
                float(result["parameters"]),
                float(result["third_objective"]),
            ], dtype=float)

        algorithm.tell(infills)
        last_population = algorithm.pop
        X = np.asarray(last_population.get("X"), dtype=int)
        F = np.asarray(last_population.get("F"), dtype=float)
        front_indices = NonDominatedSorting().do(F, only_non_dominated_front=True)

        np.savez(
            config.experiment_dir / f"generation_{generation:03d}.npz",
            X=X, F=F,
        )
        save_json(config.experiment_dir / "config.json", config.as_dict())
        print(
            f"[Spatial-SWAP] Generation {generation}: "
            f"population={len(X)} | Pareto={len(front_indices)} | "
            f"best SA-SWAP={-float(np.min(F[:, 2])):.6f}"
        )
        generation += 1

    if last_population is None:
        raise RuntimeError("Spatial-SWAP NSGA-II returned no population.")

    X = np.asarray(last_population.get("X"), dtype=int)
    F = np.asarray(last_population.get("F"), dtype=float)
    population = make_individuals(X, F, records)
    front = pareto_front(population)
    save_population(population, config.experiment_dir / "final_population.csv")
    save_pareto_front(front, config.experiment_dir / "pareto_front.csv")

    selected, ranking = _topsis_select(front, config)
    selected_row = next(row for row in ranking if json.loads(row["chromosome"]) == selected.chromosome)
    save_json(
        config.experiment_dir / "final_architecture.json",
        {
            **describe_architecture(selected.chromosome),
            "proxy": "Spatially-Aware Sample-Wise Activation Proxy",
            "objectives": ["maximize SynFlow", "minimize trainable parameters", "maximize SA-SWAP"],
            "topsis_weights": {"third_objective": 0.50, "parameters": 0.30, "synflow": 0.20},
            "topsis_coefficient": selected_row["topsis_coefficient"],
            "metrics": selected.metrics,
        },
    )
    print(f"[Spatial-SWAP] TOPSIS selected: {selected.chromosome}")
    return selected
