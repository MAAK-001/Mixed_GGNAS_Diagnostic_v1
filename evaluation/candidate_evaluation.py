"""Training-free candidate evaluation shared by both NSGA-II branches."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from typing import Any

import torch

from config import ExperimentConfig
from evaluation.model_complexity import count_parameters, estimate_flops
from models.architecture_builder import build_model_from_chromosome, describe_architecture
from search.segmentation_proxies import (
    segmentation_aware_jacobian_covariance,
    spatial_samplewise_activation_proxy,
)
from search.synflow_proxy import synflow_score
from utils.reproducibility import load_json, save_json, set_seed

CACHE_VERSION = 5


def _tensor_fingerprint(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _cache_key(
    chromosome: list[int],
    config: ExperimentConfig,
    proxy_name: str,
    proxy_images: torch.Tensor,
    proxy_masks: torch.Tensor,
) -> str:
    relevant = {
        "cache_version": CACHE_VERSION,
        "proxy": proxy_name,
        "chromosome": chromosome,
        "image_size": list(config.image_size),
        "proxy_image_size": list(proxy_images.shape[-2:]),
        "proxy_images_hash": _tensor_fingerprint(proxy_images),
        "proxy_masks_hash": _tensor_fingerprint(proxy_masks),
        "seed": config.random_seed,
        "base_channels": config.base_channels,
        "device": str(config.device),
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode("utf-8")).hexdigest()


def _append_result(path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "generation", "individual_id", "chromosome", "architecture",
        "synflow", "synflow_log10", "parameters", "flops",
        "third_objective_name", "third_objective_raw", "third_objective",
        "evaluation_seconds", "cached",
    ]
    needs_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerow({field: result.get(field) for field in fields})


def evaluate_candidate(
    chromosome: list[int],
    generation: int,
    individual_id: int,
    config: ExperimentConfig,
    proxy_name: str,
    proxy_images: torch.Tensor,
    proxy_masks: torch.Tensor,
) -> dict[str, Any]:
    """Evaluate SynFlow, parameters, and exactly one segmentation proxy.

    No optimizer step, epoch, validation loss, or test metric is used here.
    """
    cache_path = config.experiment_dir / "evaluation_cache.json"
    cache = load_json(cache_path, {}) if config.cache_enabled else {}
    key = _cache_key(chromosome, config, proxy_name, proxy_images, proxy_masks)

    if key in cache:
        result = dict(cache[key])
        result.update({"generation": generation, "individual_id": individual_id, "cached": True})
        _append_result(config.experiment_dir / "candidate_results.csv", result)
        return result

    set_seed(config.random_seed)
    started = time.perf_counter()

    # SynFlow and parameter count remain exactly the original first two criteria.
    model = build_model_from_chromosome(chromosome, base_channels=config.base_channels)
    parameters = count_parameters(model)
    flops = estimate_flops(model, config.image_size, config.device) if config.calculate_flops else None
    synflow = synflow_score(model, config.image_size, config.device)

    # The segmentation proxy sees the same reduced calibration tensors for every
    # architecture. A fresh model is used and no parameter is updated.
    model = model.to(config.device)
    if proxy_name == "spatial_swap":
        raw_proxy = spatial_samplewise_activation_proxy(model, proxy_images)
    elif proxy_name == "seg_jacobian":
        raw_proxy = segmentation_aware_jacobian_covariance(model, proxy_images, proxy_masks)
    else:
        raise ValueError(f"Unknown proxy: {proxy_name}")

    evaluation_seconds = time.perf_counter() - started

    result: dict[str, Any] = {
        "generation": generation,
        "individual_id": individual_id,
        "chromosome": json.dumps(chromosome),
        "architecture": json.dumps(describe_architecture(chromosome)),
        "synflow": float(synflow.score),
        "synflow_log10": float(synflow.log10_score),
        "parameters": int(parameters),
        "flops": flops,
        "third_objective_name": proxy_name,
        "third_objective_raw": float(raw_proxy),
        # Pymoo minimizes, so the benefit proxy is negated here.
        "third_objective": float(-raw_proxy),
        "evaluation_seconds": float(evaluation_seconds),
        "cached": False,
    }

    if not math.isfinite(result["synflow_log10"]) or not math.isfinite(result["third_objective"]):
        raise RuntimeError(f"Non-finite proxy value for chromosome {chromosome}")

    if config.cache_enabled:
        cache[key] = result
        save_json(cache_path, cache)

    _append_result(config.experiment_dir / "candidate_results.csv", result)

    del model
    if config.device.type == "cuda":
        torch.cuda.empty_cache()
    return result
