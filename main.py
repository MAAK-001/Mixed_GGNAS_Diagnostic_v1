"""Complete two-path NAS pipeline: search, TOPSIS, full training, test, compare."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from comparison import plot_pareto_fronts, write_final_comparison
from config import ExperimentConfig, make_config
from datasets.dataset_utils import create_dataloader, load_manifest
from datasets.split_dataset import create_or_load_manifest
from datasets.verify_datasets import verify_dataset
from models.architecture_builder import build_model_from_chromosome
from nsga2.nsga2_seg_jacobian import run_search as run_jacobian_search
from nsga2.nsga2_spatial_swap import run_search as run_spatial_search
from training.full_training import test_final_model, train_final_model
from utils.reproducibility import save_json, set_seed

SPATIAL_BRANCH = "spatial_swap"
JACOBIAN_BRANCH = "seg_jacobian"


def build_loaders(config: ExperimentConfig, force_split: bool = False):
    """Load the existing manifest-backed train/validation/test splits once."""
    manifest, _ = create_or_load_manifest(config, force=force_split)
    train_rows = load_manifest(manifest, "train")
    val_rows = load_manifest(manifest, "val")
    test_rows = load_manifest(manifest, "test")

    if config.max_train_samples is not None:
        train_rows = train_rows[: config.max_train_samples]
    if config.max_val_samples is not None:
        val_rows = val_rows[: config.max_val_samples]

    return (
        create_dataloader(train_rows, config.image_size, config.batch_size, config.num_workers, training=config.train_augment),
        create_dataloader(val_rows, config.image_size, config.batch_size, config.num_workers, training=False),
        create_dataloader(test_rows, config.image_size, config.batch_size, config.num_workers, training=False),
    )


def branch_config(base: ExperimentConfig, branch: str) -> ExperimentConfig:
    """Create an independent output namespace while retaining every experiment setting."""
    result = copy.deepcopy(base)
    result.run_name = branch
    return result


def selected_chromosome(config: ExperimentConfig) -> list[int]:
    path = config.experiment_dir / "final_architecture.json"
    if not path.exists():
        raise FileNotFoundError(f"No selected architecture at {path}. Run --mode search first.")
    return list(json.loads(path.read_text(encoding="utf-8"))["chromosome"])


def run_search_branch(config: ExperimentConfig, validation_loader, branch: str):
    """Run exactly one branch. This function is intentionally easy to delete later."""
    if branch == SPATIAL_BRANCH:
        return run_spatial_search(config, validation_loader)
    if branch == JACOBIAN_BRANCH:
        return run_jacobian_search(config, validation_loader)
    raise ValueError(f"Unknown branch: {branch}")


def run_final_training(config: ExperimentConfig, train_loader, validation_loader) -> None:
    chromosome = selected_chromosome(config)
    set_seed(config.random_seed)
    model = build_model_from_chromosome(chromosome, config.base_channels)
    _, validation = train_final_model(model, chromosome, train_loader, validation_loader, config)
    save_json(
        config.experiment_dir / "final_training.json",
        {"chromosome": chromosome, "validation": validation},
    )

    # Preserve the NAS-selected architecture record, while adding the scale
    # choices that were learned after TOPSIS selected the chromosome.
    architecture_path = config.experiment_dir / "final_architecture.json"
    if architecture_path.exists():
        architecture_record = json.loads(architecture_path.read_text(encoding="utf-8"))
        architecture_record["selected_scales"] = validation.get("selected_scales", [])
        architecture_record["scale_selection"] = validation.get("scale_selection", {})
        architecture_record["scale_training"] = validation.get("scale_training", {})
        save_json(architecture_path, architecture_record)

    print(f"[{config.run_name}] Final training complete: Dice={validation['dice']:.4f}, mIoU={validation['miou']:.4f}")


def run_final_test(config: ExperimentConfig, test_loader) -> None:
    chromosome = selected_chromosome(config)
    checkpoint = config.experiment_dir / "best_model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No final checkpoint at {checkpoint}. Run --mode train-final first.")

    payload = torch.load(checkpoint, map_location=config.device, weights_only=False)
    selected_scales = payload.get("selected_scales")
    if selected_scales is None:
        # Backward-compatible reconstruction from the saved named scale manifest.
        manifest = payload.get("scale_selection", {})
        selected_scales = [
            int(entry["selected_scale"])
            for entry in manifest.values()
        ]
    model = build_model_from_chromosome(
        chromosome,
        config.base_channels,
        selected_scales=selected_scales,
    ).to(config.device)
    model.load_state_dict(payload["model_state"])
    coefficients = payload.get("loss_coefficients", {"a1_BCE": 0.5, "a2_mIoU": 0.5})
    metrics = test_final_model(
        model,
        test_loader,
        config,
        a1=float(coefficients["a1_BCE"]),
        a2=float(coefficients["a2_mIoU"]),
    )
    save_json(config.experiment_dir / "final_test_results.json", {"chromosome": chromosome, "metrics": metrics, "loss_coefficients": coefficients})
    print(f"[{config.run_name}] Test: Dice={metrics['dice']:.4f}, mIoU={metrics['miou']:.4f}, loss={metrics['loss']:.4f}")


def run_pipeline(config: ExperimentConfig, force_split: bool = False, do_search: bool = True, do_train: bool = True, do_test: bool = True) -> None:
    """Execute both branches under exactly the same base conditions."""
    verify_dataset(config, force_manifest=force_split, save_examples=False)
    train_loader, validation_loader, test_loader = build_loaders(config, force_split=False)

    spatial = branch_config(config, SPATIAL_BRANCH)
    jacobian = branch_config(config, JACOBIAN_BRANCH)

    # PHASE 1 — SEARCH BOTH PATHS. They are sequential, but receive the exact
    # same validation loader and the exact same fixed calibration batch.
    if do_search:
        print("\n=== PHASE 1A: Spatial-SWAP NSGA-II ===")
        run_search_branch(spatial, validation_loader, SPATIAL_BRANCH)
        print("\n=== PHASE 1B: Segmentation-Jacobian NSGA-II ===")
        run_search_branch(jacobian, validation_loader, JACOBIAN_BRANCH)
        comparison_dir = config.experiment_dir / "comparison"
        plot_pareto_fronts(
            spatial.experiment_dir,
            jacobian.experiment_dir,
            comparison_dir / "pareto_front_comparison_3d.png",
        )

    # PHASE 2 — FULL TRAINING, independently selected by each branch's TOPSIS.
    if do_train:
        print("\n=== PHASE 2A: Full training — Spatial-SWAP selection ===")
        run_final_training(spatial, train_loader, validation_loader)
        print("\n=== PHASE 2B: Full training — SA-JacCov selection ===")
        run_final_training(jacobian, train_loader, validation_loader)

    # PHASE 3 — UNTOUCHED TESTING.
    if do_test:
        print("\n=== PHASE 3A: Test — Spatial-SWAP selection ===")
        run_final_test(spatial, test_loader)
        print("\n=== PHASE 3B: Test — SA-JacCov selection ===")
        run_final_test(jacobian, test_loader)

    # Final side-by-side comparison, including one common Pareto-front figure.
    if (spatial.experiment_dir / "final_test_results.json").exists() and (jacobian.experiment_dir / "final_test_results.json").exists():
        comparison_dir = config.experiment_dir / "comparison"
        write_final_comparison(spatial.experiment_dir, jacobian.experiment_dir, comparison_dir)
        print(f"\nComparison written to: {comparison_dir}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mixed-GGNAS two-path training-free NSGA-II pipeline.")
    parser.add_argument("--dataset", required=True, choices=["BUSI", "CVC", "IDRID"])
    parser.add_argument("--mode", required=True, choices=["verify", "dry-run", "search", "train-final", "test", "all"])
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--full-epochs", type=int)
    parser.add_argument("--scale-selection-epochs", type=int)
    parser.add_argument("--population-size", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--force-split", action="store_true")
    parser.add_argument("--include-busi-normal", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = make_config(
        args.dataset,
        data_root=args.data_root,
        output_root=args.output_root,
        full_epochs=args.full_epochs,
        scale_selection_epochs=args.scale_selection_epochs,
        population_size=args.population_size,
        generations=args.generations,
        batch_size=args.batch_size,
        base_channels=args.base_channels,
        random_seed=args.seed,
        include_busi_normal=args.include_busi_normal,
    )
    config.cache_enabled = not args.no_cache
    set_seed(config.random_seed)

    if args.mode == "verify":
        verify_dataset(config, force_manifest=args.force_split, save_examples=False)
        return

    if args.mode == "dry-run":
        config.population_size = 3
        config.generations = 2
        config.full_epochs = 1
        config.loss_weight_search_epochs = 1
        config.scale_selection_epochs = 1
        config.proxy_batch_size = 2
        run_pipeline(config, force_split=args.force_split, do_search=True, do_train=False, do_test=False)
        return

    if args.mode == "search":
        run_pipeline(config, force_split=args.force_split, do_search=True, do_train=False, do_test=False)
    elif args.mode == "train-final":
        run_pipeline(config, do_search=False, do_train=True, do_test=False)
    elif args.mode == "test":
        run_pipeline(config, do_search=False, do_train=False, do_test=True)
    elif args.mode == "all":
        run_pipeline(config, force_split=args.force_split, do_search=True, do_train=True, do_test=True)


if __name__ == "__main__":
    main()
