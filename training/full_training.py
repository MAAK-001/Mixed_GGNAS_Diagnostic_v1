"""Fresh full-budget retraining and untouched-test evaluation."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from config import ExperimentConfig
from models.scale_selection import (
    collapse_to_selected_scales,
    collect_scale_selections,
    selected_scale_labels,
)
from utils.reproducibility import save_json
from training.losses import BCE_mIoULoss
from training.validation import validate_model


def _optimize_loss_coefficients(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: ExperimentConfig,
) -> tuple[float, float]:
    """Determine a1/a2 with a small bounded one-dimensional optimizer.

    Because a1+a2=1, only a1 needs to be optimized. Each objective evaluation
    trains a fresh copy for a small fixed number of calibration epochs and
    minimizes validation BCE+mIoU loss. All three scale branches remain active
    during this coefficient calibration. The selected coefficients are then
    used for one fresh, full-budget final training run.
    """
    from scipy.optimize import minimize_scalar
    from models.architecture_builder import build_model_from_chromosome
    from utils.reproducibility import set_seed

    chromosome = list(model.chromosome)
    evaluations = []

    def objective(a1: float) -> float:
        a1 = float(a1)
        a2 = 1.0 - a1
        set_seed(config.random_seed)
        trial_model = build_model_from_chromosome(chromosome, config.base_channels).to(config.device)
        criterion = BCE_mIoULoss(a1=a1, a2=a2)
        optimizer = torch.optim.AdamW(
            trial_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=config.use_amp and config.device.type == "cuda",
        )

        for _ in range(config.loss_weight_search_epochs):
            trial_model.train()
            for images, masks in train_loader:
                images = images.to(config.device, dtype=torch.float32, non_blocking=True)
                masks = masks.to(config.device, dtype=torch.float32, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                amp_context = (
                    torch.autocast("cuda", enabled=config.use_amp)
                    if config.device.type == "cuda"
                    else nullcontext()
                )
                with amp_context:
                    loss = criterion(trial_model(images), masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        validation = validate_model(
            trial_model,
            validation_loader,
            config.device,
            config.use_amp,
            a1=a1,
            a2=a2,
        )
        score = float(validation["loss"])
        evaluations.append({
            "a1_BCE": a1,
            "a2_mIoU": a2,
            "validation_loss": score,
        })
        del trial_model
        if config.device.type == "cuda":
            torch.cuda.empty_cache()
        return score

    result = minimize_scalar(
        objective,
        bounds=(config.loss_weight_min, config.loss_weight_max),
        method="bounded",
        options={
            "xatol": config.loss_weight_optimizer_tolerance,
            "maxiter": config.loss_weight_optimizer_max_iterations,
        },
    )
    best_a1 = float(result.x)
    best_a2 = 1.0 - best_a1
    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        config.experiment_dir / "loss_weight_optimization.json",
        {
            "optimizer": "scipy.optimize.minimize_scalar(method='bounded')",
            "a1_bounds": [config.loss_weight_min, config.loss_weight_max],
            "best_a1_BCE": best_a1,
            "best_a2_mIoU": best_a2,
            "best_validation_loss": float(result.fun),
            "evaluations": evaluations,
        },
    )
    return best_a1, best_a2


def _train_epochs(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: BCE_mIoULoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: ExperimentConfig,
    number_of_epochs: int,
    start_epoch: int = 0,
) -> None:
    """Run ordinary supervised training for a specified number of epochs."""
    device = config.device
    for offset in range(number_of_epochs):
        epoch = start_epoch + offset + 1
        model.train()
        for images, masks in loader:
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            masks = masks.to(device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.autocast("cuda", enabled=config.use_amp)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


def train_final_model(
    model: torch.nn.Module,
    chromosome: list[int],
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: ExperimentConfig,
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Train all scales briefly, select the best scale per block, then finish with only those scales.

    The total number of supervised epochs remains ``config.full_epochs``. The
    first ``scale_selection_epochs`` epochs train the original three-scale
    mixtures. Their learned softmax weights determine the winning scale in each
    manual block. The model is then collapsed to those winning branches and a
    fresh optimizer continues the remaining epochs. Testing reconstructs exactly
    this collapsed architecture from the saved selected-scale manifest.
    """
    device = config.device
    model.chromosome = list(chromosome)
    model.to(device)

    # First determine the final-loss coefficients. This calibration itself keeps
    # all three scales active, because scale selection is a separate concern.
    a1, a2 = _optimize_loss_coefficients(model, train_loader, validation_loader, config)

    # Start actual final training from a fresh, deterministic initialization.
    from utils.reproducibility import set_seed
    set_seed(config.random_seed)
    criterion = BCE_mIoULoss(a1=a1, a2=a2)

    total_epochs = int(config.full_epochs)
    if total_epochs < 1:
        raise ValueError("full_epochs must be at least one")

    # Keep the requested total budget fixed. If a very small dry-run budget is
    # used, the scale-selection stage may consume the whole budget; normal runs
    # should leave at least one epoch for selected-scale training.
    scale_epochs = min(int(config.scale_selection_epochs), total_epochs)
    selected_scale_epochs = total_epochs - scale_epochs

    # ---------------------------------------------------------------
    # Stage A: all three scales are active and their mixture weights learn.
    # ---------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.use_amp and device.type == "cuda",
    )

    print(
        f"Scale-selection stage: {scale_epochs}/{total_epochs} epochs "
        "with all three scales active."
    )
    _train_epochs(
        model,
        train_loader,
        criterion,
        optimizer,
        scaler,
        config,
        number_of_epochs=scale_epochs,
        start_epoch=0,
    )

    # Record the learned mixture weights BEFORE removing the losing branches.
    learned_scale_selection = collect_scale_selections(model)
    selected_scales = selected_scale_labels(model)

    print("Selected scale per manual block:")
    for name, entry in learned_scale_selection.items():
        print(
            f"  {name}: scale={entry['selected_scale']} "
            f"weights={entry['weights']}"
        )

    # ---------------------------------------------------------------
    # Stage B: collapse every manual block and train only its winner.
    # ---------------------------------------------------------------
    collapse_to_selected_scales(model)

    # The parameter set changed when two branches + the scale logits were
    # removed, so the old optimizer must never be reused.
    del optimizer
    del scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.use_amp and device.type == "cuda",
    )

    print(
        f"Selected-scale training stage: {selected_scale_epochs}/{total_epochs} "
        "remaining epochs with one scale per manual block."
    )

    best_dice = float("-inf")
    waiting = 0
    checkpoint_path = config.experiment_dir / "best_model.pth"

    if selected_scale_epochs == 0:
        # This branch is mainly useful for tiny structural dry-runs. Normal
        # experiments should use a positive number of selected-scale epochs.
        validation = validate_model(
            model,
            validation_loader,
            device,
            config.use_amp,
            a1=a1,
            a2=a2,
        )
        best_dice = float(validation["dice"])
        torch.save(
            {
                "model_state": model.state_dict(),
                "chromosome": chromosome,
                "validation": validation,
                "loss_coefficients": {"a1_BCE": a1, "a2_mIoU": a2},
                "scale_selection": learned_scale_selection,
                "selected_scales": selected_scales,
                "scale_training": {
                    "all_scales_epochs": scale_epochs,
                    "selected_scale_epochs": selected_scale_epochs,
                    "final_model_has_one_scale_per_manual_block": True,
                },
                "config": config.as_dict(),
            },
            checkpoint_path,
        )
    else:
        for offset in range(selected_scale_epochs):
            epoch = scale_epochs + offset + 1
            model.train()
            for images, masks in train_loader:
                images = images.to(device, dtype=torch.float32, non_blocking=True)
                masks = masks.to(device, dtype=torch.float32, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                amp_context = (
                    torch.autocast("cuda", enabled=config.use_amp)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with amp_context:
                    logits = model(images)
                    loss = criterion(logits, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            validation = validate_model(
                model,
                validation_loader,
                device,
                config.use_amp,
                a1=a1,
                a2=a2,
            )
            print(
                f"Final training epoch {epoch}/{total_epochs}: "
                f"validation Dice {validation['dice']:.4f}"
            )

            if validation["dice"] > best_dice:
                best_dice = float(validation["dice"])
                waiting = 0
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "chromosome": chromosome,
                        "validation": validation,
                        "loss_coefficients": {"a1_BCE": a1, "a2_mIoU": a2},
                        "scale_selection": learned_scale_selection,
                        "selected_scales": selected_scales,
                        "scale_training": {
                            "all_scales_epochs": scale_epochs,
                            "selected_scale_epochs": selected_scale_epochs,
                            "final_model_has_one_scale_per_manual_block": True,
                        },
                        "config": config.as_dict(),
                    },
                    checkpoint_path,
                )
            else:
                waiting += 1

            if (
                config.early_stopping_patience is not None
                and waiting >= config.early_stopping_patience
            ):
                print(f"Early stopping after {epoch} epochs.")
                break

    if not checkpoint_path.exists():
        raise RuntimeError(
            "Final training finished without producing a validation-best checkpoint."
        )

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    validation_result = dict(payload["validation"])
    validation_result["loss_coefficients"] = payload["loss_coefficients"]
    validation_result["scale_selection"] = payload.get("scale_selection", {})
    validation_result["selected_scales"] = payload.get("selected_scales", selected_scales)
    validation_result["scale_training"] = payload.get("scale_training", {})
    return model, validation_result


def test_final_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    config: ExperimentConfig,
    a1: float | None = None,
    a2: float | None = None,
) -> dict[str, float]:
    """Evaluate the selected one-scale-per-block model on untouched test data."""
    return validate_model(
        model,
        test_loader,
        config.device,
        config.use_amp,
        a1=a1,
        a2=a2,
    )
