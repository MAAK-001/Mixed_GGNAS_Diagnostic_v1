from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np

import torch

from evaluation.metrics import dice_coefficient, iou_score
from training.losses import BCE_mIoULoss
from models.architecture_builder import build_model_from_chromosome
from models.scale_selection import ScaleMixture, collapse_model_to_scales
from utils.reproducibility import set_seed


def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum = dice_sum = miou_sum = 0.0
    n = 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            b = images.shape[0]
            loss_sum += loss.item() * b
            dice_sum += dice_coefficient(logits, masks).item() * b
            miou_sum += iou_score(logits, masks).item() * b
            n += b

    if n == 0:
        raise ValueError("Cannot evaluate on an empty dataloader.")

    return {
        "loss": loss_sum / n,
        "dice": dice_sum / n,
        "miou": miou_sum / n,
    }


def _cpu_state_dict(state_dict):
    """Make a CPU copy so checkpoints are independent of the live model."""
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _same_scales(saved, current):
    if saved is None and current is None:
        return True
    if saved is None or current is None:
        return False
    return list(saved) == list(current)


def _normalize_selected_scales(model, hard_selected_scales):
    """Return one scale label per actual ScaleMixture in the model.

    A scalar means: force every ScaleMixture in this architecture to that
    scale. A list must already match the architecture's actual mixture count.
    """
    mixtures = [
        module for module in model.modules()
        if isinstance(module, ScaleMixture)
    ]
    count = len(mixtures)

    if hard_selected_scales is None:
        return None

    if isinstance(hard_selected_scales, int):
        return [int(hard_selected_scales)] * count

    selected = [int(value) for value in hard_selected_scales]
    if len(selected) != count:
        raise ValueError(
            f"Scale-selection length mismatch: model has {count} "
            f"scale mixtures but {len(selected)} selections were provided."
        )
    return selected


def _build_training_model(chromosome, base_channels, hard_selected_scales):
    """Build an uncollapsed model, then optionally force/collapse its scales."""
    model = build_model_from_chromosome(
        chromosome,
        base_channels=base_channels,
    )

    selected = _normalize_selected_scales(model, hard_selected_scales)

    if selected is not None:
        collapse_model_to_scales(model, selected)

    return model, selected


def _checkpoint_payload(
    model,
    optimizer,
    scaler,
    chromosome,
    hard_selected_scales,
    epoch,
    best,
    history,
    best_model_state,
):
    return {
        # Current training state: used to resume from the next epoch.
        "model_state": _cpu_state_dict(model.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": int(epoch),
        # Best validation state: used when the completed experiment is returned.
        "best_model_state": (
            _cpu_state_dict(best_model_state)
            if best_model_state is not None
            else None
        ),
        "best": best,
        "history": history,
        "chromosome": list(chromosome),
        "hard_selected_scales": (
            list(hard_selected_scales)
            if hard_selected_scales is not None
            else None
        ),
        "rng_state": _rng_state(),
    }


def _save_checkpoint(path, payload):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def train_architecture(
    chromosome,
    train_loader,
    val_loader,
    device,
    base_channels,
    epochs,
    seed,
    lr=1e-3,
    weight_decay=5e-5,
    checkpoint: Path | None = None,
    hard_selected_scales: list[int] | int | None = None,
):
    """Train one architecture with epoch-level checkpoint/resume support.

    ``hard_selected_scales`` may be:
      * None: keep the normal three-scale mixtures;
      * an int such as 3/5/7: force every ScaleMixture to that scale;
      * a list: one selected scale per actual ScaleMixture.
    """
    set_seed(seed)

    # Build the architecture consistently with the scale-selection API.
    model, selected_scales = _build_training_model(
        chromosome,
        base_channels,
        hard_selected_scales,
    )
    model = model.to(device)

    criterion = BCE_mIoULoss(0.5, 0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best = None
    best_model_state = None
    history = []
    start_epoch = 1

    # Resume an existing run for this exact architecture/scale configuration.
    if checkpoint is not None and Path(checkpoint).exists():
        payload = _load_checkpoint(checkpoint, device)

        saved_chromosome = payload.get("chromosome")
        saved_scales = payload.get("hard_selected_scales")

        if saved_chromosome is not None and list(saved_chromosome) != list(chromosome):
            raise ValueError(
                f"Checkpoint chromosome mismatch for {checkpoint}: "
                f"saved={saved_chromosome}, current={chromosome}"
            )

        if not _same_scales(saved_scales, selected_scales):
            raise ValueError(
                f"Checkpoint scale-selection mismatch for {checkpoint}: "
                f"saved={saved_scales}, current={selected_scales}"
            )

        history = list(payload.get("history", []))
        best = payload.get("best")

        # New checkpoints contain the true current state. Old checkpoints from
        # the previous code contain only the best model; those remain readable.
        model_state = payload.get("model_state")
        if model_state is not None:
            model.load_state_dict(model_state)

        optimizer_state = payload.get("optimizer_state")
        if optimizer_state is not None:
            try:
                opt.load_state_dict(optimizer_state)
            except (RuntimeError, ValueError):
                # An old/incompatible optimizer state must not invalidate a
                # usable model checkpoint.
                pass

        scaler_state = payload.get("scaler_state")
        if scaler_state is not None:
            try:
                scaler.load_state_dict(scaler_state)
            except (RuntimeError, ValueError):
                pass

        best_model_state = payload.get("best_model_state")
        if best_model_state is None and model_state is not None:
            # Backward compatibility with the old best-only checkpoint format.
            best_model_state = copy.deepcopy(model_state)

        saved_epoch = payload.get("epoch")
        if saved_epoch is None:
            saved_epoch = history[-1]["epoch"] if history else 0

        start_epoch = int(saved_epoch) + 1
        _restore_rng_state(payload.get("rng_state"))

        if history and start_epoch > epochs:
            # Already complete. Return the saved best model without retraining.
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            if best is None:
                last = history[-1]
                best = {
                    "loss": last["val_loss"],
                    "dice": last["val_dice"],
                    "miou": last["val_miou"],
                    "epoch": last["epoch"],
                }
            return model, best, history

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            opt.zero_grad(set_to_none=True)

            with torch.autocast("cuda", enabled=amp):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            b = images.shape[0]
            loss_sum += loss.item() * b
            n += b

        if n == 0:
            raise ValueError("Cannot train on an empty dataloader.")

        val = evaluate(model, val_loader, device, criterion)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / n,
            **{f"val_{k}": v for k, v in val.items()},
        }
        history.append(row)

        if best is None or val["dice"] > best["dice"]:
            best = dict(val)
            best["epoch"] = epoch
            best_model_state = _cpu_state_dict(model.state_dict())

        # Save EVERY epoch, not only epochs that improve validation Dice.
        _save_checkpoint(
            checkpoint,
            _checkpoint_payload(
                model=model,
                optimizer=opt,
                scaler=scaler,
                chromosome=chromosome,
                hard_selected_scales=selected_scales,
                epoch=epoch,
                best=best,
                history=history,
                best_model_state=best_model_state,
            ),
        )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best, history
