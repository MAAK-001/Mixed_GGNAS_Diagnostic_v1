from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from evaluation.metrics import dice_coefficient, iou_score
from models.architecture_builder import build_model_from_chromosome
from training.losses import BCE_mIoULoss
from utils.reproducibility import set_seed


def split_integrity(manifest_path: Path):
    import csv
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = {"train": set(), "val": set(), "test": set()}
    paths = {"train": set(), "val": set(), "test": set()}
    for r in rows:
        groups[r["split"]].add(r.get("group", r["image_path"]))
        paths[r["split"]].add(r["image_path"])
    return {
        "counts": {k: len(v) for k, v in groups.items()},
        "group_overlap_train_val": len(groups["train"] & groups["val"]),
        "group_overlap_train_test": len(groups["train"] & groups["test"]),
        "group_overlap_val_test": len(groups["val"] & groups["test"]),
        "path_overlap_train_val": len(paths["train"] & paths["val"]),
        "path_overlap_train_test": len(paths["train"] & paths["test"]),
        "path_overlap_val_test": len(paths["val"] & paths["test"]),
    }


def batch_sanity(loader, device):
    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)
    checks = {
        "image_shape": list(images.shape),
        "mask_shape": list(masks.shape),
        "image_dtype": str(images.dtype),
        "mask_dtype": str(masks.dtype),
        "image_finite": bool(torch.isfinite(images).all()),
        "mask_binary": bool(torch.all((masks == 0) | (masks == 1))),
        "mask_positive_fraction": float(masks.mean().item()),
    }
    return checks


def forward_backward_sanity(loader, device, base_channels, seed=42):
    set_seed(seed)
    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)
    model = build_model_from_chromosome([0] * 8, base_channels=base_channels).to(device)
    logits = model(images)
    loss = BCE_mIoULoss(.5, .5)(logits, masks)
    loss.backward()
    grad_norm = torch.sqrt(sum((p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None)))
    return {"logits_shape": list(logits.shape), "loss": float(loss.item()), "loss_finite": bool(torch.isfinite(loss)), "grad_norm": float(grad_norm.item()), "grad_finite": bool(torch.isfinite(grad_norm))}


def tiny_overfit_sanity(loader, device, base_channels, epochs=100, seed=42):
    """Train on eight real training examples only. This diagnoses optimization capacity."""
    set_seed(seed)
    images, masks = next(iter(loader))
    images, masks = images[: min(8, images.shape[0])].to(device), masks[: min(8, masks.shape[0])].to(device)
    model = build_model_from_chromosome([0] * 8, base_channels=base_channels).to(device)
    criterion = BCE_mIoULoss(.5, .5)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = {"dice": -1.0, "miou": -1.0, "loss": float("inf"), "epoch": 0}
    for epoch in range(1, epochs + 1):
        model.train(); opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images); loss = criterion(logits, masks)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        with torch.no_grad():
            d = float(dice_coefficient(logits, masks).item()); m = float(iou_score(logits, masks).item())
        if d > best["dice"]:
            best = {"dice": d, "miou": m, "loss": float(loss.item()), "epoch": epoch}
    return {"n_images": int(images.shape[0]), "epochs": epochs, **best}
