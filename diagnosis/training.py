from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from evaluation.metrics import dice_coefficient, iou_score
from training.losses import BCE_mIoULoss
from models.architecture_builder import build_model_from_chromosome
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
    return {"loss": loss_sum / n, "dice": dice_sum / n, "miou": miou_sum / n}


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
    hard_selected_scales: list[int] | None = None,
):
    set_seed(seed)
    model = build_model_from_chromosome(chromosome, base_channels=base_channels, selected_scales=hard_selected_scales).to(device)
    criterion = BCE_mIoULoss(0.5, 0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best = None
    history = []
    for epoch in range(1, epochs + 1):
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
        val = evaluate(model, val_loader, device, criterion)
        row = {"epoch": epoch, "train_loss": loss_sum / n, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        if best is None or val["dice"] > best["dice"]:
            best = dict(val)
            best["epoch"] = epoch
            if checkpoint:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state": model.state_dict(), "chromosome": chromosome, "best": best, "history": history, "hard_selected_scales": hard_selected_scales}, checkpoint)
    return model, best, history
