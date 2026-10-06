from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ExperimentConfig
from datasets.dataset_utils import create_dataloader, load_manifest
from datasets.split_dataset import create_or_load_manifest
from datasets.verify_datasets import verify_dataset
from models.architecture_builder import BLOCK_NAMES, build_model_from_chromosome
from models.scale_selection import collect_scale_selections, collapse_to_selected_scales, selected_scale_labels
from evaluation.model_complexity import count_parameters
from diagnosis.config import DiagnosticConfig
from diagnosis.plotting import scatter_all
from diagnosis.proxies import score_all
from diagnosis.stats import correlation_table
from diagnosis.training import evaluate, train_architecture
from diagnosis.sanity import split_integrity, batch_sanity, forward_backward_sanity, tiny_overfit_sanity
from utils.reproducibility import set_seed, save_json


def parse():
    p = argparse.ArgumentParser(description="Real-data diagnostic suite for Mixed-GGNAS NAS.")
    p.add_argument("--dataset", required=True, choices=["BUSI", "CVC", "IDRID"])
    p.add_argument("--data-root", type=Path)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--architectures", type=int, default=24)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--proxy-images", type=int, default=32)
    p.add_argument("--proxy-repeats", type=int, default=3)
    p.add_argument("--proxy-size", type=int, default=64)
    p.add_argument("--scale-ablation-k", type=int, default=5)
    p.add_argument("--scale-ablation-epochs", type=int, default=50)
    p.add_argument("--baseline-epochs", type=int, default=100)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--force-split", action="store_true")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--skip-scale-ablation", action="store_true")
    p.add_argument("--no-resume", action="store_true", help="Ignore existing diagnostic outputs/checkpoints and start fresh.")
    return p.parse_args()


def arch_seed(base, chrom):
    value = base
    for g in chrom:
        value = (value * 131 + int(g) + 17) % (2**31 - 1)
    return value


def make_architectures(n, seed):
    rng = random.Random(seed)
    seen = set()
    out = []
    # Include interpretable extremes first.
    for gene in range(5):
        c = [gene] * 8
        seen.add(tuple(c)); out.append(c)
    while len(out) < n:
        c = [rng.randrange(5) for _ in range(8)]
        if tuple(c) not in seen:
            seen.add(tuple(c)); out.append(c)
    return out[:n]


def loaders(cfg, manifest):
    tr = load_manifest(manifest, "train")
    va = load_manifest(manifest, "val")
    te = load_manifest(manifest, "test")
    base = ExperimentConfig(cfg.dataset)
    base.data_root = cfg.data_root
    base.image_size = cfg.image_size
    base.batch_size = cfg.batch_size
    base.num_workers = cfg.num_workers
    base.train_augment = cfg.train_augment
    return (
        create_dataloader(tr, cfg.image_size, cfg.batch_size, cfg.num_workers, True),
        create_dataloader(va, cfg.image_size, cfg.batch_size, cfg.num_workers, False),
        create_dataloader(te, cfg.image_size, cfg.batch_size, cfg.num_workers, False),
    )


def collect_real_proxy_batch(loader, n, size, device):
    images, masks = [], []
    for x, y in loader:
        images.append(x); masks.append(y)
        if sum(t.shape[0] for t in images) >= n:
            break
    x = torch.cat(images)[:n]
    y = torch.cat(masks)[:n]
    if x.shape[0] < n:
        raise ValueError(f"Requested {n} real proxy images but only {x.shape[0]} are available in the proxy source split")
    x = torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)
    y = torch.nn.functional.interpolate(y, size=size, mode="nearest")
    return x.to(device), y.to(device)


def model_factory(chromosome, cfg, seed):
    def factory():
        set_seed(seed)
        return build_model_from_chromosome(chromosome, base_channels=cfg.base_channels)
    return factory



def _load_csv_if_valid(path, required_columns=()):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if any(column not in df.columns for column in required_columns):
        return None
    return df


def _row_key(chromosome, mode=None):
    key = json.dumps([int(x) for x in chromosome])
    return (key, mode) if mode is not None else key


def _save_rows(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)



def proxy_stage(cfg, architectures, proxy_images, proxy_masks, out, resume=True):
    out = Path(out)
    result_path = out / "proxy_scores.csv"

    existing = _load_csv_if_valid(
        result_path,
        required_columns=("arch_id", "chromosome"),
    ) if resume else None

    existing_by_id = {}
    if existing is not None:
        for _, row in existing.iterrows():
            existing_by_id[int(row["arch_id"])] = row.to_dict()

    # Regularized SWAP calibration uses the observed parameter distribution.
    params = [
        count_parameters(
            build_model_from_chromosome(c, base_channels=cfg.base_channels)
        )
        for c in architectures
    ]
    mu = float(np.mean(params))
    sigma = float(np.std(params) if np.std(params) > 0 else 1.0)

    rows = []
    for i, c in enumerate(architectures):
        if i in existing_by_id:
            rows.append(existing_by_id[i])
            print(f"Proxy {i+1}/{len(architectures)}: RESUME/SKIP")
            continue

        seed = arch_seed(cfg.seed, c)
        factory = model_factory(c, cfg, seed)
        start_time = time.perf_counter()
        scores_accum = []

        for rep in range(cfg.proxy_repeats):
            g = torch.Generator(device=proxy_images.device).manual_seed(
                cfg.seed + 1000 * rep
            )
            idx = torch.randperm(
                proxy_images.shape[0],
                generator=g,
                device=proxy_images.device,
            )[:max(8, min(32, proxy_images.shape[0]))]
            xi, yi = proxy_images[idx], proxy_masks[idx]
            scores_accum.append(
                score_all(factory, xi, yi, params[i], mu, sigma)
            )

        keys = scores_accum[0].keys()
        row = {
            "arch_id": i,
            "chromosome": json.dumps(c),
            "architecture": json.dumps({
                "encoder": [BLOCK_NAMES[g] for g in c[:4]],
                "decoder": [BLOCK_NAMES[g] for g in c[4:]],
            }),
            "parameters": params[i],
            "proxy_seconds": time.perf_counter() - start_time,
        }
        for k in keys:
            if k == "parameters":
                continue
            vals = [s[k] for s in scores_accum if np.isfinite(s[k])]
            row[k] = float(np.mean(vals)) if vals else float("nan")
            row[k + "_std"] = float(np.std(vals)) if vals else float("nan")

        rows.append(row)
        _save_rows(rows, result_path)
        print(f"Proxy {i+1}/{len(architectures)}: {c}")

    result = pd.DataFrame(rows).sort_values("arch_id").reset_index(drop=True)
    result.to_csv(result_path, index=False)
    return result



def train_stage(cfg, architectures, train_loader, val_loader, out, resume=True):
    out = Path(out)
    (out / "histories").mkdir(parents=True, exist_ok=True)
    result_path = out / "calibration_results.csv"

    existing = _load_csv_if_valid(
        result_path,
        required_columns=("arch_id", "chromosome"),
    ) if resume else None

    existing_by_id = {}
    if existing is not None:
        for _, row in existing.iterrows():
            existing_by_id[int(row["arch_id"])] = row.to_dict()

    rows = []
    for i, c in enumerate(architectures):
        if i in existing_by_id:
            rows.append(existing_by_id[i])
            print(f"Train {i+1}/{len(architectures)}: RESUME/SKIP")
            continue

        seed = arch_seed(cfg.seed, c)
        ckpt = out / "calibration_checkpoints" / f"arch_{i:03d}.pth"
        start_time = time.perf_counter()

        _, best, history = train_architecture(
            c,
            train_loader,
            val_loader,
            cfg.device,
            cfg.base_channels,
            cfg.calibration_epochs,
            seed,
            cfg.learning_rate,
            cfg.weight_decay,
            ckpt,
        )

        pd.DataFrame(history).to_csv(
            out / "histories" / f"arch_{i:03d}.csv",
            index=False,
        )

        rows.append({
            "arch_id": i,
            "chromosome": json.dumps(c),
            "seed": seed,
            "val_loss": best["loss"],
            "val_dice": best["dice"],
            "val_miou": best["miou"],
            "best_epoch": best["epoch"],
            "train_seconds": time.perf_counter() - start_time,
        })
        _save_rows(rows, result_path)

        print(
            f"Train {i+1}/{len(architectures)}: "
            f"Dice={best['dice']:.4f}, mIoU={best['miou']:.4f}"
        )

    return pd.DataFrame(rows).sort_values("arch_id").reset_index(drop=True)



def baseline_stage(
    cfg,
    architectures,
    train_loader,
    val_loader,
    test_loader,
    out,
    resume=True,
):
    # Diagnostic only: fixed canonical architectures are trained with the
    # complete three-scale mixture for the full budget. Test metrics are
    # reported, not used to choose a proxy or architecture.
    chosen = architectures[:5]
    result_path = Path(out) / "baseline_test_results.csv"

    existing = _load_csv_if_valid(
        result_path,
        required_columns=("arch_id", "chromosome"),
    ) if resume else None

    existing_by_id = {}
    if existing is not None:
        for _, row in existing.iterrows():
            existing_by_id[int(row["arch_id"])] = row.to_dict()

    rows = []
    for i, c in enumerate(chosen):
        if i in existing_by_id:
            rows.append(existing_by_id[i])
            print(f"Baseline {i+1}/5: RESUME/SKIP")
            continue

        seed = arch_seed(cfg.seed + 10000, c)
        ckpt = Path(out) / "baseline_checkpoints" / f"arch_{i:03d}.pth"

        _, best, _ = train_architecture(
            c,
            train_loader,
            val_loader,
            cfg.device,
            cfg.base_channels,
            cfg.baseline_epochs,
            seed,
            cfg.learning_rate,
            cfg.weight_decay,
            ckpt,
        )

        payload = torch.load(
            ckpt,
            map_location=cfg.device,
            weights_only=False,
        )
        model = build_model_from_chromosome(
            c,
            base_channels=cfg.base_channels,
        ).to(cfg.device)
        model.load_state_dict(
            payload.get("best_model_state", payload["model_state"])
        )

        criterion = __import__(
            "training.losses",
            fromlist=["BCE_mIoULoss"],
        ).BCE_mIoULoss(0.5, 0.5)

        test = evaluate(model, test_loader, cfg.device, criterion)

        rows.append({
            "arch_id": i,
            "chromosome": json.dumps(c),
            "val_best_dice": best["dice"],
            "val_best_miou": best["miou"],
            "test_dice": test["dice"],
            "test_miou": test["miou"],
        })
        _save_rows(rows, result_path)

        print(
            f"Baseline {i+1}/5: val Dice={best['dice']:.4f}; "
            f"TEST Dice={test['dice']:.4f}"
        )

    return pd.DataFrame(rows).sort_values("arch_id").reset_index(drop=True)



def train_current_scale_collapse(
    chromosome,
    train_loader,
    val_loader,
    cfg,
    seed,
    epochs,
    checkpoint=None,
):
    """Reproduce the current 10-epoch argmax-collapse strategy.

    The warm mixture phase and the post-collapse phase are both resumable.
    """
    from training.losses import BCE_mIoULoss

    set_seed(seed)
    checkpoint = Path(checkpoint) if checkpoint is not None else None
    warm = min(10, epochs)

    def build_uncollapsed():
        return build_model_from_chromosome(
            chromosome,
            base_channels=cfg.base_channels,
        ).to(cfg.device)

    model = build_uncollapsed()
    criterion = BCE_mIoULoss(0.5, 0.5)
    amp = cfg.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    phase = "warm"
    warm_epoch = 0
    collapsed_epoch = warm
    labels = None
    manifest = None
    history = []
    best = None
    best_model_state = None

    if checkpoint is not None and checkpoint.exists():
        payload = torch.load(
            checkpoint,
            map_location=cfg.device,
            weights_only=False,
        )

        if list(payload.get("chromosome", chromosome)) != list(chromosome):
            raise ValueError(f"Current-scale checkpoint chromosome mismatch: {checkpoint}")

        phase = payload.get("phase", "warm")
        labels = payload.get("labels")
        manifest = payload.get("manifest")
        history = list(payload.get("history", []))
        best = payload.get("best")
        best_model_state = payload.get("best_model_state")

        if phase == "collapsed":
            if labels is None:
                raise ValueError(
                    f"Collapsed checkpoint {checkpoint} has no selected-scale labels."
                )
            from models.scale_selection import collapse_model_to_scales
            collapse_model_to_scales(model, list(labels))
            model.load_state_dict(payload["model_state"])

            opt = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
            if payload.get("optimizer_state") is not None:
                opt.load_state_dict(payload["optimizer_state"])
            if payload.get("scaler_state") is not None:
                scaler.load_state_dict(payload["scaler_state"])

            collapsed_epoch = int(payload.get("epoch", warm))
            if collapsed_epoch >= epochs:
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                return best, labels, manifest
        else:
            model.load_state_dict(payload["model_state"])
            if payload.get("optimizer_state") is not None:
                opt.load_state_dict(payload["optimizer_state"])
            if payload.get("scaler_state") is not None:
                scaler.load_state_dict(payload["scaler_state"])
            warm_epoch = int(payload.get("epoch", 0))

    # Warm mixture phase.
    if phase == "warm":
        for epoch in range(warm_epoch + 1, warm + 1):
            model.train()
            for images, masks in train_loader:
                images, masks = images.to(cfg.device), masks.to(cfg.device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=amp):
                    loss = criterion(model(images), masks)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            payload = {
                "phase": "warm",
                "epoch": epoch,
                "model_state": {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                },
                "optimizer_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
                "chromosome": list(chromosome),
                "history": history,
                "best": best,
                "best_model_state": best_model_state,
                "labels": None,
                "manifest": None,
            }
            _save_checkpoint_file(checkpoint, payload)

        manifest = collect_scale_selections(model)
        labels = selected_scale_labels(model)

        collapse_to_selected_scales(model)
        del opt, scaler

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        phase = "collapsed"
        collapsed_epoch = warm

        # Persist the phase transition before expensive post-collapse training.
        _save_checkpoint_file(
            checkpoint,
            {
                "phase": "collapsed",
                "epoch": warm,
                "model_state": {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                },
                "optimizer_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
                "chromosome": list(chromosome),
                "history": history,
                "best": best,
                "best_model_state": best_model_state,
                "labels": list(labels),
                "manifest": manifest,
            },
        )

    # If the requested budget ends at the warm phase, still evaluate the
    # collapsed model once so the experiment has a valid result.
    if collapsed_epoch >= epochs and best is None:
        val = evaluate(model, val_loader, cfg.device, criterion)
        best = {**val, "epoch": collapsed_epoch}
        best_model_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
        _save_checkpoint_file(
            checkpoint,
            {
                "phase": "collapsed",
                "epoch": collapsed_epoch,
                "model_state": {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                },
                "optimizer_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
                "chromosome": list(chromosome),
                "history": history,
                "best": best,
                "best_model_state": best_model_state,
                "labels": list(labels),
                "manifest": manifest,
            },
        )

    # Post-collapse phase.
    for epoch in range(collapsed_epoch + 1, epochs + 1):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(cfg.device), masks.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                loss = criterion(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        val = evaluate(model, val_loader, cfg.device, criterion)
        history.append({
            "epoch": epoch,
            "val_loss": val["loss"],
            "val_dice": val["dice"],
            "val_miou": val["miou"],
        })

        if best is None or val["dice"] > best["dice"]:
            best = {**val, "epoch": epoch}
            best_model_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        _save_checkpoint_file(
            checkpoint,
            {
                "phase": "collapsed",
                "epoch": epoch,
                "model_state": {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                },
                "optimizer_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
                "chromosome": list(chromosome),
                "history": history,
                "best": best,
                "best_model_state": best_model_state,
                "labels": list(labels),
                "manifest": manifest,
            },
        )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return best, labels, manifest


def _save_checkpoint_file(path, payload):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)



def scale_ablation(
    cfg,
    calibration,
    train_loader,
    val_loader,
    out,
    resume=True,
):
    out = Path(out)
    result_path = out / "scale_ablation.csv"

    existing = _load_csv_if_valid(
        result_path,
        required_columns=("chromosome", "mode"),
    ) if resume else None

    rows = existing.to_dict("records") if existing is not None else []

    completed = {
        _row_key(json.loads(row["chromosome"]), row["mode"])
        for row in rows
    }

    top = calibration.sort_values("val_dice", ascending=False).head(
        cfg.top_k_scale_ablation
    )

    for _, r in top.iterrows():
        c = json.loads(r.chromosome)
        arch_id = int(r.arch_id)
        seed = arch_seed(cfg.seed + 20000, c)
        ckpt_dir = out / "scale_ablation_checkpoints"

        # 1. Current project strategy: 10-epoch mixture training, argmax,
        #    collapse, then remaining training.
        mode = "current_10epoch_argmax_collapse"
        key = _row_key(c, mode)
        if key not in completed:
            best_current, labels, manifest = train_current_scale_collapse(
                c,
                train_loader,
                val_loader,
                cfg,
                seed,
                cfg.scale_ablation_epochs,
                checkpoint=ckpt_dir / f"arch_{arch_id:03d}_current.pth",
            )
            rows.append({
                "chromosome": json.dumps(c),
                "mode": mode,
                "val_dice": best_current["dice"],
                "val_miou": best_current["miou"],
                "selected_scales": json.dumps(labels),
                "scale_weights": json.dumps(manifest),
            })
            completed.add(key)
            _save_rows(rows, result_path)

        # 2. All three scales remain active for the full budget.
        mode = "all_three_scales"
        key = _row_key(c, mode)
        if key not in completed:
            _, best_all, _ = train_architecture(
                c,
                train_loader,
                val_loader,
                cfg.device,
                cfg.base_channels,
                cfg.scale_ablation_epochs,
                seed,
                cfg.learning_rate,
                cfg.weight_decay,
                ckpt_dir / f"arch_{arch_id:03d}_all_three.pth",
            )
            rows.append({
                "chromosome": json.dumps(c),
                "mode": mode,
                "val_dice": best_all["dice"],
                "val_miou": best_all["miou"],
            })
            completed.add(key)
            _save_rows(rows, result_path)

        # 3. Force one scale consistently across every actual ScaleMixture.
        #    train_architecture expands the scalar dynamically; there is no
        #    architecture-dependent hardcoded list length.
        for scale in (3, 5, 7):
            mode = f"hard_scale_{scale}"
            key = _row_key(c, mode)

            if key in completed:
                continue

            _, best_one, _ = train_architecture(
                c,
                train_loader,
                val_loader,
                cfg.device,
                cfg.base_channels,
                cfg.scale_ablation_epochs,
                seed,
                cfg.learning_rate,
                cfg.weight_decay,
                ckpt_dir / f"arch_{arch_id:03d}_scale_{scale}.pth",
                hard_selected_scales=scale,
            )

            rows.append({
                "chromosome": json.dumps(c),
                "mode": mode,
                "val_dice": best_one["dice"],
                "val_miou": best_one["miou"],
            })
            completed.add(key)
            _save_rows(rows, result_path)

    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(result_path, index=False)
    return result



def main():
    a = parse()
    resume = not a.no_resume

    cfg = DiagnosticConfig(
        dataset=a.dataset,
        data_root=a.data_root or DiagnosticConfig(a.dataset).data_root,
        output_root=a.output_root or DiagnosticConfig(a.dataset).output_root,
        calibration_architectures=a.architectures,
        calibration_epochs=a.epochs,
        proxy_images=a.proxy_images,
        proxy_repeats=a.proxy_repeats,
        top_k_scale_ablation=a.scale_ablation_k,
        scale_ablation_epochs=a.scale_ablation_epochs,
        baseline_epochs=a.baseline_epochs,
        num_workers=a.workers,
    )
    cfg.validate()
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    base = ExperimentConfig(cfg.dataset)
    base.data_root = cfg.data_root
    base.output_root = cfg.run_dir
    base.num_workers = cfg.num_workers
    base.base_channels = cfg.base_channels
    base.image_size = cfg.image_size
    base.batch_size = cfg.batch_size
    base.train_augment = cfg.train_augment

    verify_dataset(base, force_manifest=a.force_split, save_examples=True)
    manifest, split_report = create_or_load_manifest(base, force=a.force_split)

    save_json(
        cfg.run_dir / "diagnostic_config.json",
        {
            **cfg.__dict__,
            "data_root": str(cfg.data_root),
            "output_root": str(cfg.output_root),
            "device": str(cfg.device),
            "split_report": split_report,
            "resume": resume,
        },
    )

    train_loader, val_loader, test_loader = loaders(cfg, manifest)

    # Sanity checks are expensive enough that they are also resumable.
    sanity_path = cfg.run_dir / "sanity_checks.json"
    if resume and sanity_path.exists():
        with open(sanity_path, "r", encoding="utf-8") as f:
            sanity = json.load(f)
        print("RESUME: loaded existing sanity checks.")
    else:
        sanity = {
            "split": split_integrity(manifest),
            "train_batch": batch_sanity(train_loader, cfg.device),
            "forward_backward": forward_backward_sanity(
                train_loader,
                cfg.device,
                cfg.base_channels,
                cfg.seed,
            ),
            "tiny_real_data_overfit": tiny_overfit_sanity(
                train_loader,
                cfg.device,
                cfg.base_channels,
                epochs=min(
                    100,
                    max(20, cfg.calibration_epochs * 2),
                ),
                seed=cfg.seed,
            ),
        }
        save_json(sanity_path, sanity)

    print("Sanity checks:", json.dumps(sanity, indent=2))

    # Reuse the exact real proxy batch when available.
    proxy_batch_path = cfg.run_dir / "real_proxy_batch.pt"
    if (
        resume
        and proxy_batch_path.exists()
    ):
        proxy_payload = torch.load(
            proxy_batch_path,
            map_location=cfg.device,
            weights_only=False,
        )
        proxy_images = proxy_payload["images"].to(cfg.device)
        proxy_masks = proxy_payload["masks"].to(cfg.device)

        if (
            proxy_images.shape[0] != cfg.proxy_images
            or proxy_images.shape[-2:] != (cfg.proxy_size, cfg.proxy_size)
        ):
            proxy_images, proxy_masks = collect_real_proxy_batch(
                create_dataloader(
                    load_manifest(manifest, "train"),
                    cfg.image_size,
                    cfg.batch_size,
                    cfg.num_workers,
                    False,
                ),
                cfg.proxy_images,
                cfg.proxy_size,
                cfg.device,
            )
            torch.save(
                {
                    "images": proxy_images.cpu(),
                    "masks": proxy_masks.cpu(),
                },
                proxy_batch_path,
            )
        else:
            print("RESUME: loaded existing real proxy batch.")
    else:
        proxy_source = create_dataloader(
            load_manifest(manifest, "train"),
            cfg.image_size,
            cfg.batch_size,
            cfg.num_workers,
            False,
        )
        proxy_images, proxy_masks = collect_real_proxy_batch(
            proxy_source,
            cfg.proxy_images,
            cfg.proxy_size,
            cfg.device,
        )
        torch.save(
            {
                "images": proxy_images.cpu(),
                "masks": proxy_masks.cpu(),
            },
            proxy_batch_path,
        )

    # Reuse the exact architecture list when it already exists.
    arch_path = cfg.run_dir / "calibration_architectures.csv"
    if resume and arch_path.exists():
        arch_df = pd.read_csv(arch_path)
        architectures = [
            json.loads(value)
            for value in arch_df.sort_values("arch_id")["chromosome"]
        ]
        print("RESUME: loaded existing architecture list.")
    else:
        architectures = make_architectures(
            cfg.calibration_architectures,
            cfg.seed,
        )
        pd.DataFrame([
            {"arch_id": i, "chromosome": json.dumps(c)}
            for i, c in enumerate(architectures)
        ]).to_csv(arch_path, index=False)

    # ---------------------------
    # Proxy stage
    # ---------------------------
    proxy = proxy_stage(
        cfg,
        architectures,
        proxy_images,
        proxy_masks,
        cfg.run_dir,
        resume=resume,
    )

    if a.skip_training:
        return

    # ---------------------------
    # Calibration stage
    # ---------------------------
    diagnostic_path = cfg.run_dir / "architecture_diagnostic.csv"

    if resume and diagnostic_path.exists():
        merged = pd.read_csv(diagnostic_path)
        calibration = merged[
            [
                "arch_id",
                "chromosome",
                "seed",
                "val_loss",
                "val_dice",
                "val_miou",
                "best_epoch",
                "train_seconds",
            ]
        ].copy()
        print("RESUME: loaded completed calibration/diagnostic results.")
    else:
        calibration = train_stage(
            cfg,
            architectures,
            train_loader,
            val_loader,
            cfg.run_dir,
            resume=resume,
        )
        merged = proxy.merge(
            calibration,
            on=["arch_id", "chromosome"],
            how="inner",
        )
        merged.to_csv(diagnostic_path, index=False)

    # ---------------------------
    # Correlation stage
    # ---------------------------
    for target in ("val_dice", "val_miou"):
        correlation_path = cfg.run_dir / f"correlations_{target}.csv"

        if resume and correlation_path.exists():
            table = pd.read_csv(correlation_path)
            print(f"RESUME: loaded {correlation_path.name}")
        else:
            table = correlation_table(merged, target)
            table.to_csv(correlation_path, index=False)

        print(f"\nCorrelation ranking vs {target}:")
        print(
            table[
                ["proxy", "spearman", "spearman_p", "kendall", "pearson"]
            ].to_string(index=False)
        )

        # Plotting is deterministic and inexpensive; only skip it when the
        # output directory already contains plots.
        plot_dir = cfg.run_dir / "plots" / target
        if not (resume and plot_dir.exists() and any(plot_dir.iterdir())):
            scatter_all(merged, target, plot_dir)

    # ---------------------------
    # Scale ablation
    # ---------------------------
    if not a.skip_scale_ablation:
        scale_ablation(
            cfg,
            calibration,
            train_loader,
            val_loader,
            cfg.run_dir,
            resume=resume,
        )

    # ---------------------------
    # Fixed-baseline diagnostic
    # ---------------------------
    baseline_stage(
        cfg,
        architectures,
        train_loader,
        val_loader,
        test_loader,
        cfg.run_dir,
        resume=resume,
    )

    summary = {
        "purpose": (
            "real-data diagnosis of training ceiling, proxy validity, "
            "and three-scale design"
        ),
        "proxy_correlation_targets": [
            "validation Dice",
            "validation mIoU",
        ],
        "test_usage": (
            "test set is never used for proxy correlation or "
            "architecture selection"
        ),
        "decision_rule": (
            "choose future search proxy only if it has stable, materially "
            "positive rank correlation with validation Dice/mIoU and beats "
            "simple parameter/FLOP controls; choose scale strategy only from "
            "matched all-three vs hard-scale ablation"
        ),
        "paper_sources": {
            "SWAP": (
                "ICLR 2024 SWAP-NAS; paper-faithful sample-wise ReLU "
                "pattern cardinality is implemented separately from current SA-SWAP"
            ),
            "JacCov": (
                "Mellor et al. 2021; segmentation version is explicitly "
                "an adaptation because original benchmarks use classification outputs"
            ),
        },
    }
    save_json(cfg.run_dir / "diagnosis_protocol.json", summary)

    print("\nDIAGNOSIS COMPLETE")
    print(f"Results: {cfg.run_dir}")



if __name__ == "__main__":
    main()
