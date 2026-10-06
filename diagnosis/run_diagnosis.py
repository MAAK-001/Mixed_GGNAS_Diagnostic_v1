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


def proxy_stage(cfg, architectures, proxy_images, proxy_masks):
    rows = []
    # Regularized SWAP calibration uses the observed parameter distribution.
    params = []
    for c in architectures:
        params.append(count_parameters(build_model_from_chromosome(c, base_channels=cfg.base_channels)))
    mu, sigma = float(np.mean(params)), float(np.std(params) if np.std(params) > 0 else 1.0)
    for i, c in enumerate(architectures):
        seed = arch_seed(cfg.seed, c)
        factory = model_factory(c, cfg, seed)
        start = time.perf_counter()
        scores_accum = []
        for rep in range(cfg.proxy_repeats):
            # Same real proxy pool, with deterministic bootstrap subsets. No synthetic
            # data are introduced; repeated subsets quantify sensitivity to the real
            # calibration sample composition.
            g = torch.Generator(device=proxy_images.device).manual_seed(cfg.seed + 1000 * rep)
            idx = torch.randperm(proxy_images.shape[0], generator=g, device=proxy_images.device)[: max(8, min(32, proxy_images.shape[0]))]
            xi, yi = proxy_images[idx], proxy_masks[idx]
            scores_accum.append(score_all(factory, xi, yi, params[i], mu, sigma))
        keys = scores_accum[0].keys()
        row = {"arch_id": i, "chromosome": json.dumps(c), "architecture": json.dumps({"encoder":[BLOCK_NAMES[g] for g in c[:4]],"decoder":[BLOCK_NAMES[g] for g in c[4:]]}), "parameters": params[i], "proxy_seconds": time.perf_counter()-start}
        for k in keys:
            if k == "parameters": continue
            vals = [s[k] for s in scores_accum if np.isfinite(s[k])]
            row[k] = float(np.mean(vals)) if vals else float("nan")
            row[k + "_std"] = float(np.std(vals)) if vals else float("nan")
        rows.append(row)
        print(f"Proxy {i+1}/{len(architectures)}: {c}")
    return pd.DataFrame(rows)


def train_stage(cfg, architectures, train_loader, val_loader, out):
    out = Path(out)
    (out / "histories").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, c in enumerate(architectures):
        seed = arch_seed(cfg.seed, c)
        ckpt = out / "calibration_checkpoints" / f"arch_{i:03d}.pth"
        start = time.perf_counter()
        _, best, history = train_architecture(c, train_loader, val_loader, cfg.device, cfg.base_channels, cfg.calibration_epochs, seed, cfg.learning_rate, cfg.weight_decay, ckpt)
        pd.DataFrame(history).to_csv(out / "histories" / f"arch_{i:03d}.csv", index=False)
        rows.append({"arch_id": i, "chromosome": json.dumps(c), "seed": seed, "val_loss": best["loss"], "val_dice": best["dice"], "val_miou": best["miou"], "best_epoch": best["epoch"], "train_seconds": time.perf_counter()-start})
        print(f"Train {i+1}/{len(architectures)}: Dice={best['dice']:.4f}, mIoU={best['miou']:.4f}")
    return pd.DataFrame(rows)


def baseline_stage(cfg, architectures, train_loader, val_loader, test_loader, out):
    # Diagnostic only: fixed canonical architectures are trained with the complete
    # three-scale mixture for the full budget. Test metrics are reported, not used
    # to choose a proxy or architecture.
    chosen = architectures[:5]
    rows = []
    for i, c in enumerate(chosen):
        seed = arch_seed(cfg.seed + 10000, c)
        ckpt = out / "baseline_checkpoints" / f"arch_{i:03d}.pth"
        _, best, history = train_architecture(c, train_loader, val_loader, cfg.device, cfg.base_channels, cfg.baseline_epochs, seed, cfg.learning_rate, cfg.weight_decay, ckpt)
        payload = torch.load(ckpt, map_location=cfg.device, weights_only=False)
        model = build_model_from_chromosome(c, base_channels=cfg.base_channels).to(cfg.device)
        model.load_state_dict(payload["model_state"])
        criterion = __import__('training.losses', fromlist=['BCE_mIoULoss']).BCE_mIoULoss(.5,.5)
        test = evaluate(model, test_loader, cfg.device, criterion)
        rows.append({"arch_id": i, "chromosome": json.dumps(c), "val_best_dice": best["dice"], "val_best_miou": best["miou"], "test_dice": test["dice"], "test_miou": test["miou"]})
        print(f"Baseline {i+1}/5: val Dice={best['dice']:.4f}; TEST Dice={test['dice']:.4f}")
    pd.DataFrame(rows).to_csv(out / "baseline_test_results.csv", index=False)



def train_current_scale_collapse(chromosome, train_loader, val_loader, cfg, seed, epochs):
    """Reproduce the current project strategy, but return its selected-scale manifest."""
    from training.losses import BCE_mIoULoss
    set_seed(seed)
    model = build_model_from_chromosome(chromosome, base_channels=cfg.base_channels).to(cfg.device)
    criterion = BCE_mIoULoss(.5, .5)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.device.type == "cuda")
    warm = min(10, epochs)
    for _ in range(warm):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(cfg.device), masks.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=cfg.device.type == "cuda"):
                loss = criterion(model(images), masks)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    manifest = collect_scale_selections(model)
    labels = selected_scale_labels(model)
    collapse_to_selected_scales(model)
    del opt, scaler
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.device.type == "cuda")
    best = {"dice": -1.0, "miou": -1.0, "loss": float("inf"), "epoch": 0}
    for epoch in range(warm + 1, epochs + 1):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(cfg.device), masks.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=cfg.device.type == "cuda"):
                loss = criterion(model(images), masks)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        val = evaluate(model, val_loader, cfg.device, criterion)
        if val["dice"] > best["dice"]:
            best = {**val, "epoch": epoch}
    return best, labels, manifest

def scale_ablation(cfg, calibration, train_loader, val_loader, out):
    top = calibration.sort_values("val_dice", ascending=False).head(cfg.top_k_scale_ablation)
    rows = []
    for _, r in top.iterrows():
        c = json.loads(r.chromosome)
        seed = arch_seed(cfg.seed + 20000, c)
        # All three scales: normal model.
        _, best_all, _ = train_architecture(c, train_loader, val_loader, cfg.device, cfg.base_channels, cfg.scale_ablation_epochs, seed, cfg.learning_rate, cfg.weight_decay)
        best_current, labels, manifest = train_current_scale_collapse(c, train_loader, val_loader, cfg, seed, cfg.scale_ablation_epochs)
        rows.append({"chromosome": json.dumps(c), "mode": "current_10epoch_argmax_collapse", "val_dice": best_current["dice"], "val_miou": best_current["miou"], "selected_scales": json.dumps(labels), "scale_weights": json.dumps(manifest)})
        # Hard-scale ablation uses one deterministic scale per block. We test all 3,
        # to separate the benefit of keeping mixtures from the benefit of hard selection.
        for scale in (3,5,7):
            _, best_one, _ = train_architecture(c, train_loader, val_loader, cfg.device, cfg.base_channels, cfg.scale_ablation_epochs, seed, cfg.learning_rate, cfg.weight_decay, hard_selected_scales=[scale]*7)
            rows.append({"chromosome": json.dumps(c), "mode": f"hard_scale_{scale}", "val_dice": best_one["dice"], "val_miou": best_one["miou"]})
        rows.append({"chromosome": json.dumps(c), "mode": "all_three_scales", "val_dice": best_all["dice"], "val_miou": best_all["miou"]})
    pd.DataFrame(rows).to_csv(out / "scale_ablation.csv", index=False)


def main():
    a = parse()
    cfg = DiagnosticConfig(dataset=a.dataset, data_root=a.data_root or DiagnosticConfig(a.dataset).data_root, output_root=a.output_root or DiagnosticConfig(a.dataset).output_root, calibration_architectures=a.architectures, calibration_epochs=a.epochs, proxy_images=a.proxy_images, proxy_repeats=a.proxy_repeats, top_k_scale_ablation=a.scale_ablation_k, scale_ablation_epochs=a.scale_ablation_epochs, baseline_epochs=a.baseline_epochs, num_workers=a.workers)
    cfg.validate(); cfg.run_dir.mkdir(parents=True, exist_ok=True)
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
    save_json(cfg.run_dir / "diagnostic_config.json", {**cfg.__dict__, "data_root": str(cfg.data_root), "output_root": str(cfg.output_root), "device": str(cfg.device), "split_report": split_report})
    train_loader, val_loader, test_loader = loaders(cfg, manifest)
    sanity = {
        "split": split_integrity(manifest),
        "train_batch": batch_sanity(train_loader, cfg.device),
        "forward_backward": forward_backward_sanity(train_loader, cfg.device, cfg.base_channels, cfg.seed),
        "tiny_real_data_overfit": tiny_overfit_sanity(train_loader, cfg.device, cfg.base_channels, epochs=min(100, max(20, cfg.calibration_epochs * 2)), seed=cfg.seed),
    }
    save_json(cfg.run_dir / "sanity_checks.json", sanity)
    print("Sanity checks:", json.dumps(sanity, indent=2))
    # Proxy data are real, deterministic, non-augmented training images. The
    # validation split remains the target for architecture-performance correlation,
    # and the test split remains untouched.
    proxy_source = create_dataloader(load_manifest(manifest, "train"), cfg.image_size, cfg.batch_size, cfg.num_workers, False)
    proxy_images, proxy_masks = collect_real_proxy_batch(proxy_source, cfg.proxy_images, cfg.proxy_size, cfg.device)
    torch.save({"images": proxy_images.cpu(), "masks": proxy_masks.cpu()}, cfg.run_dir / "real_proxy_batch.pt")
    architectures = make_architectures(cfg.calibration_architectures, cfg.seed)
    pd.DataFrame([{"arch_id":i,"chromosome":json.dumps(c)} for i,c in enumerate(architectures)]).to_csv(cfg.run_dir / "calibration_architectures.csv", index=False)

    proxy = proxy_stage(cfg, architectures, proxy_images, proxy_masks)
    proxy.to_csv(cfg.run_dir / "proxy_scores.csv", index=False)
    if a.skip_training:
        return
    calibration = train_stage(cfg, architectures, train_loader, val_loader, cfg.run_dir)
    merged = proxy.merge(calibration, on=["arch_id","chromosome"], how="inner")
    merged.to_csv(cfg.run_dir / "architecture_diagnostic.csv", index=False)
    for target in ("val_dice", "val_miou"):
        table = correlation_table(merged, target)
        table.to_csv(cfg.run_dir / f"correlations_{target}.csv", index=False)
        print(f"\nCorrelation ranking vs {target}:")
        print(table[["proxy","spearman","spearman_p","kendall","pearson"]].to_string(index=False))
        scatter_all(merged, target, cfg.run_dir / "plots" / target)
    if not a.skip_scale_ablation:
        scale_ablation(cfg, calibration, train_loader, val_loader, cfg.run_dir)
    baseline_stage(cfg, architectures, train_loader, val_loader, test_loader, cfg.run_dir)
    summary = {
        "purpose": "real-data diagnosis of training ceiling, proxy validity, and three-scale design",
        "proxy_correlation_targets": ["validation Dice", "validation mIoU"],
        "test_usage": "test set is never used for proxy correlation or architecture selection",
        "decision_rule": "choose future search proxy only if it has stable, materially positive rank correlation with validation Dice/mIoU and beats simple parameter/FLOP controls; choose scale strategy only from matched all-three vs hard-scale ablation",
        "paper_sources": {
            "SWAP": "ICLR 2024 SWAP-NAS; paper-faithful sample-wise ReLU pattern cardinality is implemented separately from current SA-SWAP",
            "JacCov": "Mellor et al. 2021; segmentation version is explicitly an adaptation because original benchmarks use classification outputs",
        },
    }
    save_json(cfg.run_dir / "diagnosis_protocol.json", summary)
    print("\nDIAGNOSIS COMPLETE")
    print(f"Results: {cfg.run_dir}")

if __name__ == "__main__":
    main()
