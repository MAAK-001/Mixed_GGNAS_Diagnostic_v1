from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def scatter_all(df: pd.DataFrame, target: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ignore = {"chromosome", "architecture", "seed", "train_seconds", "val_dice", "val_miou", "val_loss"}
    for col in df.columns:
        if col in ignore or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        sub = df[[col, target]].dropna()
        if len(sub) < 5:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(sub[col], sub[target], s=20, alpha=.75)
        ax.set_xlabel(col)
        ax.set_ylabel(target)
        ax.set_title(f"{col} vs {target}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{col}_vs_{target}.png", dpi=160)
        plt.close(fig)
