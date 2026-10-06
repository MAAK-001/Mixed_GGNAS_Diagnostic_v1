from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, pearsonr


def bootstrap_spearman(x, y, repeats=2000, seed=42):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    values = []
    for _ in range(repeats):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        values.append(spearmanr(x[idx], y[idx]).statistic)
    if not values:
        return {"rho": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    return {"rho": float(spearmanr(x, y).statistic), "ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975))}


def correlation_table(df: pd.DataFrame, target: str = "val_dice"):
    rows = []
    ignore = {"chromosome", "architecture", "seed", "train_seconds", target, "val_miou", "val_loss", "parameters"}
    for col in df.columns:
        if col in ignore or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        sub = df[[col, target]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 5 or sub[col].nunique() < 2:
            continue
        sr = spearmanr(sub[col], sub[target])
        kt = kendalltau(sub[col], sub[target])
        pr = pearsonr(sub[col], sub[target])
        boot = bootstrap_spearman(sub[col], sub[target])
        rows.append({"proxy": col, "target": target, "n": len(sub), "spearman": sr.statistic, "spearman_p": sr.pvalue, "spearman_ci_low": boot["ci_low"], "spearman_ci_high": boot["ci_high"], "kendall": kt.statistic, "kendall_p": kt.pvalue, "pearson": pr.statistic, "pearson_p": pr.pvalue})
    return pd.DataFrame(rows).sort_values("spearman", ascending=False)
