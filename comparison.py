"""Compare the two independent NSGA-II branches after search and final testing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_front(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def _normalized(values: np.ndarray) -> np.ndarray:
    low = float(values.min())
    high = float(values.max())
    if high - low <= 1e-12:
        return np.ones_like(values)
    return (values - low) / (high - low)


def plot_pareto_fronts(
    spatial_dir: Path,
    jacobian_dir: Path,
    output_path: Path,
) -> None:
    """Plot both three-objective Pareto fronts in one normalized 3-D figure."""
    import matplotlib.pyplot as plt

    spatial = _read_front(spatial_dir / "pareto_front.csv")
    jacobian = _read_front(jacobian_dir / "pareto_front.csv")

    def arrays(rows):
        params = np.asarray([float(r["parameters"]) / 1e6 for r in rows])
        synflow = np.asarray([float(r["synflow_log10"]) for r in rows])
        third = np.asarray([float(r["third_objective_raw"]) for r in rows])
        return params, synflow, _normalized(third)

    sx, sy, sz = arrays(spatial)
    jx, jy, jz = arrays(jacobian)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sx, sy, sz, color="tab:blue", s=65, label="Spatial-SWAP Pareto front")
    ax.scatter(jx, jy, jz, color="tab:orange", s=65, label="SA-JacCov Pareto front")

    if len(sx) > 1:
        order = np.argsort(sx)
        ax.plot(sx[order], sy[order], sz[order], color="tab:blue", linewidth=1.2)
    if len(jx) > 1:
        order = np.argsort(jx)
        ax.plot(jx[order], jy[order], jz[order], color="tab:orange", linewidth=1.2)

    ax.set_xlabel("Trainable parameters (millions)")
    ax.set_ylabel("log10(SynFlow)")
    ax.set_zlabel("Normalized third-objective value")
    ax.set_title("NSGA-II Pareto-front comparison")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_final_comparison(
    spatial_dir: Path,
    jacobian_dir: Path,
    output_dir: Path,
) -> Path:
    """Create one compact CSV/JSON comparison of selected and tested models."""
    spatial_arch = _load_json(spatial_dir / "final_architecture.json")
    jac_arch = _load_json(jacobian_dir / "final_architecture.json")
    spatial_train = _load_json(spatial_dir / "final_training.json")
    jac_train = _load_json(jacobian_dir / "final_training.json")
    spatial_test = _load_json(spatial_dir / "final_test_results.json")
    jac_test = _load_json(jacobian_dir / "final_test_results.json")

    rows = [
        {
            "path": "Spatial-SWAP",
            "third_objective": spatial_arch["proxy"],
            "chromosome": json.dumps(spatial_arch["chromosome"]),
            "parameters": spatial_arch["metrics"]["parameters"],
            "synflow_log10": spatial_arch["metrics"]["synflow_log10"],
            "topsis": spatial_arch["topsis_coefficient"],
            "val_loss": spatial_train["validation"]["loss"],
            "val_dice": spatial_train["validation"]["dice"],
            "val_miou": spatial_train["validation"]["miou"],
            "test_loss": spatial_test["metrics"]["loss"],
            "test_dice": spatial_test["metrics"]["dice"],
            "test_miou": spatial_test["metrics"]["miou"],
            "a1_BCE": spatial_train["validation"]["loss_coefficients"]["a1_BCE"],
            "a2_mIoU": spatial_train["validation"]["loss_coefficients"]["a2_mIoU"],
        },
        {
            "path": "SA-JacCov",
            "third_objective": jac_arch["proxy"],
            "chromosome": json.dumps(jac_arch["chromosome"]),
            "parameters": jac_arch["metrics"]["parameters"],
            "synflow_log10": jac_arch["metrics"]["synflow_log10"],
            "topsis": jac_arch["topsis_coefficient"],
            "val_loss": jac_train["validation"]["loss"],
            "val_dice": jac_train["validation"]["dice"],
            "val_miou": jac_train["validation"]["miou"],
            "test_loss": jac_test["metrics"]["loss"],
            "test_dice": jac_test["metrics"]["dice"],
            "test_miou": jac_test["metrics"]["miou"],
            "a1_BCE": jac_train["validation"]["loss_coefficients"]["a1_BCE"],
            "a2_mIoU": jac_train["validation"]["loss_coefficients"]["a2_mIoU"],
        },
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "final_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "final_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    plot_pareto_fronts(
        spatial_dir,
        jacobian_dir,
        output_dir / "pareto_front_comparison_3d.png",
    )
    return csv_path
