"""Central, reproducible configuration for the Mixed-GGNAS NSGA-II experiments."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch

DatasetName = Literal["BUSI", "CVC", "IDRID"]

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "Datasets"


@dataclass
class ExperimentConfig:
    """Settings for one dataset-specific architecture search or final training run."""

    dataset: DatasetName
    data_root: Path = field(default_factory=lambda: Path(os.environ.get("GGNAS_DATA_ROOT", DEFAULT_DATA_ROOT)))
    output_root: Path = field(default_factory=lambda: Path(os.environ.get("GGNAS_OUTPUT_ROOT", PROJECT_ROOT / "experiments")))
    image_size: tuple[int, int] = (256, 256)  # height, width
    batch_size: int = 8
    num_workers: int = 0  # Safe default for Windows; increase on Colab if useful.
    learning_rate: float = 1e-3
    weight_decay: float = 5e-5
    full_epochs: int = 100
    # Number of initial full-training epochs used to learn the best scale in each
    # manual block. The remaining epochs train only those selected scales.
    scale_selection_epochs: int = 10
    early_stopping_patience: int | None = 20
    population_size: int = 10
    generations: int = 10
    crossover_probability: float = 0.5
    mutation_probability: float = 0.10
    random_seed: int = 42
    # TOPSIS is shared by both branches: third proxy 50%, parameters 30%, SynFlow 20%.
    topsis_third_objective_weight: float = 0.50
    topsis_parameter_weight: float = 0.30
    topsis_synflow_weight: float = 0.20
    use_amp: bool = True
    cache_enabled: bool = True
    resume_enabled: bool = False
    base_channels: int = 16
    include_busi_normal: bool = False
    train_augment: bool = True
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    calculate_flops: bool = False
    # Fixed calibration probe used by both training-free NSGA-II paths.
    proxy_image_size: tuple[int, int] = (64, 64)
    proxy_batch_size: int = 4
    run_name: str | None = None
    # Coefficient-search settings for the final BCE + mIoU training loss.
    loss_weight_min: float = 0.10
    loss_weight_max: float = 0.90
    loss_weight_search_epochs: int = 3
    loss_weight_optimizer_tolerance: float = 0.10
    loss_weight_optimizer_max_iterations: int = 8

    def __post_init__(self) -> None:
        self.dataset = self.dataset.upper()  # type: ignore[assignment]
        if self.dataset not in {"BUSI", "CVC", "IDRID"}:
            raise ValueError("dataset must be BUSI, CVC, or IDRID")
        if min(self.topsis_third_objective_weight, self.topsis_parameter_weight, self.topsis_synflow_weight) <= 0:
            raise ValueError("TOPSIS weights must be positive")
        if abs((self.topsis_third_objective_weight + self.topsis_parameter_weight + self.topsis_synflow_weight) - 1.0) > 1e-8:
            raise ValueError("TOPSIS weights must sum to one")
        if not (0.0 <= self.loss_weight_min < self.loss_weight_max <= 1.0):
            raise ValueError("Loss coefficient bounds must satisfy 0 <= min < max <= 1")
        if self.loss_weight_search_epochs < 1 or self.scale_selection_epochs < 1:
            raise ValueError("Loss/scale selection settings are invalid")
        if self.full_epochs < 1:
            raise ValueError("full_epochs must be at least one")
        if not (0.0 < self.loss_weight_optimizer_tolerance < 1.0) or self.loss_weight_optimizer_max_iterations < 2:
            raise ValueError("Loss coefficient optimizer settings are invalid")
        self.data_root = Path(self.data_root)
        self.output_root = Path(self.output_root)
        if self.dataset == "IDRID" and self.image_size == (256, 256):
            self.image_size = (512, 320)
            self.batch_size = min(self.batch_size, 4)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def experiment_dir(self) -> Path:
        base = self.output_root / self.dataset
        return base if self.run_name is None else base / self.run_name

    @property
    def manifests_dir(self) -> Path:
        return self.experiment_dir / "manifests"

    @property
    def checkpoint_dir(self) -> Path:
        return self.experiment_dir / "checkpoints"

    def as_dict(self) -> dict:
        result = asdict(self)
        result["data_root"] = str(self.data_root)
        result["output_root"] = str(self.output_root)
        result["device"] = str(self.device)
        return result


def make_config(dataset: str, **overrides: object) -> ExperimentConfig:
    """Build one configuration, applying only explicitly supplied command-line overrides."""
    config = ExperimentConfig(dataset=dataset.upper())  # type: ignore[arg-type]
    for name, value in overrides.items():
        if value is not None and hasattr(config, name):
            setattr(config, name, value)
    config.__post_init__()
    return config
