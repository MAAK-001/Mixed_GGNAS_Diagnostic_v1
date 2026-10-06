from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class DiagnosticConfig:
    dataset: str
    data_root: Path = Path(os.environ.get("GGNAS_DATA_ROOT", "../Datasets"))
    output_root: Path = Path(os.environ.get("GGNAS_DIAGNOSTIC_OUTPUT", "diagnostic_runs"))
    image_size: tuple[int, int] = (256, 256)
    batch_size: int = 8
    num_workers: int = 0
    base_channels: int = 16
    seed: int = 42
    calibration_architectures: int = 24
    calibration_epochs: int = 30
    proxy_images: int = 32
    proxy_size: tuple[int, int] = (64, 64)
    proxy_repeats: int = 3
    top_k_scale_ablation: int = 5
    scale_ablation_epochs: int = 50
    baseline_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 5e-5
    use_amp: bool = True
    include_busi_normal: bool = False
    train_augment: bool = True

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.dataset.upper()

    def validate(self) -> None:
        self.dataset = self.dataset.upper()
        if self.dataset not in {"BUSI", "CVC", "IDRID"}:
            raise ValueError("dataset must be BUSI, CVC, or IDRID")
        if self.calibration_architectures < 5:
            raise ValueError("Use at least 5 calibration architectures")
        if self.proxy_images < 8:
            raise ValueError("Use at least 8 real proxy images")
        if self.proxy_repeats < 1:
            raise ValueError("proxy_repeats must be >= 1")
