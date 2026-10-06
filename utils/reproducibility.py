"""Seed and JSON helpers used to make search experiments reproducible."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch without forcing non-deterministic CUDA behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(path: Path, content: Any) -> None:
    """Write readable JSON, creating only the requested parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, default=str), encoding="utf-8")


def load_json(path: Path, default: Any) -> Any:
    """Load JSON when it exists; otherwise return the caller's default value."""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
