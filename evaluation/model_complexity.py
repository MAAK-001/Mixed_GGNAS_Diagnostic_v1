"""Parameter counting and an optional FLOPs estimate that does not affect NSGA-II objectives."""

from __future__ import annotations

import torch


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable model parameters exactly."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def estimate_flops(model: torch.nn.Module, image_size: tuple[int, int], device: torch.device) -> int | None:
    """Estimate FLOPs with thop when installed; return None instead of adding a mandatory dependency."""
    try:
        from thop import profile  # type: ignore[import-not-found]
    except ImportError:
        return None
    original_device = next(model.parameters()).device
    model.to(device).eval()
    with torch.no_grad():
        flops, _ = profile(model, inputs=(torch.zeros(1, 3, *image_size, device=device),), verbose=False)
    model.to(original_device)
    return int(flops)
