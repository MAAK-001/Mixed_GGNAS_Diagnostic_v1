"""Training-free SynFlow proxy for neural architecture evaluation.

SynFlow is a zero-cost NAS proxy: it scores a freshly initialized network
without optimizer steps, epochs, labels, or validation data. The implementation
follows the standard linearized-network idea: take absolute parameter values,
feed an all-ones input, backpropagate the sum of outputs, and aggregate |w * dw|.
Original parameter signs are restored before returning.

This is a proxy for architecture quality, not segmentation accuracy itself.
The final selected architecture is still fully trained in the existing final
training stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SynFlowResult:
    """Recorded SynFlow measurements for one architecture."""

    score: float
    log10_score: float


def _linearize(model: nn.Module) -> list[tuple[nn.Parameter, torch.Tensor]]:
    """Replace trainable parameters by their absolute values and save originals."""
    saved: list[tuple[nn.Parameter, torch.Tensor]] = []

    with torch.no_grad():
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            saved.append((parameter, parameter.detach().clone()))
            parameter.abs_()

    return saved


def _restore(
    saved: list[tuple[nn.Parameter, torch.Tensor]],
) -> None:
    """Restore the exact pre-SynFlow parameter values."""
    with torch.no_grad():
        for parameter, original in saved:
            parameter.copy_(original)
            parameter.grad = None


def synflow_score(
    model: nn.Module,
    input_size: tuple[int, int],
    device: torch.device,
) -> SynFlowResult:
    """Compute the data-free SynFlow score with one forward/backward pass.

    The model is evaluated in ``eval`` mode so BatchNorm uses its initialization
    statistics rather than updating running statistics. No optimizer or weight
    update is performed.
    """
    if len(input_size) != 2 or min(input_size) <= 0:
        raise ValueError(f"Invalid input_size: {input_size}")

    model = model.to(device)

    # SynFlow can grow exponentially with network depth. In this search space
    # the score can exceed float32 range (3.4e38), even though the computation
    # is mathematically finite. Evaluate the proxy in float64 to prevent that
    # numerical overflow from turning into an invalid candidate.
    original_dtype = next(
        (parameter.dtype for parameter in model.parameters() if parameter.requires_grad),
        torch.float32,
    )
    model = model.to(dtype=torch.float64)
    model.eval()

    saved = _linearize(model)

    try:
        model.zero_grad(set_to_none=True)

        # SynFlow is deliberately evaluated in float64. The product of path
        # contributions can exceed float32 range in deeper candidate networks.
        # AMP is not used because underflow/overflow in the proxy would make
        # architecture ranking unstable.
        x = torch.ones(
            (1, 3, input_size[0], input_size[1]),
            device=device,
            dtype=torch.float64,
        )

        output = model(x)
        if not torch.isfinite(output).all():
            raise RuntimeError("SynFlow forward pass produced non-finite values.")

        # SynFlow uses a data-free scalar objective.
        objective = output.sum()
        objective.backward()

        total = 0.0
        for parameter, _ in saved:
            if parameter.grad is None:
                continue
            contribution = (parameter.detach() * parameter.grad.detach()).abs().sum()
            value = float(contribution.item())
            if not math.isfinite(value):
                raise RuntimeError("SynFlow produced a non-finite parameter contribution.")
            total += value

        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError(f"SynFlow score must be finite and positive, got {total}.")

        return SynFlowResult(
            score=total,
            log10_score=math.log10(total),
        )
    finally:
        _restore(saved)
        model.zero_grad(set_to_none=True)
        # Restore the candidate model to its original dtype before it is
        # discarded by the evaluator.
        model.to(dtype=original_dtype)
