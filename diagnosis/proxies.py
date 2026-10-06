"""Diagnostic zero-cost proxies.

This module intentionally distinguishes paper-faithful implementations from the
project's current segmentation-specific variants. No synthetic input is used.
"""
from __future__ import annotations

import math
import hashlib
from contextlib import contextmanager
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from search.segmentation_proxies import (
    segmentation_aware_jacobian_covariance,
    spatial_samplewise_activation_proxy,
)
from search.synflow_proxy import synflow_score


@contextmanager
def preserve_rng():
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def _relu_outputs(model: nn.Module, images: torch.Tensor) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    hooks = []

    def hook(_m, _i, out):
        if isinstance(out, torch.Tensor) and out.ndim >= 2:
            outputs.append(out.detach())

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook))
    try:
        model.eval()
        with torch.no_grad():
            model(images)
    finally:
        for h in hooks:
            h.remove()
    return outputs


def original_swap_score(model: nn.Module, images: torch.Tensor) -> float:
    """Paper-faithful SWAP construction: unique sample-wise ReLU patterns.

    For every intermediate activation value, its binary activation vector across
    the S real input samples is a sample-wise pattern. The score is the cardinality
    of the set of distinct patterns over all captured ReLU intermediate values.

    This is deliberately not the project's SA-SWAP decoder/spatial heuristic.
    """
    if images.shape[0] < 2:
        raise ValueError("SWAP requires at least two real samples")
    activations = _relu_outputs(model, images)
    if not activations:
        raise RuntimeError("No ReLU activations found for paper-style SWAP")

    patterns: set[bytes] = set()
    for activation in activations:
        binary = (activation > 0).permute(1, 0, *range(2, activation.ndim)).reshape(-1, images.shape[0])
        # SWAP pattern is across samples. Store each bit-vector compactly.
        for row in binary.cpu().numpy():
            patterns.add(hashlib.blake2b(row.tobytes(), digest_size=16).digest())
    return float(len(patterns))


def regularized_swap(score: float, parameters: int, mu: float, sigma: float) -> float:
    sigma = max(float(sigma), 1e-12)
    return float(score * math.exp(-((parameters - mu) ** 2) / (2.0 * sigma ** 2)))


def original_jacobian_covariance(model: nn.Module, images: torch.Tensor, jitter: float = 1e-4) -> float:
    """Mellor-style Jacobian covariance proxy, adapted to one-channel segmentation.

    The original method scores covariance of input Jacobians across a minibatch.
    For a segmentation model, we reduce each image's output to the spatial mean
    logit, then compute its input Jacobian. This preserves the key paper idea—
    covariance across samples—without selecting pixels using ground-truth masks.

    This is explicitly labeled an adaptation because the original benchmarks used
    classification outputs; it is not claimed to be the exact original code path.
    """
    model.eval()
    jacobians = []
    for image in images:
        x = image.unsqueeze(0).detach().clone().requires_grad_(True)
        y = model(x)[0, 0].mean()
        grad = torch.autograd.grad(y, x, retain_graph=False, create_graph=False)[0]
        jacobians.append(grad.reshape(-1).detach())
    J = torch.stack(jacobians)
    J = J - J.mean(dim=0, keepdim=True)
    cov = (J @ J.T) / max(J.shape[1] - 1, 1)
    cov = 0.5 * (cov + cov.T) + jitter * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    sign, logdet = torch.linalg.slogdet(cov)
    return float(logdet.item()) if sign > 0 and torch.isfinite(logdet) else float("-inf")


def naswot_relu_logdet(model: nn.Module, images: torch.Tensor, jitter: float = 1e-4) -> float:
    """Standard NASWOT-style ReLU activation kernel logdet control proxy."""
    activations = _relu_outputs(model, images)
    if not activations:
        raise RuntimeError("No ReLU activations found")
    K = None
    for a in activations:
        b = (a > 0).flatten(1).float()
        k = b @ b.T + (1.0 - b) @ (1.0 - b).T
        K = k if K is None else K + k
    K = K / max(len(activations), 1)
    K = K + jitter * torch.eye(K.shape[0], device=K.device)
    sign, logdet = torch.linalg.slogdet(K)
    return float(logdet.item()) if sign > 0 and torch.isfinite(logdet) else float("-inf")


def gradient_norm(model: nn.Module, images: torch.Tensor, masks: torch.Tensor) -> float:
    model.zero_grad(set_to_none=True)
    logits = model(images)
    loss = F.binary_cross_entropy_with_logits(logits, masks)
    loss.backward()
    return float(torch.sqrt(sum((p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None))).item())


def snip(model: nn.Module, images: torch.Tensor, masks: torch.Tensor) -> float:
    model.zero_grad(set_to_none=True)
    logits = model(images)
    loss = F.binary_cross_entropy_with_logits(logits, masks)
    loss.backward()
    score = sum((p.detach().abs() * p.grad.detach().abs()).sum() for p in model.parameters() if p.grad is not None)
    return float(score.item())


def score_all(
    model_factory: Callable[[], nn.Module],
    images: torch.Tensor,
    masks: torch.Tensor,
    parameters: int,
    swap_mu: float,
    swap_sigma: float,
) -> dict[str, float]:
    """Score one architecture with a common real-data batch and deterministic init."""
    results: dict[str, float] = {"parameters": float(parameters)}

    # Every proxy gets a fresh model with the same factory-controlled initialization.
    m = model_factory().to(images.device)
    results["synflow_log10"] = float(synflow_score(m, tuple(images.shape[-2:]), images.device).log10_score)
    del m

    m = model_factory().to(images.device)
    results["current_sa_swap"] = float(spatial_samplewise_activation_proxy(m, images))
    del m

    m = model_factory().to(images.device)
    results["current_sa_jaccov"] = float(segmentation_aware_jacobian_covariance(m, images, masks))
    del m

    m = model_factory().to(images.device)
    raw_swap = original_swap_score(m, images)
    results["original_swap"] = raw_swap
    results["regularized_swap"] = regularized_swap(raw_swap, parameters, swap_mu, swap_sigma)
    del m

    m = model_factory().to(images.device)
    results["original_jacobian_covariance"] = original_jacobian_covariance(m, images)
    del m

    m = model_factory().to(images.device)
    results["naswot_relu_logdet"] = naswot_relu_logdet(m, images)
    del m

    m = model_factory().to(images.device)
    results["grad_norm"] = gradient_norm(m, images, masks)
    del m

    m = model_factory().to(images.device)
    results["snip"] = snip(m, images, masks)
    del m

    return results
