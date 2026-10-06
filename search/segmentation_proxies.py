"""Training-free, dataset-conditioned segmentation NAS proxies.

The two proxies implemented here are intentionally evaluated at a small fixed
proxy resolution. They never update model weights.

1. Spatially-Aware Sample-Wise Activation Proxy (SA-SWAP)
   Uses decoder activation patterns across the same calibration images and
   preserves their spatial locations instead of flattening the feature maps.

2. Segmentation-Aware Jacobian Covariance (SA-JacCov)
   Uses the ground-truth masks only to select balanced foreground/background/
   boundary output locations, then measures the diversity of their input
   Jacobians. No loss is computed and no model parameter is optimized.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F

from models.architecture_builder import build_model_from_chromosome


@torch.no_grad()
def spatial_samplewise_activation_proxy(
    model: torch.nn.Module,
    images: torch.Tensor,
    decoder_weights: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40),
) -> float:
    """Compute a spatially-aware SWAP-style expressivity score.

    For each decoder stage, every channel/spatial location produces a binary
    activation pattern across the calibration images. The score is the mean
    number of distinct patterns per spatial location, normalized by the number
    of possible patterns. Later, higher-resolution decoder stages receive more
    weight because they are more directly involved in reconstructing masks.
    """
    if images.ndim != 4 or images.shape[0] < 2:
        raise ValueError("SA-SWAP requires at least two calibration images.")

    if len(decoder_weights) != 4:
        raise ValueError("Exactly four decoder-stage weights are required.")

    weight_sum = sum(decoder_weights)
    if weight_sum <= 0:
        raise ValueError("Decoder weights must have a positive sum.")

    weights = tuple(w / weight_sum for w in decoder_weights)
    activations: list[torch.Tensor] = []
    handles = []

    def capture(_module, _inputs, output):
        activations.append(output.detach())

    for decoder in model.decoders:
        handles.append(decoder.register_forward_hook(capture))

    try:
        model.eval()
        _ = model(images)
    finally:
        for handle in handles:
            handle.remove()

    if len(activations) != 4:
        raise RuntimeError("Failed to capture all four decoder activations.")

    batch_size = images.shape[0]
    max_patterns = float(2**batch_size)
    stage_scores: list[float] = []

    for activation in activations:
        # Encode each channel's activation across the calibration images as one
        # integer code. For every spatial location we then count how many distinct
        # channel codes occur. This preserves the spatial grid instead of collapsing
        # all H*W locations into one global set of patterns.
        binary = (activation > 0).to(torch.int64)
        bit_weights = (2 ** torch.arange(batch_size, device=activation.device, dtype=torch.int64)).view(batch_size, 1, 1, 1)
        codes = (binary * bit_weights).sum(dim=0)  # [C,H,W]
        flat_codes = codes.reshape(codes.shape[0], -1)
        sorted_codes, _ = torch.sort(flat_codes, dim=0)
        if sorted_codes.shape[0] == 1:
            unique_counts = torch.ones(sorted_codes.shape[1], device=activation.device)
        else:
            unique_counts = 1 + (sorted_codes[1:] != sorted_codes[:-1]).sum(dim=0)
        # Normalize by the maximum number of distinct sample-wise codes possible
        # at a spatial location.
        stage_capacity = float(min(codes.shape[0], int(max_patterns)))
        stage_scores.append(float((unique_counts.float() / stage_capacity).mean().item()))

    return float(sum(w * s for w, s in zip(weights, stage_scores)))


def _boundary_map(mask: torch.Tensor) -> torch.Tensor:
    """Return a one-pixel-ish boundary band from a binary mask."""
    mask = mask.float()
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).abs() > 0


def _sample_probe_indices(
    mask: torch.Tensor,
    probes_per_region: int,
) -> list[int]:
    """Select deterministic, approximately class-balanced spatial probes."""
    flat_mask = mask.squeeze(0).flatten()
    boundary = _boundary_map(mask.unsqueeze(0)).squeeze(0).flatten()

    foreground = torch.where(flat_mask > 0.5)[0]
    background = torch.where(flat_mask <= 0.5)[0]
    boundary_idx = torch.where(boundary)[0]

    # Prefer boundary points that are also foreground/background separated by
    # the boundary band. If a region is absent, deterministic fallback samples
    # are taken from the available pixels.
    regions = [foreground, background, boundary_idx]
    selected: list[int] = []
    for region in regions:
        if region.numel() == 0:
            continue
        count = min(probes_per_region, int(region.numel()))
        # Evenly spaced selection avoids dependence on a random generator.
        positions = torch.linspace(0, region.numel() - 1, steps=count).round().long()
        selected.extend(int(region[position].item()) for position in positions)

    # Remove duplicate locations while preserving deterministic order.
    return list(dict.fromkeys(selected))


def segmentation_aware_jacobian_covariance(
    model: torch.nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    probes_per_region: int = 2,
    jitter: float = 1e-4,
) -> float:
    """Compute a segmentation-aware Jacobian covariance score.

    The proxy is calculated from local segmentation-logit sensitivities with
    respect to the input pixels. Ground-truth masks are used only to select a
    balanced set of foreground, background, and boundary output locations.
    For each image, the normalized Gram matrix of those local Jacobians is
    formed and its log-determinant is returned. A larger value means the
    architecture provides a richer and less redundant set of local input
    sensitivities before training.

    This is a zero-training proxy, but it necessarily uses backward passes to
    obtain Jacobians. The proxy therefore belongs to the zero-cost-NAS family
    rather than meaning literally zero GPU operations.
    """
    if images.ndim != 4 or masks.ndim != 4:
        raise ValueError("Images and masks must be four-dimensional tensors.")
    if images.shape[0] != masks.shape[0]:
        raise ValueError("Images and masks must have the same batch size.")

    model.eval()
    image_scores: list[float] = []

    for image, mask in zip(images, masks):
        probe_indices = _sample_probe_indices(mask, probes_per_region)
        if len(probe_indices) < 2:
            continue

        # A fresh leaf tensor is required because autograd.grad is taken with
        # respect to the input image, not with respect to model parameters.
        input_image = image.unsqueeze(0).detach().clone().requires_grad_(True)
        logits = model(input_image)[0, 0]
        height, width = logits.shape

        jacobians: list[torch.Tensor] = []
        for flat_index in probe_indices:
            row = flat_index // width
            col = flat_index % width
            scalar = logits[row, col]
            gradient = torch.autograd.grad(
                scalar,
                input_image,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            vector = gradient.reshape(-1).detach()
            norm = torch.linalg.vector_norm(vector).clamp_min(1e-12)
            jacobians.append(vector / norm)

        J = torch.stack(jacobians, dim=0)
        gram = J @ J.T
        gram = 0.5 * (gram + gram.T)
        gram = gram + jitter * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)

        sign, logdet = torch.linalg.slogdet(gram)
        if sign <= 0 or not torch.isfinite(logdet):
            # A numerically singular Gram matrix carries little useful
            # discriminative information, so use a large finite penalty rather
            # than allowing NaNs to corrupt NSGA-II.
            image_scores.append(float(torch.logdet(gram.clamp_min(jitter)).item()))
        else:
            image_scores.append(float(logdet.item()))

        del logits, input_image, J, gram

    if not image_scores:
        raise RuntimeError("Could not obtain valid Jacobian probes from calibration masks.")

    return float(sum(image_scores) / len(image_scores))


def make_proxy_model(
    chromosome: list[int],
    base_channels: int,
    proxy_image_size: tuple[int, int],
    device: torch.device,
) -> torch.nn.Module:
    """Build a fresh candidate model at the reduced proxy resolution."""
    del proxy_image_size  # Architecture itself is resolution agnostic.
    model = build_model_from_chromosome(chromosome, base_channels=base_channels)
    return model.to(device)
