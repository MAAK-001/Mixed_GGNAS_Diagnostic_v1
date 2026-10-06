"""Differentiable three-scale convolution mixtures used by manual blocks.

During zero-cost NAS all three scale branches remain active.  After the selected
architecture has been chosen by TOPSIS, a short scale-selection training stage
learns the three mixture weights in every manual block.  Each block is then
collapsed to its highest-weight scale before the remaining full training and the
final test, so the final model contains only the selected scale per block.
"""

from __future__ import annotations

import torch
from torch import nn


class ScaleMixture(nn.Module):
    """Blend exactly three scale variants, with an optional final collapse."""

    def __init__(
        self,
        *branches: nn.Module,
        scale_labels: tuple[int, int, int] = (3, 5, 7),
    ) -> None:
        super().__init__()

        if len(branches) != 3:
            raise ValueError("ScaleMixture requires exactly three branches")
        if len(scale_labels) != 3:
            raise ValueError("scale_labels must contain exactly three labels")

        self.branches = nn.ModuleList(branches)
        self.logits = nn.Parameter(torch.zeros(3))
        self.scale_labels = tuple(int(label) for label in scale_labels)
        self._selected_index: int | None = None

    @property
    def is_collapsed(self) -> bool:
        """Whether this block has been reduced to one selected scale."""
        return self._selected_index is not None

    def weights(self) -> torch.Tensor:
        """Return the three scale weights, or a one-hot vector after collapse."""
        if self.is_collapsed:
            weights = torch.zeros(
                3,
                device=next(self.parameters()).device,
                dtype=torch.float32,
            )
            weights[self._selected_index] = 1.0  # type: ignore[index]
            return weights
        return torch.softmax(self.logits, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # During NAS and scale-selection training all three branches are used.
        if not self.is_collapsed:
            weights = self.weights()
            outputs = [branch(x) for branch in self.branches]
            result = outputs[0] * weights[0]
            for weight, output in zip(weights[1:], outputs[1:]):
                result = result + weight * output
            return result

        # After scale selection, only the winning branch remains in the module.
        return self.branches[0](x)

    def selected_index(self) -> int:
        """Return the selected branch index."""
        if self.is_collapsed:
            assert self._selected_index is not None
            return self._selected_index
        return int(torch.argmax(self.logits.detach()).item())

    def selected_scale(self) -> int:
        """Return the scale label with the highest learned weight."""
        return self.scale_labels[self.selected_index()]

    def scale_weights(self) -> list[float]:
        """Return normalized scale weights for experiment logs."""
        return [float(value) for value in self.weights().detach().cpu()]

    def collapse_to_selected(self) -> int:
        """Keep only the highest-weight branch and discard the other two.

        The method is intentionally called only after scale-selection training.
        A fresh optimizer must be created after this operation because the
        module's parameter set changes when two branches and the scale logits
        are removed.
        """
        if self.is_collapsed:
            return self.selected_scale()

        selected_index = self.selected_index()
        selected_branch = self.branches[selected_index]

        # Replace the three-branch container with the winning branch only.
        self.branches = nn.ModuleList([selected_branch])
        del self.logits
        self._selected_index = selected_index
        return self.selected_scale()


def collapse_to_selected_scales(model: nn.Module) -> dict[str, dict[str, object]]:
    """Collapse every manual block to its learned best scale.

    Returns a manifest keyed by module name so the exact selected scales can be
    reconstructed later for untouched test evaluation.
    """
    selections: dict[str, dict[str, object]] = {}
    for name, module in model.named_modules():
        if isinstance(module, ScaleMixture):
            selected_scale = module.collapse_to_selected()
            selections[name] = {
                "selected_scale": selected_scale,
                "weights": module.scale_weights(),
            }
    return selections


def collect_scale_selections(model: nn.Module) -> dict[str, dict[str, object]]:
    """Collect scale weights/selections without changing the model."""
    selections: dict[str, dict[str, object]] = {}
    for name, module in model.named_modules():
        if isinstance(module, ScaleMixture):
            selections[name] = {
                "selected_scale": module.selected_scale(),
                "weights": module.scale_weights(),
                "collapsed": module.is_collapsed,
            }
    return selections


def selected_scale_labels(model: nn.Module) -> list[int]:
    """Return selected scale labels in module traversal order."""
    return [
        module.selected_scale()
        for module in model.modules()
        if isinstance(module, ScaleMixture)
    ]


def collapse_model_to_scales(model: nn.Module, selected_scales: list[int]) -> dict[str, dict[str, object]]:
    """Collapse a freshly built model using a previously saved scale manifest.

    ``selected_scales`` follows the traversal order of ScaleMixture modules.
    This is used only when reconstructing a final checkpoint for testing.
    """
    mixtures = [module for module in model.modules() if isinstance(module, ScaleMixture)]
    if len(mixtures) != len(selected_scales):
        raise ValueError(
            f"Scale-selection length mismatch: model has {len(mixtures)} "
            f"scale mixtures but {len(selected_scales)} selections were provided."
        )

    selections: dict[str, dict[str, object]] = {}
    for selected_scale, module in zip(selected_scales, mixtures):
        if selected_scale not in module.scale_labels:
            raise ValueError(
                f"Invalid selected scale {selected_scale}; "
                f"expected one of {module.scale_labels}."
            )
        selected_index = module.scale_labels.index(selected_scale)
        # Force the requested branch by setting the logits before collapsing.
        with torch.no_grad():
            module.logits.fill_(-20.0)
            module.logits[selected_index] = 20.0
        selected = module.collapse_to_selected()
        selections[str(len(selections))] = {
            "selected_scale": selected,
            "weights": module.scale_weights(),
            "collapsed": True,
        }
    return selections
