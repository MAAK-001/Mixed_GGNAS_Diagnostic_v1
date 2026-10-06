"""Fixed DARTS-style cells for the two U-Net positions used by this project.

The paper describes DARTS cells with two input nodes and four intermediate
nodes. In this project NSGA-II does not search the internal DARTS topology;
the cell is fixed and only chromosome gene 4 selects DARTS.

The down/up variants use the position-specific operation families listed in
Table 2 of Mixed-GGNAS. Spatial resizing is handled by the surrounding U-Net
so that all block types keep the same tensor interface.
"""

from __future__ import annotations

import torch
from torch import nn


class _ChannelWeightedConv(nn.Module):
    """Lightweight convolution followed by channel weighting."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        padding = dilation * (kernel_size // 2)
        hidden = max(1, channels // 4)

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=False)

        self.weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.norm(self.conv(x)))
        return y * self.weight(y)


class _DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable convolution used by fixed DARTS edges."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()

        padding = dilation

        self.layers = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _DARTSOperation(nn.Module):
    """One fixed operation from the Mixed-GGNAS DARTS operation families."""

    def __init__(self, name: str, channels: int) -> None:
        super().__init__()

        self.name = name

        if name == "avg_pool":
            self.operation = nn.AvgPool2d(
                3,
                stride=1,
                padding=1,
                count_include_pad=False,
            )

        elif name == "max_pool":
            self.operation = nn.MaxPool2d(
                3,
                stride=1,
                padding=1,
            )

        elif name in {"down_cweight", "up_cweight"}:
            self.operation = _ChannelWeightedConv(channels)

        elif name in {"down_dil_conv", "up_dil_conv"}:
            self.operation = _ChannelWeightedConv(
                channels,
                dilation=2,
            )

        elif name in {"down_dep_conv", "up_dep_conv"}:
            self.operation = _DepthwiseSeparableConv(channels)

        elif name in {"down_conv", "up_conv"}:
            self.operation = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=False),
            )

        else:
            raise ValueError(
                f"Unknown fixed DARTS operation: {name}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.operation(x)


class FixedDARTSBlock(nn.Module):
    """Fixed two-input/four-intermediate-node DARTS-style DAG.

    The topology and operations are fixed. Learnable edge strengths are
    optimized by normal gradient descent, but there is no internal DARTS
    architecture search.

    Chromosome gene 4 only decides whether this block is placed at a
    particular encoder or decoder position.
    """

    # (source node A, source node B, operation A, operation B)
    _DOWN_EDGES = (
        (0, 1, "avg_pool", "down_cweight"),
        (0, 2, "max_pool", "down_dil_conv"),
        (1, 2, "down_dep_conv", "down_conv"),
        (2, 3, "down_cweight", "down_dep_conv"),
    )

    _UP_EDGES = (
        (0, 1, "up_cweight", "up_dep_conv"),
        (0, 2, "up_conv", "up_dil_conv"),
        (1, 2, "up_dep_conv", "up_conv"),
        (2, 3, "up_cweight", "up_dep_conv"),
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        direction: str,
    ) -> None:
        super().__init__()

        if direction not in {"down", "up"}:
            raise ValueError(
                "direction must be 'down' or 'up'"
            )

        self.direction = direction

        edges = (
            self._DOWN_EDGES
            if direction == "down"
            else self._UP_EDGES
        )

        self.preprocess0 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.preprocess1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.edge_indices = [
            (left, right)
            for left, right, _, _ in edges
        ]

        self.operations = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _DARTSOperation(
                            operation_a,
                            out_channels,
                        ),
                        _DARTSOperation(
                            operation_b,
                            out_channels,
                        ),
                    ]
                )
                for _, _, operation_a, operation_b in edges
            ]
        )

        # These weights are optimized by gradient descent.
        # They are NOT architecture genes and are NOT searched by NSGA-II.
        self.edge_logits = nn.Parameter(
            torch.zeros(len(edges))
        )

        self.project = nn.Sequential(
            nn.Conv2d(
                out_channels * 4,
                out_channels,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(
                in_channels,
                out_channels,
                1,
                bias=False,
            )
        )

        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = [
            self.preprocess0(x),
            self.preprocess1(x),
        ]

        edge_weights = torch.softmax(
            self.edge_logits,
            dim=0,
        )

        for edge_index, (
            operations,
            (left_index, right_index),
        ) in enumerate(
            zip(
                self.operations,
                self.edge_indices,
            )
        ):
            node = (
                operations[0](states[left_index])
                + operations[1](states[right_index])
            )

            states.append(
                edge_weights[edge_index] * node
            )

        output = self.project(
            torch.cat(states[2:], dim=1)
        )

        return self.activation(
            output + self.shortcut(x)
        )


class FixedDownsampleDARTSBlock(FixedDARTSBlock):
    """Fixed DARTS cell assigned to encoder positions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            direction="down",
        )


class FixedUpsampleDARTSBlock(FixedDARTSBlock):
    """Fixed DARTS cell assigned to decoder positions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            direction="up",
        )