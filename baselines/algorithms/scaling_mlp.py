"""Paper-style MLP encoders for controlled BC scaling experiments.

The residual body follows Wang et al., "1000 Layer Networks for
Self-Supervised RL": an activated input projection followed by residual
blocks of four Dense -> LayerNorm -> Swish units, with the residual addition
immediately after the fourth activation.  This module implements the same
building blocks in PyTorch while leaving the deterministic d3rlpy action head
outside the encoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from d3rlpy.models.encoders import EncoderFactory, register_encoder_factory
from d3rlpy.models.torch import Encoder
from d3rlpy.types import Shape, TorchObservation


PAPER_BLOCK_SIZE = 4
_BODY_TYPES = {"plain_mlp", "residual_mlp"}


def _paper_linear(in_features: int, out_features: int) -> nn.Linear:
    """Create the paper's LeCun-uniform, zero-bias Dense layer.

    Flax's ``variance_scaling(1/3, "fan_in", "uniform")`` has uniform bounds
    +/- 1/sqrt(fan_in), which is also PyTorch's default weight bound for a
    Linear layer with ``a=sqrt(5)``.  The paper explicitly initializes every
    bias to zero, unlike PyTorch's default random bias initialization.
    """

    layer = nn.Linear(in_features, out_features)
    bound = 1.0 / math.sqrt(in_features)
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.zeros_(layer.bias)
    return layer


class DenseLayerNormSwish(nn.Module):
    """One Dense -> LayerNorm -> Swish unit from the paper."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.dense = _paper_linear(in_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features)
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layer_norm(self.dense(values)))


class PaperResidualBlock(nn.Module):
    """Four Dense-LayerNorm-Swish units followed by an identity addition."""

    def __init__(self, width: int):
        super().__init__()
        self.units = nn.ModuleList(
            [DenseLayerNormSwish(width, width) for _ in range(PAPER_BLOCK_SIZE)]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        identity = values
        for unit in self.units:
            values = unit(values)
        return values + identity


class ScalingMLPEncoder(Encoder):
    """Vector encoder with paper-compatible plain or residual body."""

    def __init__(
        self,
        observation_size: int,
        *,
        body_type: str,
        width: int,
        body_depth: int,
        block_size: int = PAPER_BLOCK_SIZE,
    ):
        super().__init__()
        _validate_scaling_spec(
            body_type=body_type,
            width=width,
            body_depth=body_depth,
            block_size=block_size,
        )
        self.body_type = body_type
        self.width = width
        self.body_depth = body_depth
        self.block_size = block_size
        self.input_projection = DenseLayerNormSwish(observation_size, width)
        if body_type == "plain_mlp":
            self.body = nn.ModuleList(
                [DenseLayerNormSwish(width, width) for _ in range(body_depth)]
            )
        else:
            self.body = nn.ModuleList(
                [PaperResidualBlock(width) for _ in range(body_depth // block_size)]
            )

    @property
    def body_dense_count(self) -> int:
        """Dense transformations after the input projection, i.e. paper depth."""

        return self.body_depth

    @property
    def encoder_dense_count(self) -> int:
        """Input projection plus the body; excludes d3rlpy's action head."""

        return 1 + self.body_dense_count

    def structure(self) -> dict[str, int | str]:
        """Stable architecture metadata for run artifacts and audits."""

        return {
            "body_type": self.body_type,
            "width": self.width,
            "body_depth": self.body_depth,
            "block_size": self.block_size,
            "residual_block_count": (
                0
                if self.body_type == "plain_mlp"
                else self.body_depth // self.block_size
            ),
            "input_projection_dense_count": 1,
            "body_dense_count": self.body_dense_count,
            "encoder_dense_count": self.encoder_dense_count,
            "unit_order": "Dense -> LayerNorm -> Swish",
        }

    def forward(self, values: TorchObservation) -> torch.Tensor:
        assert isinstance(values, torch.Tensor)
        values = self.input_projection(values)
        for layer in self.body:
            values = layer(values)
        return values


def _validate_scaling_spec(
    *,
    body_type: str,
    width: int,
    body_depth: int,
    block_size: int,
) -> None:
    if body_type not in _BODY_TYPES:
        raise ValueError(f"Unsupported scaling MLP body_type: {body_type!r}")
    for value, field_name in (
        (width, "width"),
        (body_depth, "body_depth"),
        (block_size, "block_size"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field_name} must be a positive integer")
    if block_size != PAPER_BLOCK_SIZE:
        raise ValueError(
            f"Paper scaling MLP requires block_size={PAPER_BLOCK_SIZE}, got {block_size}"
        )
    if body_type == "residual_mlp" and body_depth % block_size != 0:
        raise ValueError(
            "residual_mlp body_depth must be divisible by its residual block size"
        )


@dataclass()
class ScalingMLPEncoderFactory(EncoderFactory):
    """d3rlpy-serializable factory for the paper-style BC scaling encoders."""

    body_type: str = "residual_mlp"
    width: int = 256
    body_depth: int = 4
    block_size: int = PAPER_BLOCK_SIZE

    def create(self, observation_shape: Shape) -> ScalingMLPEncoder:
        if len(observation_shape) != 1:
            raise ValueError("ScalingMLPEncoder only supports vector observations")
        _validate_scaling_spec(
            body_type=self.body_type,
            width=self.width,
            body_depth=self.body_depth,
            block_size=self.block_size,
        )
        return ScalingMLPEncoder(
            int(observation_shape[0]),
            body_type=self.body_type,
            width=self.width,
            body_depth=self.body_depth,
            block_size=self.block_size,
        )

    def create_with_action(
        self,
        observation_shape: Shape,
        action_size: int,
        discrete_action: bool = False,
    ):
        del observation_shape, action_size, discrete_action
        raise ValueError(
            "ScalingMLPEncoderFactory is currently supported only by mlp_bc "
            "and cannot create a state-action encoder"
        )

    @staticmethod
    def get_type() -> str:
        return "crossmaze_scaling_mlp"


register_encoder_factory(ScalingMLPEncoderFactory)
