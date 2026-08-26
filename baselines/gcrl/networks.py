"""Small state-based network subset adapted from OGBench's MIT code."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp


def default_init(scale: float = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


def ensemblize(cls, num_members: int, out_axes: int = 0, **kwargs):
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_members,
        **kwargs,
    )


def _activation(name: str):
    activations = {
        "relu": nn.relu,
        "gelu": nn.gelu,
        "tanh": jnp.tanh,
        "swish": nn.swish,
    }
    try:
        return activations[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported GCRL activation: {name!r}") from exc


class DiagonalNormal(NamedTuple):
    """Minimal diagonal Gaussian API used by the two upstream actor losses."""

    loc: Any
    scale_diag: Any

    def mode(self):
        return self.loc

    def sample(self, *, seed):
        return self.loc + self.scale_diag * jax.random.normal(seed, self.loc.shape)

    def log_prob(self, values):
        standardized = (values - self.loc) / self.scale_diag
        elementwise = (
            -0.5 * standardized**2
            - jnp.log(self.scale_diag)
            - 0.5 * jnp.log(2.0 * jnp.pi)
        )
        return elementwise.sum(axis=-1)


class Identity(nn.Module):
    def __call__(self, values):
        return values


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activation: str = "gelu"
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, values):
        activation = _activation(self.activation)
        for index, size in enumerate(self.hidden_dims):
            values = nn.Dense(size, kernel_init=self.kernel_init)(values)
            if index + 1 < len(self.hidden_dims) or self.activate_final:
                values = activation(values)
                if self.layer_norm:
                    values = nn.LayerNorm()(values)
        return values


class LengthNormalize(nn.Module):
    @nn.compact
    def __call__(self, values):
        norm = jnp.maximum(jnp.linalg.norm(values, axis=-1, keepdims=True), 1e-6)
        return values / norm * jnp.sqrt(values.shape[-1])


class Param(nn.Module):
    """Scalar parameter module adapted from OGBench."""

    init_value: float = 0.0

    @nn.compact
    def __call__(self):
        return self.param(
            "value",
            init_fn=lambda key: jnp.full((), self.init_value),
        )


class LogParam(nn.Module):
    """Positive scalar parameter represented in log space."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param(
            "log_value",
            init_fn=lambda key: jnp.full((), jnp.log(self.init_value)),
        )
        return jnp.exp(log_value)


class GoalConditionedEncoder(nn.Module):
    state_encoder: nn.Module | None = None
    concat_encoder: nn.Module | None = None

    @nn.compact
    def __call__(self, observations, goals=None, goal_encoded: bool = False):
        representations = []
        if self.state_encoder is not None:
            representations.append(self.state_encoder(observations))
        if goals is not None:
            if goal_encoded:
                if self.concat_encoder is not None:
                    representations.append(goals)
            elif self.concat_encoder is not None:
                representations.append(
                    self.concat_encoder(jnp.concatenate([observations, goals], axis=-1))
                )
        if not representations:
            raise ValueError("GoalConditionedEncoder has no active representation path")
        return jnp.concatenate(representations, axis=-1)


class GoalConditionedActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    activation: str = "gelu"
    constant_std: bool = True
    goal_encoder: nn.Module | None = None
    final_fc_init_scale: float = 1e-2

    def setup(self):
        self.actor_net = MLP(
            self.hidden_dims,
            activation=self.activation,
            activate_final=True,
        )
        self.mean_net = nn.Dense(
            self.action_dim,
            kernel_init=default_init(self.final_fc_init_scale),
        )
        if not self.constant_std:
            self.log_stds = self.param(
                "log_stds", nn.initializers.zeros, (self.action_dim,)
            )

    def __call__(
        self,
        observations,
        goals=None,
        *,
        goal_encoded: bool = False,
        temperature: float = 1.0,
    ):
        if self.goal_encoder is not None:
            inputs = self.goal_encoder(
                observations, goals, goal_encoded=goal_encoded
            )
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)
        means = self.mean_net(outputs)
        log_stds = jnp.zeros_like(means) if self.constant_std else self.log_stds
        log_stds = jnp.clip(log_stds, -5.0, 2.0)
        return DiagonalNormal(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )


class GoalConditionedValue(nn.Module):
    hidden_dims: Sequence[int]
    activation: str = "gelu"
    layer_norm: bool = True
    ensemble: bool = True
    goal_encoder: nn.Module | None = None

    def setup(self):
        mlp_module = ensemblize(MLP, 2) if self.ensemble else MLP
        self.value_net = mlp_module(
            (*self.hidden_dims, 1),
            activation=self.activation,
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def __call__(self, observations, goals=None, actions=None):
        if self.goal_encoder is not None:
            inputs = [self.goal_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        return self.value_net(jnp.concatenate(inputs, axis=-1)).squeeze(-1)


class GoalConditionedBilinearValue(nn.Module):
    hidden_dims: Sequence[int]
    latent_dim: int
    activation: str = "gelu"
    layer_norm: bool = True
    ensemble: bool = True

    def setup(self):
        mlp_module = ensemblize(MLP, 2) if self.ensemble else MLP
        kwargs = {
            "activation": self.activation,
            "activate_final": False,
            "layer_norm": self.layer_norm,
        }
        self.phi = mlp_module((*self.hidden_dims, self.latent_dim), **kwargs)
        self.psi = mlp_module((*self.hidden_dims, self.latent_dim), **kwargs)

    def __call__(self, observations, goals, actions=None, *, info: bool = False):
        phi_inputs = (
            observations
            if actions is None
            else jnp.concatenate([observations, actions], axis=-1)
        )
        phi = self.phi(phi_inputs)
        psi = self.psi(goals)
        values = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
        if info:
            return values, phi, psi
        return values


class GoalConditionedMRNValue(nn.Module):
    """Metric residual network quasimetric from OGBench QRL."""

    hidden_dims: Sequence[int]
    latent_dim: int
    activation: str = "gelu"
    layer_norm: bool = True

    def setup(self):
        self.phi = MLP(
            (*self.hidden_dims, self.latent_dim),
            activation=self.activation,
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def __call__(self, observations, goals, *, is_phi: bool = False, info: bool = False):
        if is_phi:
            phi_state = observations
            phi_goal = goals
        else:
            phi_state = self.phi(observations)
            phi_goal = self.phi(goals)
        symmetric_state = phi_state[..., : self.latent_dim // 2]
        symmetric_goal = phi_goal[..., : self.latent_dim // 2]
        asymmetric_state = phi_state[..., self.latent_dim // 2 :]
        asymmetric_goal = phi_goal[..., self.latent_dim // 2 :]
        squared_distance = ((symmetric_state - symmetric_goal) ** 2).sum(axis=-1)
        quasimetric = jax.nn.relu(
            (asymmetric_state - asymmetric_goal).max(axis=-1)
        )
        values = jnp.sqrt(jnp.maximum(squared_distance, 1e-12)) + quasimetric
        if info:
            return values, phi_state, phi_goal
        return values


class GoalConditionedIQEValue(nn.Module):
    """Interval quasimetric embedding from OGBench QRL."""

    hidden_dims: Sequence[int]
    latent_dim: int
    dim_per_component: int
    activation: str = "gelu"
    layer_norm: bool = True

    def setup(self):
        self.phi = MLP(
            (*self.hidden_dims, self.latent_dim),
            activation=self.activation,
            activate_final=False,
            layer_norm=self.layer_norm,
        )
        self.alpha = Param()

    def __call__(self, observations, goals, *, is_phi: bool = False, info: bool = False):
        alpha = jax.nn.sigmoid(self.alpha())
        if is_phi:
            phi_state = observations
            phi_goal = goals
        else:
            phi_state = self.phi(observations)
            phi_goal = self.phi(goals)
        x = jnp.reshape(
            phi_state,
            (*phi_state.shape[:-1], -1, self.dim_per_component),
        )
        y = jnp.reshape(
            phi_goal,
            (*phi_goal.shape[:-1], -1, self.dim_per_component),
        )
        valid = x < y
        xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
        interval_order = xy.argsort(axis=-1)
        sorted_xy = jnp.take_along_axis(xy, interval_order, axis=-1)
        negative_increment_copies = jnp.take_along_axis(
            valid,
            interval_order % self.dim_per_component,
            axis=-1,
        ) * jnp.where(interval_order < self.dim_per_component, -1, 1)
        negative_input_copies = jnp.cumsum(
            negative_increment_copies,
            axis=-1,
        )
        negative_function = -1.0 * (negative_input_copies < 0)
        negative_function_increment = jnp.concatenate(
            [
                negative_function[..., :1],
                negative_function[..., 1:] - negative_function[..., :-1],
            ],
            axis=-1,
        )
        components = (sorted_xy * negative_function_increment).sum(axis=-1)
        values = (
            alpha * components.mean(axis=-1)
            + (1.0 - alpha) * components.max(axis=-1)
        )
        if info:
            return values, phi_state, phi_goal
        return values
