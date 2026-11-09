"""Flax spatial encoders for CRL; relabeled goal contents remain unchanged."""
from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from baselines.data.spatial import position_encoding
from baselines.gcrl.networks import MLP, _activation, ensemblize


def map_channels(values, layout):
    rows, cols = layout["shape"]
    channels = []
    valid = None
    for item in layout["maps"]:
        grid = values[..., item["start"]:item["end"]].reshape(-1, rows, cols)
        current_valid = grid >= 0
        valid = current_valid if valid is None else valid & current_valid
        channels.append(grid == 1)
        if item["name"] == "dynamic_map":
            channels.extend([(grid == 2) | (grid == 4), (grid == 3) | (grid == 4)])
    channels.append(valid)
    return jnp.stack(channels, axis=-1).astype(values.dtype), valid


class SpatialEncoder(nn.Module):
    layout: Any
    architecture: str
    settings: Any
    activation: str = "relu"
    layer_norm: bool = False

    @nn.compact
    def __call__(self, values):
        if values.shape[-1] != self.layout["dimension"]:
            raise ValueError("Observation dimension does not match spatial layout")
        settings = self.settings
        activation = _activation(self.activation)
        grid, valid = map_channels(values, self.layout)
        if self.architecture == "cnn_map":
            features = grid
            for channels in settings["cnn_channels"]:
                features = activation(nn.Conv(channels, (3, 3), padding="SAME")(features))
                features = features * valid[..., None]
            features = features.reshape(values.shape[0], -1)
        else:
            dimension = settings["transformer_dim"]
            rows, cols = self.layout["shape"]
            tokens = nn.Dense(dimension)(grid.reshape(values.shape[0], rows * cols, -1))
            tokens = tokens + jnp.asarray(position_encoding(rows, cols, dimension))
            cls = self.param("cls_token", nn.initializers.zeros, (1, 1, dimension))
            tokens = jnp.concatenate([jnp.broadcast_to(cls, (values.shape[0], 1, dimension)), tokens], axis=1)
            valid_tokens = jnp.concatenate([jnp.ones((values.shape[0], 1), dtype=bool), valid.reshape(values.shape[0], -1)], axis=1)
            # Mask keys, retaining CLS even for an entirely padded observation.
            mask = valid_tokens[:, None, None, :]
            for _ in range(settings["transformer_layers"]):
                hidden = nn.LayerNorm()(tokens)
                tokens = tokens + nn.MultiHeadDotProductAttention(
                    num_heads=settings["transformer_heads"], qkv_features=dimension,
                    dropout_rate=0.0,
                )(hidden, mask=mask)
                hidden = nn.LayerNorm()(tokens)
                hidden = nn.gelu(nn.Dense(settings["transformer_ffn_dim"])(hidden))
                tokens = tokens + nn.Dense(dimension)(hidden)
            features = tokens[:, 0]
        map_features = activation(nn.Dense(settings["map_dim"])(features))
        vector = jnp.take(values, jnp.asarray(self.layout["vector_indices"]), axis=-1)
        vector = MLP(tuple(settings["vector_hidden_units"]), activation=self.activation,
                     activate_final=True, layer_norm=self.layer_norm)(vector)
        return MLP(tuple(settings["fusion_hidden_units"]), activation=self.activation,
                   activate_final=True, layer_norm=self.layer_norm)(jnp.concatenate([map_features, vector], axis=-1))


class SpatialPairEncoder(nn.Module):
    layout: Any
    architecture: str
    settings: Any
    activation: str = "relu"
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations, goals, goal_encoded=False):
        if goal_encoded:
            raise ValueError("CRL spatial actor expects raw goal observations")
        kwargs = dict(layout=self.layout, architecture=self.architecture, settings=self.settings,
                      activation=self.activation, layer_norm=self.layer_norm)
        state = SpatialEncoder(**kwargs, name="state")(observations)
        goal = SpatialEncoder(**kwargs, name="goal")(goals)
        return jnp.concatenate([state, goal], axis=-1)


class SpatialEmbedding(nn.Module):
    layout: Any
    architecture: str
    settings: Any
    hidden_dims: Any
    latent_dim: int
    activation: str
    layer_norm: bool

    @nn.compact
    def __call__(self, observations, actions=None):
        features = SpatialEncoder(self.layout, self.architecture, self.settings,
                                  self.activation, self.layer_norm)(observations)
        if actions is not None:
            features = jnp.concatenate([features, actions], axis=-1)
        return MLP((*self.hidden_dims, self.latent_dim), activation=self.activation,
                   layer_norm=self.layer_norm)(features)


class SpatialBilinearValue(nn.Module):
    layout: Any
    architecture: str
    settings: Any
    hidden_dims: Any
    latent_dim: int
    activation: str
    layer_norm: bool

    def setup(self):
        # Each twin critic owns independent state/action and goal map encoders.
        embedding = ensemblize(SpatialEmbedding, 2)
        kwargs = dict(layout=self.layout, architecture=self.architecture, settings=self.settings,
                      hidden_dims=self.hidden_dims, latent_dim=self.latent_dim,
                      activation=self.activation, layer_norm=self.layer_norm)
        self.phi = embedding(**kwargs)
        self.psi = embedding(**kwargs)

    def __call__(self, observations, goals, actions=None, *, info=False):
        phi, psi = self.phi(observations, actions), self.psi(goals)
        values = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
        return (values, phi, psi) if info else values
