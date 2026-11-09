"""Map encoders and serializable d3rlpy factories/scalers."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from d3rlpy.models.encoders import EncoderFactory, register_encoder_factory
from d3rlpy.models.torch import Encoder, EncoderWithAction
from d3rlpy.preprocessing import StandardObservationScaler
from d3rlpy.preprocessing.observation_scalers import register_observation_scaler

from baselines.data.spatial import channel_count, map_indices, position_encoding


def _activation(name):
    return {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "swish": nn.SiLU}[name]()


def _mlp(input_dim, hidden, activation, layer_norm):
    layers = []
    for size in hidden:
        layers.extend([nn.Linear(input_dim, size), _activation(activation)])
        if layer_norm:
            layers.append(nn.LayerNorm(size))
        input_dim = size
    return nn.Sequential(*layers)


def map_channels(values, layout):
    """Return BHWC semantic channels and a BHW valid-cell mask."""
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
    return torch.stack(channels, dim=-1).to(values.dtype), valid


class SpatialEncoder(Encoder):
    def __init__(self, layout, architecture, settings, activation="relu", layer_norm=False):
        super().__init__()
        self.layout = layout
        self.architecture = architecture
        self.register_buffer("vector_indices", torch.tensor(layout["vector_indices"], dtype=torch.long))
        rows, cols = layout["shape"]
        channels = channel_count(layout)
        if architecture == "cnn_map":
            self.convs = nn.ModuleList()
            for size in settings["cnn_channels"]:
                self.convs.append(nn.Conv2d(channels, size, 3, padding=1))
                channels = size
            self.conv_activation = _activation(activation)
            self.map_projection = nn.Linear(rows * cols * channels, settings["map_dim"])
        else:
            dimension = settings["transformer_dim"]
            self.token_projection = nn.Linear(channels, dimension)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dimension))
            self.register_buffer("positions", torch.from_numpy(position_encoding(rows, cols, dimension)))
            layer = nn.TransformerEncoderLayer(
                d_model=dimension, nhead=settings["transformer_heads"],
                dim_feedforward=settings["transformer_ffn_dim"], dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, settings["transformer_layers"], enable_nested_tensor=False
            )
            # TransformerEncoder clones layers; initialize matrices independently.
            for block in self.transformer.layers:
                for parameter in block.parameters():
                    if parameter.ndim > 1:
                        nn.init.xavier_uniform_(parameter)
            self.map_projection = nn.Linear(dimension, settings["map_dim"])
        self.map_activation = _activation(activation)
        self.vector_net = _mlp(len(layout["vector_indices"]), settings["vector_hidden_units"], activation, layer_norm)
        self.fusion_net = _mlp(settings["map_dim"] + settings["vector_hidden_units"][-1], settings["fusion_hidden_units"], activation, layer_norm)

    def forward(self, values):
        if values.shape[-1] != self.layout["dimension"]:
            raise ValueError("Observation dimension does not match spatial layout")
        grid, valid = map_channels(values, self.layout)
        if self.architecture == "cnn_map":
            features = grid.permute(0, 3, 1, 2)
            for conv in self.convs:
                features = self.conv_activation(conv(features)) * valid[:, None].to(values.dtype)
            features = features.permute(0, 2, 3, 1).flatten(1)
        else:
            tokens = self.token_projection(grid.flatten(1, 2)) + self.positions
            cls = self.cls_token.expand(values.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            padding = torch.cat([torch.zeros_like(valid.flatten(1)[:, :1]), ~valid.flatten(1)], dim=1)
            features = self.transformer(tokens, src_key_padding_mask=padding)[:, 0]
        map_features = self.map_activation(self.map_projection(features))
        vector_features = self.vector_net(values.index_select(-1, self.vector_indices))
        return self.fusion_net(torch.cat([map_features, vector_features], dim=-1))


class SpatialEncoderWithAction(EncoderWithAction):
    def __init__(self, encoder, action_size, settings, activation, layer_norm):
        super().__init__()
        self.encoder = encoder
        self.action_fusion = _mlp(
            settings["fusion_hidden_units"][-1] + action_size,
            [settings["fusion_hidden_units"][-1]], activation, layer_norm,
        )

    def forward(self, values, action):
        return self.action_fusion(torch.cat([self.encoder(values), action], dim=-1))


@dataclass
class SpatialEncoderFactory(EncoderFactory):
    layout: dict = field(default_factory=dict)
    architecture: str = "cnn_map"
    settings: dict = field(default_factory=dict)
    activation: str = "relu"
    layer_norm: bool = False

    def create(self, observation_shape):
        if tuple(observation_shape) != (self.layout["dimension"],):
            raise ValueError("Observation shape does not match spatial layout")
        return SpatialEncoder(self.layout, self.architecture, self.settings, self.activation, self.layer_norm)

    def create_with_action(self, observation_shape, action_size, discrete_action=False):
        if discrete_action:
            raise ValueError("Spatial baselines support continuous actions only")
        return SpatialEncoderWithAction(
            self.create(observation_shape), action_size, self.settings,
            self.activation, self.layer_norm,
        )

    @staticmethod
    def get_type():
        return "crossmaze_spatial"


@dataclass
class SpatialObservationScaler(StandardObservationScaler):
    """Standardize numeric features but preserve map codes/padding exactly."""
    layout: dict = field(default_factory=dict)

    def _torch_preserve(self, original, transformed):
        indices = torch.tensor(map_indices(self.layout), device=original.device)
        return transformed.index_copy(-1, indices, original.index_select(-1, indices))

    def _numpy_preserve(self, original, transformed):
        indices = map_indices(self.layout)
        transformed[..., indices] = original[..., indices]
        return transformed

    def transform(self, x):
        return self._torch_preserve(x, super().transform(x))

    def reverse_transform(self, x):
        return self._torch_preserve(x, super().reverse_transform(x))

    def transform_numpy(self, x):
        return self._numpy_preserve(x, super().transform_numpy(x))

    def reverse_transform_numpy(self, x):
        return self._numpy_preserve(x, super().reverse_transform_numpy(x))

    @staticmethod
    def get_type():
        return "crossmaze_spatial_standard"


register_encoder_factory(SpatialEncoderFactory)
register_observation_scaler(SpatialObservationScaler)
