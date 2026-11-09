"""Backend-independent layout and positional encoding for spatial baselines."""
from __future__ import annotations

import numpy as np

from baselines.data.observation import observation_schema, goal_conditioned_observation_schema

SPATIAL_ARCHITECTURES = {"cnn_map", "transformer_map"}


def spatial_layout(env_family: str, observation: dict, *, goal_conditioned=False) -> dict:
    schema = (goal_conditioned_observation_schema if goal_conditioned else observation_schema)(
        env_family, observation
    )
    components = schema["state_components" if goal_conditioned else "components"]
    maps, vector_indices, offset = [], [], 0
    for component in components:
        end = offset + component["dimension"]
        if component["name"] in {"map", "dynamic_map"}:
            maps.append({"name": component["name"], "start": offset, "end": end})
            shape = list(component["shape"])
        else:
            vector_indices.extend(range(offset, end))
        offset = end
    if not maps:
        raise ValueError("Spatial encoders require include_map or include_dynamic_map")
    return {"dimension": offset, "shape": shape, "maps": maps,
            "vector_indices": vector_indices}


def map_indices(layout: dict) -> list[int]:
    return [i for item in layout["maps"] for i in range(item["start"], item["end"])]


def channel_count(layout: dict) -> int:
    return sum(1 if item["name"] == "map" else 3 for item in layout["maps"]) + 1


def position_encoding(rows: int, cols: int, dimension: int) -> np.ndarray:
    """Fixed 2D sin/cos positions, row-major, shared by Torch and Flax."""
    if dimension % 4:
        raise ValueError("Transformer dimension must be divisible by four")
    row, col = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    frequencies = 10000.0 ** (-np.arange(dimension // 4) / (dimension // 4))
    parts = []
    for coordinates in (row, col):
        angles = coordinates.reshape(-1, 1) * frequencies
        parts.extend([np.sin(angles), np.cos(angles)])
    return np.concatenate(parts, axis=-1).astype(np.float32)
