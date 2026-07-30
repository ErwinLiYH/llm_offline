"""CrossMaze: maze environments with structured layout + sensing observations.

Public API:
- `make(env_family, variant, ...)`: build a wrapped PointMaze/AntMaze env whose
  observations carry a structured `obs["crossmaze"]` field (static/dynamic map
  layouts, location sensing, wall sensing). Text rendering stays on the
  algorithm side.
- `make_seed_map(env_family, map_seed, ...)`: build the same wrapper directly
  from a versioned procedural map seed without registering a variant.
- `CrossMazeEnv` / `CROSSMAZE_OBS_KEY`: the wrapper class and the obs key.
- `build_dynamic_map` / `format_dynamic_visual_map`: numeric and prompt-text
  dynamic-map helpers (`2=C`, `3=G`, `4=S`).
- `NEIGHBOR_DIRECTIONS` / `NEIGHBOR_STATUS_*`: the fixed numeric neighbor
  observation contract (`[up, down, left, right]`, `0=free, 1=wall, 2=risk`).
- `REWARD_TYPES` / `normalize_reward_type` / `resolve_reward_type`: the shared
  configurable sparse/dense reward contract.
- `compute_sensing_state` / `render_sensing_text`: the sensing compute/render
  split shared with the offline training pipeline.
- `get_map_difficulty_config` / `path_difficulty_config`: versioned geometric
  path/map difficulty metadata shared by ordinary eval and eval hard-sample.

Heavy imports (gymnasium, env registries) are deferred until `make` or
`CrossMazeEnv` is first accessed so that `crossmaze.layout` and
`crossmaze.sensing` stay lightweight for tokenization workers.
"""

from crossmaze.layout import (  # noqa: F401
    DYNAMIC_MAP_CURRENT,
    DYNAMIC_MAP_GOAL,
    DYNAMIC_MAP_SUCCESS,
    build_dynamic_map,
    format_dynamic_visual_map,
)
from crossmaze.sensing import (  # noqa: F401
    CROSSMAZE_OBS_KEY,
    NEIGHBOR_DIRECTIONS,
    NEIGHBOR_STATUS_FREE,
    NEIGHBOR_STATUS_RISK,
    NEIGHBOR_STATUS_WALL,
    build_sensing,
    compute_sensing_arrays,
    compute_sensing_state,
    render_sensing_text,
)
from crossmaze.reward import (  # noqa: F401
    REWARD_TYPES,
    normalize_reward_type,
    resolve_reward_type,
    reward_typed_dataset_path,
)
from crossmaze.seed_map import (  # noqa: F401
    SEED_MAP_VERSION,
    SeedMapSpec,
    build_seed_map_env_paras,
    build_seed_map_prompt_vars,
    generate_seed_map,
    normalize_seed_map_spec,
    seed_map_generator_hash,
    seed_map_hash,
    seed_map_shape,
)

_LAZY_EXPORTS = {
    "make": ("crossmaze._make", "make"),
    "make_seed_map": ("crossmaze._make", "make_seed_map"),
    "CrossMazeEnv": ("crossmaze.wrapper", "CrossMazeEnv"),
    "ENV_FACTS": ("crossmaze.variants", "ENV_FACTS"),
    "get_env_facts": ("crossmaze.variants", "get_env_facts"),
    "list_variants": ("crossmaze.variants", "list_variants"),
    "eval_env_spec": ("crossmaze.variants", "eval_env_spec"),
    "eval_reset_options": ("crossmaze.variants", "eval_reset_options"),
    "get_map_difficulty_config": (
        "crossmaze.eval_position",
        "get_map_difficulty_config",
    ),
    "path_difficulty_config": (
        "crossmaze.eval_position",
        "path_difficulty_config",
    ),
}

__all__ = [
    "CROSSMAZE_OBS_KEY",
    "CrossMazeEnv",
    "DYNAMIC_MAP_CURRENT",
    "DYNAMIC_MAP_GOAL",
    "DYNAMIC_MAP_SUCCESS",
    "ENV_FACTS",
    "NEIGHBOR_DIRECTIONS",
    "NEIGHBOR_STATUS_FREE",
    "NEIGHBOR_STATUS_RISK",
    "NEIGHBOR_STATUS_WALL",
    "REWARD_TYPES",
    "SEED_MAP_VERSION",
    "SeedMapSpec",
    "build_seed_map_env_paras",
    "build_seed_map_prompt_vars",
    "build_dynamic_map",
    "build_sensing",
    "compute_sensing_arrays",
    "compute_sensing_state",
    "eval_env_spec",
    "eval_reset_options",
    "format_dynamic_visual_map",
    "generate_seed_map",
    "get_env_facts",
    "get_map_difficulty_config",
    "list_variants",
    "make",
    "make_seed_map",
    "normalize_reward_type",
    "normalize_seed_map_spec",
    "path_difficulty_config",
    "render_sensing_text",
    "resolve_reward_type",
    "reward_typed_dataset_path",
    "seed_map_generator_hash",
    "seed_map_hash",
    "seed_map_shape",
]


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        import importlib

        module_name, attr = _LAZY_EXPORTS[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
