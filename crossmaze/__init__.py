"""CrossMaze: maze environments with structured layout + sensing observations.

Public API:
- `make(env_family, variant, ...)`: build a wrapped PointMaze/AntMaze env whose
  observations carry a structured `obs["crossmaze"]` field (map layout, location
  sensing, wall sensing). Text rendering stays on the algorithm side via
  `render_sensing_text`.
- `CrossMazeEnv` / `CROSSMAZE_OBS_KEY`: the wrapper class and the obs key.
- `compute_sensing_state` / `render_sensing_text`: the sensing compute/render
  split shared with the offline training pipeline.

Heavy imports (gymnasium, env registries) are deferred until `make` or
`CrossMazeEnv` is first accessed so that `crossmaze.layout` and
`crossmaze.sensing` stay lightweight for tokenization workers.
"""

from crossmaze.sensing import (  # noqa: F401
    CROSSMAZE_OBS_KEY,
    build_sensing,
    compute_sensing_state,
    render_sensing_text,
)

_LAZY_EXPORTS = {
    "make": ("crossmaze._make", "make"),
    "CrossMazeEnv": ("crossmaze.wrapper", "CrossMazeEnv"),
    "ENV_FACTS": ("crossmaze.variants", "ENV_FACTS"),
    "get_env_facts": ("crossmaze.variants", "get_env_facts"),
    "list_variants": ("crossmaze.variants", "list_variants"),
    "eval_env_spec": ("crossmaze.variants", "eval_env_spec"),
    "eval_reset_options": ("crossmaze.variants", "eval_reset_options"),
}

__all__ = [
    "CROSSMAZE_OBS_KEY",
    "CrossMazeEnv",
    "ENV_FACTS",
    "build_sensing",
    "compute_sensing_state",
    "eval_env_spec",
    "eval_reset_options",
    "get_env_facts",
    "list_variants",
    "make",
    "render_sensing_text",
]


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        import importlib

        module_name, attr = _LAZY_EXPORTS[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
