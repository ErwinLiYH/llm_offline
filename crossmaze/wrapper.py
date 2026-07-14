"""Gym wrapper attaching structured CrossMaze layout + sensing to observations."""

import gymnasium as gym
import numpy as np

from crossmaze.families import extract_position_goal
from crossmaze.sensing import (
    CROSSMAZE_OBS_KEY,
    compute_sensing_state,
    resolve_sensing_config,
)


class CrossMazeEnv(gym.Wrapper):
    """Adds `obs["crossmaze"]` with numeric map layout and sensing state.

    A plain `gym.Wrapper` (not ObservationWrapper) on purpose: the extra key
    intentionally sits outside the declared observation space, and inner env
    attributes (`unwrapped.maze`, renderers, TimeLimit) stay reachable. The
    inner observation dict is shallow-copied; its arrays are never modified.
    """

    def __init__(
        self,
        env,
        *,
        env_family: str,
        layout: dict,
        sensing_config: dict | None = None,
        default_reset_options: dict | None = None,
    ):
        super().__init__(env)
        self.env_family = str(env_family)
        maze_map = [list(row) for row in layout["maze_map"]]
        if not maze_map or not maze_map[0]:
            raise ValueError("CrossMazeEnv layout requires a non-empty maze_map")
        self._maze_map = maze_map
        self._maze_size_scaling = float(layout.get("maze_size_scaling", 1.0))
        self._maze_shape = [len(maze_map), len(maze_map[0])]
        self._sensing_config = resolve_sensing_config(sensing_config)
        self._meta = {
            "maze_map": self._maze_map,
            "maze_size_scaling": self._maze_size_scaling,
            **self._sensing_config,
        }
        self.default_reset_options = None
        if default_reset_options:
            self.default_reset_options = {
                key: [int(value[0]), int(value[1])]
                for key, value in default_reset_options.items()
            }

    def reset(self, *, seed=None, options=None, **kwargs):
        if options is None and self.default_reset_options is not None:
            options = {
                key: np.asarray(value, dtype=np.int64)
                for key, value in self.default_reset_options.items()
            }
        obs, info = self.env.reset(seed=seed, options=options, **kwargs)
        return self._enrich(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._enrich(obs), reward, terminated, truncated, info

    def _enrich(self, obs):
        if not isinstance(obs, dict):
            return obs
        position, goal = extract_position_goal(self.env_family, obs)
        state = compute_sensing_state(position, goal, self._meta)
        enriched = dict(obs)
        enriched[CROSSMAZE_OBS_KEY] = {
            "maze_map": self._maze_map,
            "maze_size_scaling": self._maze_size_scaling,
            "maze_shape": list(self._maze_shape),
            **state,
        }
        return enriched

    def assert_meta_consistent(self, prompt_vars: dict) -> None:
        """Fail fast if prompt vars would sense differently than this wrapper."""
        problems = []
        resolved = resolve_sensing_config(prompt_vars)
        for key, value in self._sensing_config.items():
            if resolved[key] != value:
                problems.append(f"{key}: wrapper={value!r}, prompt_vars={resolved[key]!r}")
        prompt_scaling = float(prompt_vars.get("maze_size_scaling", 1.0))
        if prompt_scaling != self._maze_size_scaling:
            problems.append(
                f"maze_size_scaling: wrapper={self._maze_size_scaling!r}, "
                f"prompt_vars={prompt_scaling!r}"
            )
        prompt_map = prompt_vars.get("maze_map")
        if prompt_map is None or [list(row) for row in prompt_map] != self._maze_map:
            problems.append("maze_map: wrapper layout differs from prompt_vars maze_map")
        if problems:
            raise ValueError(
                "CrossMazeEnv layout/sensing config is inconsistent with rollout prompt_vars:\n  "
                + "\n  ".join(problems)
            )
