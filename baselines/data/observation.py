from __future__ import annotations

from collections.abc import Mapping

import gymnasium as gym
import numpy as np


OBSERVATION_DIMS = {
    "pointmaze": 6,
    "antmaze": 31,
}


def vectorize_observation(observation, env_family: str) -> np.ndarray:
    if not isinstance(observation, Mapping):
        raise ValueError("CrossMaze baseline observation must be a mapping")
    if env_family == "pointmaze":
        state = np.asarray(observation["observation"], dtype=np.float32)
        goal = np.asarray(observation["desired_goal"], dtype=np.float32)
        vector = np.concatenate([state, goal], axis=-1)
    elif env_family == "antmaze":
        achieved_goal = np.asarray(observation["achieved_goal"], dtype=np.float32)
        state = np.asarray(observation["observation"], dtype=np.float32)
        desired_goal = np.asarray(observation["desired_goal"], dtype=np.float32)
        vector = np.concatenate([achieved_goal, state, desired_goal], axis=-1)
    else:
        raise ValueError(f"Unsupported env_family: {env_family!r}")
    expected_dim = OBSERVATION_DIMS[env_family]
    if vector.shape[-1] != expected_dim:
        raise ValueError(
            f"Unexpected {env_family} baseline observation dimension: "
            f"expected {expected_dim}, got {vector.shape[-1]}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{env_family} baseline observation contains non-finite values")
    return np.asarray(vector, dtype=np.float32)


class BaselineObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, *, env_family: str):
        super().__init__(env)
        if env_family not in OBSERVATION_DIMS:
            raise ValueError(f"Unsupported env_family: {env_family!r}")
        self.env_family = env_family
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_DIMS[env_family],),
            dtype=np.float32,
        )

    def observation(self, observation):
        return vectorize_observation(observation, self.env_family)
