from __future__ import annotations

import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from baselines.evaluation import evaluate_rollouts


class _PredictZero:
    def predict(self, observations):
        return np.zeros((len(observations), 2), dtype=np.float32)


class _EpisodeRecordEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    observation_space = gym.spaces.Dict(
        {
            "observation": gym.spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float32
            ),
            "desired_goal": gym.spaces.Box(
                -np.inf, np.inf, shape=(2,), dtype=np.float32
            ),
        }
    )

    def __init__(self):
        self._seed = 0
        self._step = 0

    def _observation(self):
        start_col = 1 + self._seed % 2
        return {
            "observation": np.array(
                [start_col, 1.0, 0.0, 0.0], dtype=np.float32
            ),
            "desired_goal": np.array([3.0, 1.0], dtype=np.float32),
            "crossmaze": {
                "position_cell": [1, start_col],
                "goal_cell": [3, 1],
                "position_xy": [float(start_col), 1.0],
                "goal_xy": [3.0, 1.0],
            },
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._seed = int(seed or 0)
        self._step = 0
        return self._observation(), {}

    def step(self, action):
        self._step += 1
        success = self._seed == 10 and self._step >= 2
        truncated = self._step >= 3
        return self._observation(), float(success), False, truncated, {
            "success": success
        }


class _FixedEpisodeRecordEnv(_EpisodeRecordEnv):
    def _observation(self):
        return {
            "achieved_goal": np.array([1.0, 1.0], dtype=np.float32),
            "observation": np.zeros(27, dtype=np.float32),
            "desired_goal": np.array([3.0, 1.0], dtype=np.float32),
            "crossmaze": {
                "position_cell": [1, 1],
                "goal_cell": [3, 1],
                "position_xy": [1.0, 1.0],
                "goal_xy": [3.0, 1.0],
            },
        }


class EvaluationRecordTest(unittest.TestCase):
    @patch("baselines.evaluation.crossmaze.make", return_value=_EpisodeRecordEnv())
    def test_episode_start_goal_and_first_success_step_are_recorded(self, _make):
        result = evaluate_rollouts(
            _PredictZero(),
            env_family="pointmaze",
            variants=["umaze"],
            reward_types={"umaze": "sparse"},
            evaluation_config={
                "seed": 10,
                "num_episodes": 2,
                "env_config": {},
            },
            observation_config={
                "include_map": False,
                "include_location_sensing": False,
                "include_wall_sensing": False,
                "wall_sensing_version": "v3",
                "map_sensing_boundary_risk_threshold": 0.1,
            },
        )

        variant = result["variants"]["umaze"]
        self.assertEqual(variant["successful_episode_count"], 1)
        self.assertEqual(variant["success_rate"], 0.5)
        self.assertEqual(variant["first_success_step_mean"], 2.0)
        self.assertEqual(variant["first_success_step_std"], 0.0)
        self.assertEqual(variant["unique_start_goal_count"], 2)

        first, second = variant["episodes"]
        self.assertEqual(first["seed"], 10)
        self.assertEqual(first["start_goal"]["sampling_mode"], "random-start-goal")
        self.assertEqual(
            first["start_goal"]["selection_policy"], "env_default_random"
        )
        self.assertEqual(first["start_goal"]["start_cell"], [1, 1])
        self.assertEqual(first["start_goal"]["goal_cell"], [3, 1])
        self.assertTrue(first["success"])
        self.assertEqual(first["first_success_step"], 2)
        self.assertFalse(second["success"])
        self.assertIsNone(second["first_success_step"])

        aggregate = result["aggregate"]
        self.assertEqual(aggregate["successful_episode_count"], 1)
        self.assertEqual(aggregate["first_success_step_mean"], 2.0)

    @patch(
        "baselines.evaluation.crossmaze.make", return_value=_FixedEpisodeRecordEnv()
    )
    def test_antmaze_default_records_fixed_start_goal(self, _make):
        result = evaluate_rollouts(
            _PredictZero(),
            env_family="antmaze",
            variants=["umaze"],
            reward_types={"umaze": "sparse"},
            evaluation_config={
                "seed": 10,
                "num_episodes": 2,
                "env_config": {},
            },
            observation_config={
                "include_map": False,
                "include_location_sensing": False,
                "include_wall_sensing": False,
                "wall_sensing_version": "v3",
                "map_sensing_boundary_risk_threshold": 0.1,
            },
        )

        variant = result["variants"]["umaze"]
        self.assertEqual(variant["unique_start_goal_count"], 1)
        for episode in variant["episodes"]:
            self.assertEqual(
                episode["start_goal"]["sampling_mode"], "fix-start-goal"
            )
            self.assertEqual(episode["start_goal"]["selection_policy"], "fixed")
            self.assertEqual(episode["start_goal"]["start_cell"], [1, 1])
            self.assertEqual(episode["start_goal"]["goal_cell"], [3, 1])


if __name__ == "__main__":
    unittest.main()
