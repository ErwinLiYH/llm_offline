from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from baselines.evaluation import BaselineEpochCallback, evaluate_rollouts

try:
    from baselines.trainer_state import load_trainer_state

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    load_trainer_state = None
    _TORCH_AVAILABLE = False


class _PredictZero:
    def predict(self, observations):
        return np.zeros((len(observations), 2), dtype=np.float32)


class _PredictGoalConditionedZero:
    def predict(self, observations):
        assert set(observations) == {"state", "goal"}
        assert observations["state"].ndim == 2
        assert observations["goal"].ndim == 2
        assert observations["state"].shape == observations["goal"].shape
        return np.zeros((len(observations["state"]), 2), dtype=np.float32)


class _PredictRecordingZero:
    def __init__(self):
        self.batch_sizes = []

    def predict(self, observations):
        self.batch_sizes.append(len(observations))
        return np.zeros((len(observations), 2), dtype=np.float32)


class _SaveOnlyAlgo:
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")


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
        self.goal = np.array([3.0, 1.0], dtype=np.float64)
        self.data = type(
            "Data", (), {"qpos": np.zeros(2), "qvel": np.zeros(2)}
        )()
        self.point_env = self

    def set_state(self, qpos, qvel):
        self.data.qpos = np.asarray(qpos, dtype=np.float64).copy()
        self.data.qvel = np.asarray(qvel, dtype=np.float64).copy()

    def _get_obs(self, raw=None):
        if raw is None:
            return np.concatenate([self.data.qpos, self.data.qvel])
        return {
            "observation": np.asarray(raw, dtype=np.float32),
            "achieved_goal": np.asarray(raw[:2], dtype=np.float32),
            "desired_goal": self.goal.astype(np.float32),
        }

    def _enrich(self, observation):
        observation = dict(observation)
        position = observation["observation"][:2]
        position_cell = (
            [3, 1]
            if np.array_equal(position, self.goal.astype(np.float32))
            else [1, int(round(position[0]))]
        )
        goal_cell = [3, 1]
        maze_map = [
            [1] * 5,
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1] * 5,
        ]
        observation["crossmaze"] = {
            "maze_map": maze_map,
            "maze_size_scaling": 1.0,
            "position_cell": position_cell,
            "goal_cell": goal_cell,
            "position_xy": position.astype(float).tolist(),
            "goal_xy": self.goal.astype(float).tolist(),
        }
        return observation

    def _observation(self):
        start_col = 1 + self._seed % 2
        self.data.qpos = np.array([start_col, 1.0], dtype=np.float64)
        self.data.qvel = np.zeros(2, dtype=np.float64)
        return self._enrich(self._get_obs(self._get_obs()))

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
    @unittest.skipUnless(_TORCH_AVAILABLE, "requires the d3rlpy baseline environment")
    def test_epoch_callback_saves_matching_trainer_state(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = BaselineEpochCallback(
                config={
                    "save_interval_epochs": 1,
                    "evaluation": {"enabled": False},
                },
                selections=None,
                validation_buffer=None,
                action_probe=None,
                run_dir=Path(directory),
                total_epochs=1,
            )
            callback(_SaveOnlyAlgo(), epoch=1, total_step=25_000)

            checkpoint = Path(directory) / "checkpoints/step_25000.d3"
            trainer_state_path = (
                Path(directory) / "checkpoints/trainer_state_step_25000.pt"
            )
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(trainer_state_path.is_file())
            trainer_state = load_trainer_state(trainer_state_path)
            self.assertEqual(trainer_state["epoch"], 1)
            self.assertEqual(trainer_state["step"], 25_000)

    @unittest.skipUnless(_TORCH_AVAILABLE, "requires the d3rlpy baseline environment")
    def test_epoch_callback_applies_resume_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = BaselineEpochCallback(
                config={
                    "save_interval_epochs": 4,
                    "evaluation": {"enabled": False},
                },
                selections=None,
                validation_buffer=None,
                action_probe=None,
                run_dir=Path(directory),
                total_epochs=4,
                epoch_offset=40,
                step_offset=1_000_000,
            )
            callback(_SaveOnlyAlgo(), epoch=4, total_step=100_000)

            checkpoint = Path(directory) / "checkpoints/step_1100000.d3"
            trainer_state = load_trainer_state(
                Path(directory)
                / "checkpoints/trainer_state_step_1100000.pt"
            )
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(trainer_state["epoch"], 44)
            self.assertEqual(trainer_state["step"], 1_100_000)

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

    @patch("baselines.evaluation.crossmaze.make", return_value=_EpisodeRecordEnv())
    def test_goal_conditioned_policy_receives_separate_batched_arrays(self, _make):
        result = evaluate_rollouts(
            _PredictGoalConditionedZero(),
            env_family="pointmaze",
            variants=["umaze"],
            reward_types={"umaze": "sparse"},
            evaluation_config={"seed": 10, "num_episodes": 1, "env_config": {}},
            observation_config={
                "include_map": False,
                "include_location_sensing": False,
                "include_wall_sensing": False,
                "wall_sensing_version": "v3",
                "map_sensing_boundary_risk_threshold": 0.1,
            },
            goal_conditioned=True,
        )
        self.assertEqual(result["aggregate"]["num_episodes"], 1)

    @patch(
        "baselines.evaluation.crossmaze.make",
        side_effect=lambda *_args, **_kwargs: _EpisodeRecordEnv(),
    )
    def test_rollout_batch_keeps_per_episode_records(self, _make):
        policy = _PredictRecordingZero()
        result = evaluate_rollouts(
            policy,
            env_family="pointmaze",
            variants=["umaze"],
            reward_types={"umaze": "sparse"},
            evaluation_config={
                "seed": 10,
                "num_episodes": 2,
                "rollout_batch_size": 2,
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

        self.assertEqual(policy.batch_sizes, [2, 2, 2])
        episodes = result["variants"]["umaze"]["episodes"]
        self.assertEqual([episode["seed"] for episode in episodes], [10, 11])
        self.assertTrue(episodes[0]["success"])
        self.assertFalse(episodes[1]["success"])


if __name__ == "__main__":
    unittest.main()
