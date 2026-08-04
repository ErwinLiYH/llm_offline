from __future__ import annotations

import unittest

import numpy as np

from baselines.config import normalize_baseline_config
from baselines.data.observation import (
    GOAL_CONDITIONED_STATE_DIMS,
    GoalConditionedObservationWrapper,
    family_map_shape,
)
from baselines.evaluation import _make_evaluation_env


class GCRLRolloutObservationTest(unittest.TestCase):
    def _env(self, env_family: str) -> GoalConditionedObservationWrapper:
        config = normalize_baseline_config(
            {
                "algorithm": "crl",
                "env_family": env_family,
                "train_variants": ["umaze"],
                "device": "cpu",
                "observation": {
                    "include_map": True,
                    "include_dynamic_map": True,
                    "include_location_sensing": True,
                    "include_wall_sensing": True,
                },
            }
        )
        return _make_evaluation_env(
            env_family=env_family,
            variant="umaze",
            reward_type="sparse",
            env_config=config["evaluation"]["env_config"],
            observation_config=config["observation"],
            goal_conditioned=True,
        )

    def test_point_and_ant_full_goal_capture_is_repeatable_and_contains_success_map(self):
        expected_dims = {"pointmaze": 460, "antmaze": 419}
        for env_family in ("pointmaze", "antmaze"):
            with self.subTest(env_family=env_family):
                env = self._env(env_family)
                try:
                    first, _ = env.reset(seed=17)
                    second, _ = env.reset(seed=17)
                    self.assertEqual(first["state"].shape, (expected_dims[env_family],))
                    self.assertEqual(first["goal"].shape, first["state"].shape)
                    self.assertTrue(np.all(np.isfinite(first["goal"])))
                    np.testing.assert_array_equal(first["state"], second["state"])
                    np.testing.assert_array_equal(first["goal"], second["goal"])

                    goal_xy = np.asarray(env.last_crossmaze_state["goal_xy"])
                    np.testing.assert_allclose(first["goal"][:2], goal_xy, atol=1e-6)
                    if env_family == "antmaze":
                        self.assertEqual(first["goal"][:29].shape, (29,))

                    rows, cols = family_map_shape(env_family)
                    map_dim = rows * cols
                    dynamic_start = GOAL_CONDITIONED_STATE_DIMS[env_family] + map_dim
                    dynamic_map = first["goal"][
                        dynamic_start : dynamic_start + map_dim
                    ]
                    self.assertEqual(np.count_nonzero(dynamic_map == 4), 1)
                    self.assertEqual(np.count_nonzero(dynamic_map == 2), 0)
                    self.assertEqual(np.count_nonzero(dynamic_map == 3), 0)

                    base_env = env.env.unwrapped
                    agent_env = (
                        base_env.point_env
                        if env_family == "pointmaze"
                        else base_env.ant_env
                    )
                    qpos = base_env.data.qpos.copy()
                    qvel = base_env.data.qvel.copy()
                    qpos[:2] = goal_xy
                    agent_env.set_state(qpos, qvel)
                    _observation, _reward, terminated, _truncated, info = env.step(
                        np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
                    )
                    self.assertTrue(info["success"])
                    self.assertTrue(terminated)
                finally:
                    env.close()

    def test_point_and_ant_fresh_seed_maps_use_resolved_dmap_slots(self):
        cases = {
            "pointmaze": {
                "map_shape": [15, 15],
                "state_dim": 229,
                "spec": {
                    "version": "v1",
                    "size_mode": "random",
                    "min_size": 9,
                    "max_size": 15,
                    "fixed_rows": None,
                    "fixed_cols": None,
                },
            },
            "antmaze": {
                "map_shape": [13, 16],
                "state_dim": 237,
                "spec": {
                    "version": "v1",
                    "size_mode": "random",
                    "min_size": 9,
                    "max_size": 13,
                    "fixed_rows": None,
                    "fixed_cols": None,
                },
            },
        }
        for env_family, case in cases.items():
            with self.subTest(env_family=env_family):
                config = normalize_baseline_config(
                    {
                        "algorithm": "crl",
                        "env_family": env_family,
                        "train_variants": ["umaze"],
                        "observation": {
                            "include_dynamic_map": True,
                            "map_shape": case["map_shape"],
                        },
                    }
                )
                env = _make_evaluation_env(
                    env_family=env_family,
                    variant="seed-map-1001",
                    reward_type="sparse",
                    env_config={"eval_start_goal_mode": "random-start-goal"},
                    observation_config=config["observation"],
                    goal_conditioned=True,
                    seed_map_target={
                        "map_seed": 1001,
                        "seed_map_spec": case["spec"],
                        "reward_type": "sparse",
                        "max_episode_steps": 20,
                    },
                )
                try:
                    observation, _ = env.reset(seed=0)
                    self.assertEqual(
                        observation["state"].shape,
                        (case["state_dim"],),
                    )
                    self.assertEqual(
                        observation["goal"].shape,
                        observation["state"].shape,
                    )
                    self.assertTrue(np.all(np.isfinite(observation["state"])))
                    self.assertTrue(np.all(np.isfinite(observation["goal"])))
                    base_dim = GOAL_CONDITIONED_STATE_DIMS[env_family]
                    goal_dmap = observation["goal"][base_dim:]
                    self.assertEqual(len(goal_dmap), np.prod(case["map_shape"]))
                    self.assertEqual(np.count_nonzero(goal_dmap == 4), 1)
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
