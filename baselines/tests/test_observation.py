from __future__ import annotations

import unittest
from collections import UserDict

import numpy as np

from baselines.data.observation import (
    MAP_PADDING_VALUE,
    family_map_shape,
    observation_dim,
    vectorize_observation,
)
from crossmaze import compute_sensing_state, get_env_facts


class ObservationTest(unittest.TestCase):
    def _structured_config(self, **overrides):
        config = {
            "include_map": False,
            "include_location_sensing": False,
            "include_wall_sensing": False,
            "wall_sensing_version": "v5",
            "map_sensing_boundary_risk_threshold": 0.1,
        }
        config.update(overrides)
        return config

    def test_pointmaze_order(self):
        observation = UserDict(
            {
                "observation": np.array([1, 2, 3, 4], dtype=np.float64),
                "achieved_goal": np.array([9, 9]),
                "desired_goal": np.array([5, 6]),
            }
        )
        result = vectorize_observation(observation, "pointmaze")
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 5, 6])
        self.assertEqual(result.dtype, np.float32)

    def test_antmaze_order(self):
        observation = {
            "achieved_goal": np.array([1, 2]),
            "observation": np.arange(3, 30),
            "desired_goal": np.array([30, 31]),
        }
        result = vectorize_observation(observation, "antmaze")
        np.testing.assert_array_equal(result, np.arange(1, 32))

    def test_rejects_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "expected 6"):
            vectorize_observation(
                {"observation": np.zeros(3), "desired_goal": np.zeros(2)},
                "pointmaze",
            )

    def test_components_are_independently_configurable(self):
        self.assertEqual(family_map_shape("pointmaze"), (15, 15))
        self.assertEqual(family_map_shape("antmaze"), (12, 16))
        self.assertEqual(
            observation_dim(
                "pointmaze", self._structured_config(include_map=True)
            ),
            231,
        )
        self.assertEqual(
            observation_dim(
                "pointmaze",
                self._structured_config(include_location_sensing=True),
            ),
            10,
        )
        self.assertEqual(
            observation_dim(
                "pointmaze", self._structured_config(include_wall_sensing=True)
            ),
            10,
        )
        self.assertEqual(
            observation_dim(
                "pointmaze",
                self._structured_config(
                    include_map=True,
                    include_location_sensing=True,
                    include_wall_sensing=True,
                ),
            ),
            239,
        )
        self.assertEqual(
            observation_dim(
                "antmaze",
                self._structured_config(
                    include_map=True,
                    include_location_sensing=True,
                    include_wall_sensing=True,
                ),
            ),
            231,
        )

    def test_map_location_and_wall_features_match_crossmaze(self):
        facts = get_env_facts("pointmaze", "umaze")
        observation = {
            "observation": np.array([0.0, 1.0, 0.25, -0.5]),
            "desired_goal": np.array([1.0, -1.0]),
        }
        config = self._structured_config(
            include_map=True,
            include_location_sensing=True,
            include_wall_sensing=True,
        )
        result = vectorize_observation(
            observation,
            "pointmaze",
            observation_config=config,
            variant="umaze",
        )
        expected_sensing = compute_sensing_state(
            observation["observation"],
            observation["desired_goal"],
            {
                "maze_map": facts["maze_map"],
                "maze_size_scaling": facts["maze_size_scaling"],
                "wall_sensing_version": "v5",
                "map_sensing_boundary_risk_threshold": 0.1,
            },
        )
        np.testing.assert_array_equal(result[:6], [0, 1, 0.25, -0.5, 1, -1])
        map_slot = result[6:231].reshape(15, 15)
        np.testing.assert_array_equal(
            map_slot[:5, :5], np.asarray(facts["maze_map"], dtype=np.float32)
        )
        self.assertTrue(np.all(map_slot[5:, :] == MAP_PADDING_VALUE))
        self.assertTrue(np.all(map_slot[:5, 5:] == MAP_PADDING_VALUE))
        np.testing.assert_array_equal(
            result[231:235],
            expected_sensing["position_cell"] + expected_sensing["goal_cell"],
        )
        np.testing.assert_array_equal(
            result[235:239], expected_sensing["neighbor_status"]
        )

    def test_batched_offline_and_attached_online_layouts_match(self):
        facts = get_env_facts("pointmaze", "umaze")
        observations = {
            "observation": np.array(
                [[0.0, 1.0, 0.0, 0.0], [1.0, -1.0, 0.1, 0.2]],
                dtype=np.float32,
            ),
            "desired_goal": np.array([[1.0, -1.0], [0.0, 1.0]], dtype=np.float32),
        }
        attached = {
            **observations,
            "crossmaze": {
                "maze_map": facts["maze_map"],
                "maze_size_scaling": facts["maze_size_scaling"],
            },
        }
        config = self._structured_config(
            include_map=True,
            include_location_sensing=True,
            include_wall_sensing=True,
        )
        offline = vectorize_observation(
            observations,
            "pointmaze",
            observation_config=config,
            variant="umaze",
        )
        online = vectorize_observation(
            attached,
            "pointmaze",
            observation_config=config,
        )
        np.testing.assert_array_equal(offline, online)


if __name__ == "__main__":
    unittest.main()
