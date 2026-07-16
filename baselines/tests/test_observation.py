from __future__ import annotations

import unittest
from collections import UserDict

import numpy as np

from baselines.data.observation import vectorize_observation


class ObservationTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
