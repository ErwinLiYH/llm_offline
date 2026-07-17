from __future__ import annotations

import unittest

from baselines.config import normalize_baseline_config


class BaselineConfigTest(unittest.TestCase):
    def test_normalizes_minimal_config(self):
        config = normalize_baseline_config(
            {
                "algorithm": "mlp_bc",
                "train_variants": ["umaze"],
                "device": False,
            }
        )
        self.assertEqual(config["env_family"], "pointmaze")
        self.assertEqual(config["n_steps"], 1_000_000)
        self.assertEqual(config["evaluation"]["every_epochs"], 10)
        self.assertFalse(config["observation"]["include_map"])
        self.assertFalse(config["observation"]["include_location_sensing"])
        self.assertFalse(config["observation"]["include_wall_sensing"])
        self.assertEqual(config["observation"]["wall_sensing_version"], "v3")

    def test_rejects_unknown_keys(self):
        with self.assertRaisesRegex(ValueError, "Unknown baseline config keys"):
            normalize_baseline_config(
                {"algorithm": "iql", "train_variants": ["umaze"], "typo": 1}
            )

    def test_rejects_non_divisible_update_groups(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            normalize_baseline_config(
                {
                    "algorithm": "td3_bc",
                    "train_variants": ["umaze"],
                    "n_steps": 11,
                    "n_steps_per_epoch": 10,
                }
            )

    def test_reward_type_must_be_top_level(self):
        with self.assertRaisesRegex(ValueError, "top level"):
            normalize_baseline_config(
                {
                    "algorithm": "iql",
                    "train_variants": ["umaze"],
                    "evaluation": {"env_config": {"reward_type": "dense"}},
                }
            )

    def test_nested_env_reward_type_is_rejected_early(self):
        with self.assertRaisesRegex(ValueError, "top level"):
            normalize_baseline_config(
                {
                    "algorithm": "td3_bc",
                    "train_variants": ["umaze"],
                    "evaluation": {
                        "env_config": {"env_kwargs": {"reward_type": "dense"}}
                    },
                }
            )

    def test_bc_rejects_irrelevant_gamma(self):
        with self.assertRaisesRegex(ValueError, "Unknown algorithm_config"):
            normalize_baseline_config(
                {
                    "algorithm": "mlp_bc",
                    "train_variants": ["umaze"],
                    "algorithm_config": {"gamma": 0.99},
                }
            )

    def test_observation_components_are_independent(self):
        config = normalize_baseline_config(
            {
                "algorithm": "mlp_bc",
                "train_variants": ["umaze"],
                "observation": {
                    "include_map": True,
                    "include_location_sensing": False,
                    "include_wall_sensing": True,
                    "wall_sensing_version": "v5",
                    "map_sensing_boundary_risk_threshold": 0.2,
                },
            }
        )
        self.assertTrue(config["observation"]["include_map"])
        self.assertFalse(config["observation"]["include_location_sensing"])
        self.assertTrue(config["observation"]["include_wall_sensing"])
        self.assertEqual(config["observation"]["wall_sensing_version"], "v5")
        self.assertEqual(
            config["observation"]["map_sensing_boundary_risk_threshold"], 0.2
        )

    def test_sensing_must_not_be_configured_only_for_eval(self):
        with self.assertRaisesRegex(ValueError, "under observation"):
            normalize_baseline_config(
                {
                    "algorithm": "mlp_bc",
                    "train_variants": ["umaze"],
                    "evaluation": {
                        "env_config": {"wall_sensing_version": "v5"}
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
