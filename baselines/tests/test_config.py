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


if __name__ == "__main__":
    unittest.main()
