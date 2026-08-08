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
        self.assertFalse(config["observation"]["include_dynamic_map"])
        self.assertFalse(config["observation"]["include_location_sensing"])
        self.assertFalse(config["observation"]["include_wall_sensing"])
        self.assertEqual(config["observation"]["wall_sensing_version"], "v3")

    def test_seed_map_sections_and_explicit_map_shape_are_preserved(self):
        seed_map_train = {
            "enabled": True,
            "dataset_path": "/tmp/seed-map-corpus",
            "seed_ranges": [[1, 101]],
            "seed_count": 100,
            "trajectories_per_seed": 50,
            "selection_seed": 0,
            "split_unit": "trajectory",
        }
        seed_map_eval = {
            "enabled": True,
            "seed_ranges": [[1, 101], [1001, 1051]],
            "seed_count": 150,
            "seed_map_size_mode": "random",
            "seed_map_min_size": 9,
            "seed_map_max_size": 13,
        }
        config = normalize_baseline_config(
            {
                "algorithm": "crl",
                "seed_map_train": seed_map_train,
                "seed_map_eval": seed_map_eval,
                "observation": {"map_shape": [13, 16]},
            }
        )
        self.assertEqual(config["seed_map_train"], seed_map_train)
        self.assertEqual(config["seed_map_eval"], seed_map_eval)
        self.assertEqual(config["observation"]["map_shape"], [13, 16])

    def test_seed_map_enabled_must_be_boolean(self):
        with self.assertRaisesRegex(ValueError, "seed_map_train.enabled"):
            normalize_baseline_config(
                {
                    "algorithm": "crl",
                    "seed_map_train": {"enabled": 1},
                }
            )

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

    def test_resume_is_available_to_all_d3rlpy_algorithms(self):
        resume = {
            "checkpoint": "/tmp/step_100000.d3",
            "trainer_state": "/tmp/trainer_state_step_100000.pt",
        }
        for algorithm in ("mlp_bc", "td3_bc", "iql", "rebrac"):
            with self.subTest(algorithm=algorithm):
                config = normalize_baseline_config(
                    {
                        "algorithm": algorithm,
                        "train_variants": ["umaze"],
                        "resume": resume,
                    }
                )
                self.assertEqual(config["resume"], resume)

    def test_resume_rejects_jax_algorithms(self):
        with self.assertRaisesRegex(ValueError, "d3rlpy algorithms"):
            normalize_baseline_config(
                {
                    "algorithm": "crl",
                    "train_variants": ["umaze"],
                    "resume": {
                        "checkpoint": "/tmp/step_100000.msgpack",
                        "trainer_state": "/tmp/trainer_state_step_100000.pt",
                    },
                }
            )

    def test_gcrl_early_stopping_is_normalized(self):
        config = normalize_baseline_config(
            {
                "algorithm": "crl",
                "train_variants": ["umaze"],
                "n_steps": 2_000_000,
                "early_stopping": {
                    "enabled": True,
                    "min_steps": 1_000_000,
                    "patience_evaluations": 4,
                    "min_delta": 0.02,
                },
            }
        )
        self.assertEqual(
            config["early_stopping"],
            {
                "enabled": True,
                "min_steps": 1_000_000,
                "patience_evaluations": 4,
                "min_delta": 0.02,
            },
        )

    def test_early_stopping_rejects_non_gcrl_and_disabled_eval(self):
        with self.assertRaisesRegex(ValueError, "only by CRL/HIQL"):
            normalize_baseline_config(
                {
                    "algorithm": "mlp_bc",
                    "train_variants": ["umaze"],
                    "early_stopping": {"enabled": True},
                }
            )
        with self.assertRaisesRegex(ValueError, "evaluation.enabled=true"):
            normalize_baseline_config(
                {
                    "algorithm": "hiql",
                    "train_variants": ["umaze"],
                    "evaluation": {"enabled": False},
                    "early_stopping": {"enabled": True},
                }
            )

    def test_rebrac_regularization_coefficients_must_be_nonnegative(self):
        config = normalize_baseline_config(
            {
                "algorithm": "rebrac",
                "train_variants": ["umaze"],
                "algorithm_config": {"actor_beta": 0.0, "critic_beta": 0.01},
            }
        )
        self.assertEqual(config["algorithm_config"]["actor_beta"], 0.0)
        with self.assertRaisesRegex(ValueError, "actor_beta"):
            normalize_baseline_config(
                {
                    "algorithm": "rebrac",
                    "train_variants": ["umaze"],
                    "algorithm_config": {"actor_beta": -1.0},
                }
            )

    def test_observation_components_are_independent(self):
        config = normalize_baseline_config(
            {
                "algorithm": "mlp_bc",
                "train_variants": ["umaze"],
                "observation": {
                    "include_map": True,
                    "include_dynamic_map": True,
                    "include_location_sensing": False,
                    "include_wall_sensing": True,
                    "wall_sensing_version": "v5",
                    "map_sensing_boundary_risk_threshold": 0.2,
                },
            }
        )
        self.assertTrue(config["observation"]["include_map"])
        self.assertTrue(config["observation"]["include_dynamic_map"])
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

    def test_crl_defaults_and_network_are_normalized(self):
        config = normalize_baseline_config(
            {
                "algorithm": "crl",
                "train_variants": ["umaze"],
                "network": {
                    "hidden_units": [32, 32],
                    "activation": "gelu",
                    "use_layer_norm": True,
                },
            }
        )
        self.assertEqual(config["algorithm_config"]["batch_size"], 1024)
        self.assertEqual(config["algorithm_config"]["latent_dim"], 512)
        self.assertEqual(config["algorithm_config"]["discount"], 0.99)
        self.assertTrue(config["algorithm_config"]["value_geom_sample"])
        self.assertTrue(config["network"]["use_layer_norm"])
        self.assertEqual(
            config["evaluation"]["env_config"]["env_kwargs"],
            {"continuing_task": False, "reset_target": False},
        )

    def test_gcrl_rejects_conflicting_multi_goal_rollout_config(self):
        with self.assertRaisesRegex(ValueError, "single-goal"):
            normalize_baseline_config(
                {
                    "algorithm": "crl",
                    "train_variants": ["umaze"],
                    "evaluation": {
                        "env_config": {"env_kwargs": {"continuing_task": True}}
                    },
                }
            )

    def test_gcrl_goal_probabilities_must_sum_to_one(self):
        with self.assertRaisesRegex(ValueError, "probabilities must sum to 1"):
            normalize_baseline_config(
                {
                    "algorithm": "hiql",
                    "train_variants": ["umaze"],
                    "algorithm_config": {
                        "value_p_curgoal": 0.2,
                        "value_p_trajgoal": 0.2,
                        "value_p_randomgoal": 0.2,
                    },
                }
            )

    def test_crl_rejects_singleton_contrastive_batch(self):
        with self.assertRaisesRegex(ValueError, "batch_size >= 2"):
            normalize_baseline_config(
                {
                    "algorithm": "crl",
                    "train_variants": ["umaze"],
                    "algorithm_config": {"batch_size": 1},
                }
            )


if __name__ == "__main__":
    unittest.main()
