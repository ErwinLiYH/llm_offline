from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from baselines.config import normalize_baseline_config
from baselines.registry import resolve_baseline_selections
from baselines.tests.seed_map_fixture import make_seed_map_corpus
from crossmaze.seed_map import SeedMapSpec


class RegistryTest(unittest.TestCase):
    def _config(self, **overrides):
        raw = {
            "algorithm": "iql",
            "train_mode": "single",
            "train_variants": ["local-layout-01"],
        }
        raw.update(overrides)
        return normalize_baseline_config(raw)

    def test_local_dense_override_is_resolved(self):
        selections = resolve_baseline_selections(self._config(reward_type="dense"))
        self.assertEqual(
            selections.train_reward_types, {"local-layout-01": "dense"}
        )

    def test_remote_reward_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fixed 'sparse'"):
            resolve_baseline_selections(
                self._config(train_variants=["umaze"], reward_type="dense")
            )

    def test_mixed_remote_rewards_require_opt_in(self):
        config = self._config(
            train_mode="all",
            train_variants=["umaze", "umaze-dense"],
        )
        with self.assertRaisesRegex(ValueError, "mix reward types"):
            resolve_baseline_selections(config)

    def test_mixed_remote_rewards_can_be_explicit(self):
        selections = resolve_baseline_selections(
            self._config(
                train_mode="all",
                train_variants=["umaze", "umaze-dense"],
                allow_mixed_reward_types=True,
            )
        )
        self.assertEqual(
            selections.train_reward_types,
            {"umaze": "sparse", "umaze-dense": "dense"},
        )

    def test_seed_map_sections_override_legacy_variants_and_record_shape(self):
        with TemporaryDirectory() as directory:
            corpus = make_seed_map_corpus(Path(directory), env_family="pointmaze")
            config = self._config(
                algorithm="crl",
                train_variants=["umaze"],
                eval_mode="single",
                eval_variants=["umaze"],
                evaluation={"num_episodes": 7},
                seed_map_train={
                    "enabled": True,
                    "dataset_path": str(corpus),
                    "seed_ranges": [[1, 3]],
                    "seed_count": 2,
                    "trajectories_per_seed": 2,
                    "selection_seed": 0,
                    "split_unit": "trajectory",
                },
                seed_map_eval={
                    "enabled": True,
                    "seed_ranges": [[1, 3], [1001, 1003]],
                    "seed_count": 4,
                    "seed_map_size_mode": "fixed",
                    "seed_map_fixed_rows": 7,
                    "seed_map_fixed_cols": 7,
                },
            )
            selections = resolve_baseline_selections(config)

        self.assertEqual(
            selections.train.selected_variants,
            ["seed-map-1", "seed-map-2"],
        )
        self.assertEqual(
            selections.eval.selected_variants,
            ["seed-map-1", "seed-map-2", "seed-map-1001", "seed-map-1002"],
        )
        self.assertEqual(config["evaluation"]["num_episodes"], 7)
        self.assertEqual(config["observation"]["map_shape"], [15, 15])
        self.assertEqual(config["resolved_seed_map_train"]["seed_count"], 2)
        self.assertEqual(config["resolved_seed_map_eval"]["seed_count"], 4)

    def test_ant_seed_maps_expand_the_legacy_row_slot(self):
        spec = SeedMapSpec(size_mode="random", min_size=9, max_size=13)
        with TemporaryDirectory() as directory:
            corpus = make_seed_map_corpus(
                Path(directory),
                env_family="antmaze",
                seed_map_spec=spec,
            )
            config = self._config(
                algorithm="hiql",
                env_family="antmaze",
                seed_map_train={
                    "enabled": True,
                    "dataset_path": str(corpus),
                    "seed_ranges": [[1, 3]],
                    "seed_count": 2,
                    "trajectories_per_seed": 2,
                    "split_unit": "seed",
                },
                seed_map_eval={
                    "enabled": True,
                    "seed_ranges": [[1001, 1003]],
                    "seed_count": 2,
                    "seed_map_size_mode": "random",
                    "seed_map_min_size": 9,
                    "seed_map_max_size": 13,
                },
            )
            resolve_baseline_selections(config)

        self.assertEqual(config["observation"]["map_shape"], [13, 16])

    def test_seed_map_reward_conflict_is_rejected(self):
        with TemporaryDirectory() as directory:
            corpus = make_seed_map_corpus(Path(directory), env_family="pointmaze")
            config = self._config(
                algorithm="crl",
                reward_type="dense",
                seed_map_train={
                    "enabled": True,
                    "dataset_path": str(corpus),
                    "seed_ranges": [[1, 3]],
                    "seed_count": 2,
                    "trajectories_per_seed": 1,
                },
            )
            with self.assertRaisesRegex(ValueError, "corpus reward type conflicts"):
                resolve_baseline_selections(config)

    def test_disabled_seed_map_sections_preserve_legacy_selection(self):
        config = self._config(
            seed_map_train={"enabled": False},
            seed_map_eval={"enabled": False},
        )
        selections = resolve_baseline_selections(config)

        self.assertEqual(selections.train.selected_variants, ["local-layout-01"])
        self.assertEqual(selections.eval.selected_variants, ["local-layout-01"])
        self.assertIsNone(selections.seed_map_train)
        self.assertIsNone(selections.seed_map_eval)
        self.assertNotIn("map_shape", config["observation"])

    def test_train_only_seed_map_keeps_explicit_legacy_eval_fallback(self):
        with TemporaryDirectory() as directory:
            corpus = make_seed_map_corpus(Path(directory), env_family="pointmaze")
            config = self._config(
                algorithm="crl",
                seed_map_train={
                    "enabled": True,
                    "dataset_path": str(corpus),
                    "seed_ranges": [[1, 3]],
                    "seed_count": 2,
                    "trajectories_per_seed": 2,
                    "split_unit": "trajectory",
                },
            )
            selections = resolve_baseline_selections(config)

        self.assertEqual(
            selections.train.selected_variants,
            ["seed-map-1", "seed-map-2"],
        )
        self.assertEqual(selections.eval.selected_variants, ["local-layout-01"])
        self.assertIsNotNone(selections.seed_map_train)
        self.assertIsNone(selections.seed_map_eval)

    def test_eval_only_seed_map_keeps_legacy_training_data(self):
        config = self._config(
            algorithm="hiql",
            seed_map_eval={
                "enabled": True,
                "seed_ranges": [[1001, 1003]],
                "seed_count": 2,
                "seed_map_size_mode": "fixed",
                "seed_map_fixed_rows": 7,
                "seed_map_fixed_cols": 7,
            },
        )
        selections = resolve_baseline_selections(config)

        self.assertEqual(selections.train.selected_variants, ["local-layout-01"])
        self.assertEqual(
            selections.eval.selected_variants,
            ["seed-map-1001", "seed-map-1002"],
        )
        self.assertIsNone(selections.seed_map_train)
        self.assertIsNotNone(selections.seed_map_eval)

    def test_seed_map_resolution_rejects_too_small_explicit_map_slot(self):
        with TemporaryDirectory() as directory:
            corpus = make_seed_map_corpus(Path(directory), env_family="pointmaze")
            config = self._config(
                algorithm="crl",
                observation={"include_dynamic_map": True, "map_shape": [7, 7]},
                seed_map_train={
                    "enabled": True,
                    "dataset_path": str(corpus),
                    "seed_ranges": [[1, 3]],
                    "seed_count": 2,
                    "trajectories_per_seed": 2,
                },
            )
            with self.assertRaisesRegex(ValueError, "map_shape is too small"):
                resolve_baseline_selections(config)


if __name__ == "__main__":
    unittest.main()
