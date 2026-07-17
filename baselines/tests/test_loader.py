from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from baselines.config import normalize_baseline_config
from baselines.data.loader import LoadedVariant, prepare_datasets


def _raw_episode(seed: int, length: int = 3):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        observations={
            "observation": rng.normal(size=(length + 1, 4)),
            "achieved_goal": rng.normal(size=(length + 1, 2)),
            "desired_goal": rng.normal(size=(length + 1, 2)),
        },
        actions=np.tanh(rng.normal(size=(length, 2))),
        rewards=rng.normal(size=length),
        terminations=np.array([False] * (length - 1) + [True]),
        truncations=np.zeros(length, dtype=bool),
    )


def _loaded(variant: str, count: int):
    return LoadedVariant(
        variant=variant,
        source=f"test:{variant}",
        reward_type="sparse",
        episodes=[_raw_episode(index) for index in range(count)],
        dataset_path=f"/test/{variant}",
        warnings=[],
    )


class LoaderTest(unittest.TestCase):
    def _config(self, **overrides):
        raw = {
            "algorithm": "mlp_bc",
            "train_mode": "all",
            "train_variants": ["local-layout-01", "local-layout-02"],
            "device": False,
            "train_data_ratio": 0.8,
        }
        raw.update(overrides)
        return normalize_baseline_config(raw)

    @patch("baselines.data.loader.load_variant_episodes")
    def test_balancing_and_split_are_deterministic(self, load):
        load.side_effect = lambda _family, variant, **_kwargs: _loaded(
            variant, 8 if variant == "local-layout-01" else 5
        )
        config = self._config(balance_variant_episode_count=True)
        reward_types = {
            "local-layout-01": "sparse",
            "local-layout-02": "sparse",
        }
        first = prepare_datasets(config, list(reward_types), reward_types)
        second = prepare_datasets(config, list(reward_types), reward_types)

        self.assertEqual(first.manifest["balanced_episode_target"], 5)
        self.assertEqual(first.manifest["train_episode_count"], 8)
        self.assertEqual(first.manifest["validation_episode_count"], 2)
        self.assertEqual(first.train_buffer.transition_count, 24)
        self.assertEqual(first.validation_buffer.transition_count, 6)
        self.assertEqual(first.manifest["observation_schema"]["dimension"], 6)
        self.assertEqual(
            first.manifest["variants"]["local-layout-01"]["train_episode_indices"],
            second.manifest["variants"]["local-layout-01"]["train_episode_indices"],
        )

    @patch("baselines.data.loader.load_variant_episodes")
    def test_per_variant_keep_disables_balancing(self, load):
        load.side_effect = lambda _family, variant, **_kwargs: _loaded(variant, 8)
        config = self._config(
            balance_variant_episode_count=True,
            episode_keep_per_variant={
                "local-layout-01": 5,
                "local-layout-02": 6,
            },
        )
        reward_types = {
            "local-layout-01": "sparse",
            "local-layout-02": "sparse",
        }
        prepared = prepare_datasets(config, list(reward_types), reward_types)
        self.assertFalse(prepared.manifest["balance_variant_episode_count"])
        self.assertIn("was ignored", prepared.manifest["warnings"][0])


if __name__ == "__main__":
    unittest.main()
