from __future__ import annotations

import unittest

from baselines.config import normalize_baseline_config
from baselines.registry import resolve_baseline_selections


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


if __name__ == "__main__":
    unittest.main()
