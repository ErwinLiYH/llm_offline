from __future__ import annotations

import unittest

import numpy as np

from baselines.gcrl.data import GCRLDataset, GCRLEpisode, GCRLNormalizer


def _episode(variant: str, value: float, length: int = 5) -> GCRLEpisode:
    times = np.arange(length + 1, dtype=np.float32)[:, None]
    states = np.concatenate(
        [np.full((length + 1, 2), value, dtype=np.float32), times], axis=-1
    )
    goals = np.concatenate(
        [np.full((length + 1, 1), value, dtype=np.float32), times], axis=-1
    )
    return GCRLEpisode(
        variant=variant,
        states=states,
        goals=goals,
        actions=np.full((length, 2), value, dtype=np.float32),
    )


def _sampling_config(**overrides):
    config = {
        "discount": 0.99,
        "subgoal_steps": 2,
        "value_p_curgoal": 0.0,
        "value_p_trajgoal": 1.0,
        "value_p_randomgoal": 0.0,
        "value_geom_sample": True,
        "actor_p_curgoal": 0.0,
        "actor_p_trajgoal": 1.0,
        "actor_p_randomgoal": 0.0,
        "actor_geom_sample": False,
    }
    config.update(overrides)
    return config


class GCRLDatasetTest(unittest.TestCase):
    def test_each_contrastive_batch_is_variant_homogeneous(self):
        dataset = GCRLDataset(
            [_episode("map-a", 0.1), _episode("map-b", 0.9)], seed=7
        )
        observed = set()
        for _ in range(20):
            batch = dataset.sample(32, algorithm="crl", config=_sampling_config())
            unique_actions = np.unique(batch["actions"])
            self.assertEqual(len(unique_actions), 1)
            observed.add(round(float(unique_actions[0]), 1))
        self.assertEqual(observed, {0.1, 0.9})

    def test_current_goal_rewards_and_masks_match_ogbench_contract(self):
        dataset = GCRLDataset([_episode("map", 0.2)], seed=3)
        batch = dataset.sample(
            8,
            algorithm="crl",
            config=_sampling_config(
                value_p_curgoal=1.0,
                value_p_trajgoal=0.0,
                value_p_randomgoal=0.0,
            ),
        )
        np.testing.assert_array_equal(batch["rewards"], np.ones(8))
        np.testing.assert_array_equal(batch["masks"], np.zeros(8))

    def test_hiql_batch_keeps_state_and_goal_targets_separate(self):
        dataset = GCRLDataset([_episode("map", 0.2)], seed=3)
        batch = dataset.sample(8, algorithm="hiql", config=_sampling_config())
        self.assertEqual(batch["high_actor_target_states"].shape, (8, 3))
        self.assertEqual(batch["high_actor_target_goals"].shape, (8, 2))
        self.assertEqual(batch["low_actor_goals"].shape, (8, 2))
        self.assertEqual(batch["rewards"].shape, (8,))

    def test_normalizer_handles_constant_features(self):
        dataset = GCRLDataset([_episode("map", 0.2)], seed=3)
        normalizer = GCRLNormalizer.fit(dataset)
        self.assertTrue(np.all(np.isfinite(normalizer.state_std)))
        self.assertTrue(np.all(normalizer.state_std > 0))
        self.assertTrue(np.all(np.isfinite(normalizer.goal_std)))

    def test_static_map_is_compact_but_restored_in_original_state_order(self):
        episode = _episode("map", 0.2, length=4)
        # Stored states omit the two map values between base state (two dims)
        # and the location-like tail (one dim).  Samples restore the exact
        # [base, map, tail] order expected by online vectorization.
        compact_episode = GCRLEpisode(
            variant=episode.variant,
            states=episode.states,
            goals=episode.goals,
            actions=episode.actions,
            map_features=np.asarray([7.0, 8.0], dtype=np.float32),
            map_insert_index=2,
        )
        dataset = GCRLDataset([compact_episode], seed=3)
        self.assertEqual(dataset.state_dim, 5)
        batch = dataset.sample(8, algorithm="hiql", config=_sampling_config())
        np.testing.assert_array_equal(batch["observations"][:, 2:4], [[7.0, 8.0]] * 8)
        np.testing.assert_array_equal(
            batch["next_observations"][:, 2:4], [[7.0, 8.0]] * 8
        )
        np.testing.assert_array_equal(
            batch["high_actor_target_states"][:, 2:4], [[7.0, 8.0]] * 8
        )
        normalizer = GCRLNormalizer.fit(dataset)
        np.testing.assert_array_equal(normalizer.state_mean[2:4], [7.0, 8.0])
        np.testing.assert_array_equal(normalizer.state_std[2:4], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
