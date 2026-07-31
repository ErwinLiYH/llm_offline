from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from baselines.data.observation import vectorize_gcrl_state_observation
from baselines.gcrl.data import (
    GCRLDataset,
    GCRLEpisode,
    GCRLNormalizer,
    _convert_episode,
)


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

    def test_hiql_batch_uses_full_state_goals_and_one_high_target(self):
        dataset = GCRLDataset([_episode("map", 0.2)], seed=3)
        batch = dataset.sample(8, algorithm="hiql", config=_sampling_config())
        self.assertNotIn("high_actor_target_states", batch)
        self.assertNotIn("high_actor_target_goals", batch)
        for key in (
            "value_goals",
            "low_actor_goals",
            "high_actor_goals",
            "high_actor_targets",
        ):
            self.assertEqual(batch[key].shape, (8, 3))
        self.assertEqual(batch["rewards"].shape, (8,))

    def test_normalizer_handles_constant_features(self):
        dataset = GCRLDataset([_episode("map", 0.2)], seed=3)
        normalizer = GCRLNormalizer.fit(dataset)
        self.assertTrue(np.all(np.isfinite(normalizer.observation_std)))
        self.assertTrue(np.all(normalizer.observation_std > 0))
        self.assertNotIn("goal_mean", normalizer.to_dict())
        self.assertNotIn("goal_std", normalizer.to_dict())
        restored = GCRLNormalizer.from_dict(normalizer.to_dict())
        np.testing.assert_array_equal(
            restored.observation_mean, normalizer.observation_mean
        )
        with self.assertRaisesRegex(ValueError, "compact_xy"):
            GCRLNormalizer.from_dict(
                {
                    "version": 1,
                    "state_mean": [0.0],
                    "goal_mean": [0.0],
                }
            )

    def test_current_goals_remain_exactly_equal_after_normalization(self):
        dataset = GCRLDataset([_episode("map", 0.2)], seed=3)
        batch = dataset.sample(
            8,
            algorithm="crl",
            config=_sampling_config(
                value_p_curgoal=1.0,
                value_p_trajgoal=0.0,
                value_p_randomgoal=0.0,
                actor_p_curgoal=1.0,
                actor_p_trajgoal=0.0,
                actor_p_randomgoal=0.0,
            ),
        )
        normalizer = GCRLNormalizer.fit(dataset)
        normalized = normalizer.normalize_batch(batch)
        np.testing.assert_array_equal(
            normalized["observations"], normalized["value_goals"]
        )
        np.testing.assert_array_equal(
            normalized["observations"], normalized["actor_goals"]
        )

    def test_static_map_is_compact_but_restored_in_original_state_order(self):
        episode = _episode("map", 0.2, length=4)
        # Stored states omit the two map values between base state (two dims)
        # and the location-like tail (one dim).  Samples restore the exact
        # [base, map, tail] order expected by online vectorization.
        compact_episode = GCRLEpisode(
            variant=episode.variant,
            states=episode.states,
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
            batch["high_actor_targets"][:, 2:4], [[7.0, 8.0]] * 8
        )
        normalizer = GCRLNormalizer.fit(dataset)
        np.testing.assert_array_equal(normalizer.observation_mean[2:4], [7.0, 8.0])
        np.testing.assert_array_equal(normalizer.observation_std[2:4], [1.0, 1.0])

    def test_dynamic_map_compaction_and_exact_statistics(self):
        episode = _episode("map", 0.2, length=4)
        positions = np.asarray([0, 0, 1, 1, 0], dtype=np.int64)
        goals = np.asarray([1, 0, 1, 0, 0], dtype=np.int64)
        compact_episode = GCRLEpisode(
            variant=episode.variant,
            states=episode.states,
            actions=episode.actions,
            map_features=np.asarray([0.0, 1.0], dtype=np.float32),
            map_insert_index=2,
            dynamic_map_insert_index=4,
            dynamic_position_indices=positions,
            dynamic_goal_indices=goals,
        )
        dataset = GCRLDataset([compact_episode], seed=3)
        arrays = dataset.variants["map"]
        restored = dataset._restore_observation_features(
            arrays, np.arange(len(arrays.states))
        )
        self.assertEqual(restored.shape, (5, 7))
        np.testing.assert_array_equal(restored[:, 2:4], [[0.0, 1.0]] * 5)
        expected_dynamic = np.asarray(
            [[2, 3], [4, 1], [0, 4], [3, 2], [4, 1]], dtype=np.float32
        )
        np.testing.assert_array_equal(restored[:, 4:6], expected_dynamic)
        normalizer = GCRLNormalizer.fit(dataset)
        np.testing.assert_allclose(normalizer.observation_mean, restored.mean(axis=0))
        expected_std = restored.std(axis=0)
        expected_std[expected_std < 1e-6] = 1.0
        np.testing.assert_allclose(normalizer.observation_std, expected_std, atol=1e-6)

    def test_offline_compact_dynamic_map_restores_exact_vectorizer_rows(self):
        observations = {
            "observation": np.asarray(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.5, 1.0, 0.1, 0.0],
                    [1.0, -1.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "desired_goal": np.asarray([[1.0, -1.0]] * 3, dtype=np.float32),
        }
        config = {
            "include_map": True,
            "include_dynamic_map": True,
            "include_location_sensing": True,
            "include_wall_sensing": True,
            "wall_sensing_version": "v5",
            "map_sensing_boundary_risk_threshold": 0.1,
        }
        raw = SimpleNamespace(
            observations=observations,
            actions=np.zeros((2, 2), dtype=np.float32),
        )
        episode = _convert_episode(
            raw,
            env_family="pointmaze",
            variant="umaze",
            observation_config=config,
        )
        dataset = GCRLDataset([episode], seed=0)
        arrays = dataset.variants["umaze"]
        restored = dataset._restore_observation_features(
            arrays, np.arange(3, dtype=np.int64)
        )
        expected = vectorize_gcrl_state_observation(
            observations,
            "pointmaze",
            observation_config=config,
            variant="umaze",
        )
        np.testing.assert_array_equal(restored, expected)


if __name__ == "__main__":
    unittest.main()
