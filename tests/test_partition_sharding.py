import sys
import tempfile
import types
import unittest

import torch

unsloth_stub = types.ModuleType("unsloth")
unsloth_stub.FastLanguageModel = object()
sys.modules.setdefault("unsloth", unsloth_stub)

from data.pointmaze.dataset import (
    PointMazeDataset,
    select_variant_episode_indices,
    split_episode_segments_for_partitions,
)
from train import (
    _compute_partition_round_stats,
    _validation_cache_config,
    build_dataset_request,
    resolve_dataset_load_partitions,
)
from utils.distributed import DistributedContext
from utils.distributed_sampler import LocalShardPaddingSampler


def _fake_episode_loader(_variant, family_data_config=None):
    del family_data_config
    step_counts = [3, 5, 2, 7, 4, 6, 8, 1, 9, 10]
    episodes = [
        types.SimpleNamespace(actions=[None] * step_count)
        for step_count in step_counts
    ]
    return {}, episodes, step_counts


class EpisodeSplitTest(unittest.TestCase):
    def test_train_and_val_episode_indices_are_disjoint(self):
        selection = select_variant_episode_indices(
            "unit",
            train_data_ratio=0.6,
            episode_keep_num=7,
            sampling_seed=42,
            episode_loader=_fake_episode_loader,
        )

        train_indices = set(selection["train_indices"])
        val_indices = set(selection["val_indices"])

        self.assertTrue(train_indices.isdisjoint(val_indices))
        self.assertEqual(len(train_indices | val_indices), selection["sampled_episode_count"])
        self.assertEqual(len(train_indices), selection["train_episode_count"])
        self.assertEqual(len(val_indices), selection["val_episode_count"])

    def test_train_shards_use_only_train_timesteps_once(self):
        selection = select_variant_episode_indices(
            "unit",
            train_data_ratio=0.6,
            episode_keep_num=8,
            sampling_seed=7,
            episode_loader=_fake_episode_loader,
        )
        _, _, step_counts = _fake_episode_loader("unit")
        train_indices = set(selection["train_indices"])
        val_indices = set(selection["val_indices"])

        shards = split_episode_segments_for_partitions(
            list(selection["train_indices"]),
            step_counts,
            partition_count=3,
            prompt_count=1,
            variant="unit",
            sampling_seed=7,
        )

        covered_train_timesteps = set()
        for shard in shards:
            for segment in shard:
                episode_idx = int(segment["episode_idx"])
                self.assertIn(episode_idx, train_indices)
                self.assertNotIn(episode_idx, val_indices)
                for timestep in range(int(segment["start_t"]), int(segment["end_t"])):
                    key = (episode_idx, timestep)
                    self.assertNotIn(key, covered_train_timesteps)
                    covered_train_timesteps.add(key)

        expected_train_timesteps = {
            (episode_idx, timestep)
            for episode_idx in train_indices
            for timestep in range(step_counts[episode_idx])
        }
        self.assertEqual(covered_train_timesteps, expected_train_timesteps)


class EpisodeSegmentPlannerTest(unittest.TestCase):
    def test_segments_cover_each_timestep_once(self):
        step_counts = [3, 5, 2]
        shards = split_episode_segments_for_partitions(
            [0, 1, 2],
            step_counts,
            partition_count=4,
            prompt_count=2,
            variant="unit",
            sampling_seed=123,
        )

        covered = set()
        sample_counts = []
        for shard in shards:
            sample_counts.append(sum(segment["sample_count"] for segment in shard))
            for segment in shard:
                for timestep in range(segment["start_t"], segment["end_t"]):
                    key = (segment["episode_idx"], timestep)
                    self.assertNotIn(key, covered)
                    covered.add(key)

        expected = {
            (episode_idx, timestep)
            for episode_idx, count in enumerate(step_counts)
            for timestep in range(count)
        }
        self.assertEqual(covered, expected)
        self.assertLessEqual(max(sample_counts) - min(sample_counts), 2)

    def test_large_episode_can_be_split_across_shards(self):
        shards = split_episode_segments_for_partitions(
            [0],
            [10],
            partition_count=3,
            prompt_count=1,
            variant="unit",
            sampling_seed=0,
        )

        self.assertEqual(sum(len(shard) for shard in shards), 3)
        self.assertTrue(any(shard[0]["start_t"] > 0 for shard in shards[1:]))

    def test_empty_shards_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "dataset_load_partitions exceeds"):
            split_episode_segments_for_partitions(
                [0],
                [2],
                partition_count=3,
                prompt_count=1,
            )


class LocalShardPaddingSamplerTest(unittest.TestCase):
    def test_padding_is_deterministic_and_local(self):
        sampler = LocalShardPaddingSampler(3, num_samples=8, seed=7)
        first = list(iter(sampler))
        second = list(iter(LocalShardPaddingSampler(3, num_samples=8, seed=7)))

        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(set(first[:3]), {0, 1, 2})
        self.assertTrue(all(0 <= index < 3 for index in first))

    def test_weighted_sampler_stays_in_local_shard(self):
        sampler = LocalShardPaddingSampler(
            4,
            num_samples=10,
            seed=5,
            weights=[0.0, 1.0, 0.0, 0.0],
        )

        self.assertEqual(list(iter(sampler)), [1] * 10)


class PartitionRoundStatsTest(unittest.TestCase):
    def test_round_target_batches_use_max_local_batches(self):
        round_stats = _compute_partition_round_stats(
            [
                {"partition_index": 0, "train_samples": 5},
                {"partition_index": 1, "train_samples": 9},
                {"partition_index": 2, "train_samples": 1},
                {"partition_index": 3, "train_samples": 8},
            ],
            world_size=2,
            batch_size=4,
        )

        self.assertEqual(
            [(stat["partition_indices"], stat["target_batches"]) for stat in round_stats],
            [([0, 1], 3), ([2, 3], 2)],
        )

    def test_ddp_partition_count_must_align_to_world_size(self):
        context = DistributedContext(
            backend="ddp",
            world_size=2,
            is_distributed=True,
            device=torch.device("cpu"),
        )
        with tempfile.TemporaryDirectory() as cache_dir:
            with self.assertRaisesRegex(ValueError, "divisible by world_size"):
                resolve_dataset_load_partitions(
                    {
                        "dataset_load_partitions": 3,
                        "dataset_cache_dir": cache_dir,
                    },
                    context,
                )


class ValidationCacheConfigTest(unittest.TestCase):
    def test_partitioned_validation_cache_request_is_split_specific(self):
        base_config = {
            "env_family": "pointmaze",
            "model_name": "unit-tokenizer",
            "max_length": 128,
            "dataset_workers": 1,
            "action_dim": 2,
        }

        val_config = _validation_cache_config(base_config, partition_count=4)
        request = build_dataset_request(val_config, object(), "umaze", "val")

        self.assertEqual(request.dataset_partition_count, 4)
        self.assertEqual(request.dataset_partition_index, 0)
        self.assertNotIn("dataset_partition_count", base_config)
        self.assertNotIn("dataset_partition_index", base_config)

    def test_partition_marker_makes_dataset_select_val_indices_only(self):
        selection = {"train_indices": [0, 1, 2], "val_indices": [3, 4]}
        non_partitioned_val = types.SimpleNamespace(
            episode_segments=None,
            dataset_partition_count=1,
            split="val",
        )
        partitioned_val = types.SimpleNamespace(
            episode_segments=None,
            dataset_partition_count=4,
            split="val",
        )

        self.assertEqual(
            PointMazeDataset._selected_indices_for_config(non_partitioned_val, selection),
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            PointMazeDataset._selected_indices_for_config(partitioned_val, selection),
            [3, 4],
        )

    def test_single_partition_validation_cache_config_preserves_default_request(self):
        base_config = {
            "env_family": "pointmaze",
            "model_name": "unit-tokenizer",
            "max_length": 128,
            "dataset_workers": 1,
            "action_dim": 2,
        }

        val_config = _validation_cache_config(base_config, partition_count=1)
        request = build_dataset_request(val_config, object(), "umaze", "val")

        self.assertEqual(request.dataset_partition_count, 1)
        self.assertIsNone(request.dataset_partition_index)

    def test_pointmaze_data_config_is_passed_to_dataset_request(self):
        config = {
            "env_family": "pointmaze",
            "model_name": "unit-tokenizer",
            "max_length": 128,
            "dataset_workers": 1,
            "action_dim": 2,
            "pointmaze_data_config": {
                "truncate": True,
                "truncate_holding": 3,
            },
        }

        request = build_dataset_request(config, object(), "open", "train")

        self.assertEqual(request.family_data_config, config["pointmaze_data_config"])


if __name__ == "__main__":
    unittest.main()
