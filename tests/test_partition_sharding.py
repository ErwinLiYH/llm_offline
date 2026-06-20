import sys
import tempfile
import types
import unittest

import torch

unsloth_stub = types.ModuleType("unsloth")
unsloth_stub.FastLanguageModel = object()
sys.modules.setdefault("unsloth", unsloth_stub)

from data.pointmaze.dataset import split_episode_segments_for_partitions
from train import _compute_partition_round_stats, resolve_dataset_load_partitions
from utils.distributed import DistributedContext
from utils.distributed_sampler import LocalShardPaddingSampler


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


if __name__ == "__main__":
    unittest.main()
