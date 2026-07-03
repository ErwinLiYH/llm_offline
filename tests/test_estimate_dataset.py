import contextlib
import io
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

unsloth_stub = types.ModuleType("unsloth")
unsloth_stub.FastLanguageModel = object()
sys.modules.setdefault("unsloth", unsloth_stub)

from data.antmaze.dataset import AntMazeDataset
from data.pointmaze.dataset import PointMazeDataset
from estimate_dataset import (
    SampleFootprint,
    VariantData,
    _balanced_train_episode_count as estimate_balanced_train_episode_count,
    build_dataset_request as build_estimate_dataset_request,
    build_estimate,
    estimate_bytes_for_steps,
    estimate_epoch_batches,
    load_variant_data,
)
from train import (
    _resolve_balanced_train_episode_count,
    build_dataset_request as build_train_dataset_request,
)
from utils.distributed import DistributedContext
from utils.episode_keep import (
    RESOLVED_EPISODE_KEEP_PER_VARIANT_KEY,
    resolve_episode_keep_per_variant,
)


def _variant_data(*, train_steps: int, val_steps: int, prompt_count: int = 1) -> VariantData:
    return VariantData(
        variant="unit",
        episodes=[],
        step_counts=[train_steps, val_steps],
        prompt_count=prompt_count,
        selection={
            "total_episodes": 2,
            "total_steps": train_steps + val_steps,
            "sampled_episode_count": 2,
            "train_indices": [0],
            "val_indices": [1],
            "train_episode_count": 1,
            "val_episode_count": 1,
            "train_steps": train_steps,
            "val_steps": val_steps,
        },
    )


class EstimateDatasetTest(unittest.TestCase):
    def test_episode_keep_without_per_variant_config_uses_global_keep(self):
        resolved = resolve_episode_keep_per_variant(
            {"episode_keep_num": 8},
            ["open", "medium"],
            available_variants=["open", "medium"],
        )

        self.assertEqual(resolved, {"open": 8, "medium": 8})

    def test_episode_keep_overrides_selected_variants_and_falls_back_to_global(self):
        resolved = resolve_episode_keep_per_variant(
            {
                "episode_keep_num": 8,
                "episode_keep_per_varient": {
                    "open": 5,
                    "medium": None,
                },
            },
            ["open", "umaze", "medium"],
            available_variants=["open", "umaze", "medium"],
        )

        self.assertEqual(resolved, {"open": 5, "umaze": 8, "medium": None})

    def test_episode_keep_rejects_unknown_variant_keys(self):
        with self.assertRaisesRegex(ValueError, "unknown variants: missing"):
            resolve_episode_keep_per_variant(
                {"episode_keep_per_varient": {"missing": 2}},
                ["open"],
                available_variants=["open", "medium"],
            )

    def test_episode_keep_rejects_unselected_variant_keys(self):
        with self.assertRaisesRegex(ValueError, "not selected: medium"):
            resolve_episode_keep_per_variant(
                {"episode_keep_per_varient": {"medium": 2}},
                ["open"],
                available_variants=["open", "medium"],
            )

    def test_episode_keep_rejects_invalid_override_values(self):
        for bad_value in (0, -1, 1.5, "2", True):
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(ValueError, "episode_keep_per_varient.open"):
                    resolve_episode_keep_per_variant(
                        {"episode_keep_per_varient": {"open": bad_value}},
                        ["open"],
                        available_variants=["open"],
                    )

    def test_episode_keep_rejects_non_dict_override_config(self):
        with self.assertRaisesRegex(ValueError, "must be a dict"):
            resolve_episode_keep_per_variant(
                {"episode_keep_per_varient": ["open"]},
                ["open"],
                available_variants=["open"],
            )

    def test_train_and_estimate_requests_receive_effective_episode_keep(self):
        config = {
            "model_name": "unit-tokenizer",
            "max_length": 128,
            "episode_keep_num": 8,
            RESOLVED_EPISODE_KEEP_PER_VARIANT_KEY: {
                "open": 5,
                "umaze": 8,
                "medium": None,
            },
        }

        train_request = build_train_dataset_request(config, object(), "open", "train")
        estimate_request = build_estimate_dataset_request(config, object(), "medium", "train")

        self.assertEqual(train_request.episode_keep_num, 5)
        self.assertIsNone(estimate_request.episode_keep_num)

    def test_estimator_selection_uses_per_variant_episode_keep_values(self):
        def fake_loader(cls, variant, family_data_config=None, local_dataset_root=None):
            del cls, variant, family_data_config, local_dataset_root
            episodes = [
                SimpleNamespace(actions=np.zeros((step_count, 2), dtype=np.float32))
                for step_count in (1, 2, 3, 4, 5)
            ]
            return {}, episodes, [len(episode.actions) for episode in episodes]

        config = {
            "env_family": "pointmaze",
            "prompt_templete_index": ["unit"],
            "train_data_ratio": 0.5,
            "sampling_seed": 0,
            "episode_keep_num": 4,
            "episode_keep_per_varient": {
                "open": 2,
                "medium": None,
            },
        }

        with mock.patch.object(
            PointMazeDataset,
            "_load_variant_episodes",
            new=classmethod(fake_loader),
        ):
            data = load_variant_data(config, ["open", "umaze", "medium"])

        sampled_counts = {
            item.variant: item.selection["sampled_episode_count"]
            for item in data
        }
        self.assertEqual(sampled_counts, {"open": 2, "umaze": 4, "medium": 5})

    def test_train_balance_is_disabled_when_per_variant_episode_keep_is_configured(self):
        class FakeDataset:
            @classmethod
            def collect_variant_episode_stats(cls, *args, **kwargs):
                raise AssertionError("balance stats should not be collected")

        context = DistributedContext(backend="single")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = _resolve_balanced_train_episode_count(
                {
                    "balance_variant_episode_count": True,
                    "episode_keep_num": 8,
                    "episode_keep_per_varient": {"open": 5},
                },
                FakeDataset,
                ["open", "umaze"],
                context,
            )

        self.assertIsNone(result)
        self.assertIn("episode_keep_per_varient is configured", stdout.getvalue())

    def test_estimate_balance_is_disabled_when_per_variant_episode_keep_is_configured(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = estimate_balanced_train_episode_count(
                {
                    "balance_variant_episode_count": True,
                    "episode_keep_num": 8,
                    "episode_keep_per_varient": {"open": 5},
                },
                ["open", "umaze"],
                {
                    "open": ({}, [object()] * 10, [1] * 10),
                    "umaze": ({}, [object()] * 10, [1] * 10),
                },
            )

        self.assertIsNone(result)
        self.assertIn("episode_keep_per_varient is configured", stderr.getvalue())

    def test_size_estimate_uses_step_ratio(self):
        self.assertEqual(estimate_bytes_for_steps(1000, 10, 25), 2500)

        data = _variant_data(train_steps=100, val_steps=50, prompt_count=3)
        estimate = build_estimate(
            {
                "env_family": "pointmaze",
                "model_name": "dummy",
                "action_token_mode": "text",
                "prompt_templete_index": ["a", "b", "c"],
                "batch_size": 8,
            },
            [data],
            [
                SampleFootprint(
                    variant="unit",
                    sampled_episodes=1,
                    sampled_steps=10,
                    sampled_samples=30,
                    sampled_pickle_bytes=1000,
                    sampled_memory_bytes=2000,
                    sampled_tokens=300,
                )
            ],
            partition_count=1,
            world_size=1,
            sample_seed=0,
        )

        self.assertEqual(estimate["size"]["train_bytes"], 10_000)
        self.assertEqual(estimate["size"]["val_bytes"], 5_000)
        self.assertEqual(estimate["variants"][0]["train_samples"], 300)

    def test_world_size_only_changes_batch_math(self):
        data = _variant_data(train_steps=17, val_steps=0, prompt_count=2)

        estimate = estimate_epoch_batches(
            [data],
            {
                "prompt_templete_index": ["p0", "p1"],
                "batch_size": 4,
            },
            partition_count=1,
            world_size=4,
        )

        self.assertFalse(estimate["partitioned"])
        self.assertEqual(estimate["selected_train_samples"], 34)
        self.assertEqual(estimate["per_rank_samples_per_epoch"], 9)
        self.assertEqual(estimate["sampler_samples_per_epoch"], 36)
        self.assertEqual(estimate["train_batches_per_epoch"], 3)

    def test_antmaze_data_config_is_passed_to_loader(self):
        episodes = [
            SimpleNamespace(actions=np.zeros((4, 8), dtype=np.float32)),
            SimpleNamespace(actions=np.zeros((2, 8), dtype=np.float32)),
        ]
        captured_configs = []

        def fake_loader(cls, variant, family_data_config=None, local_dataset_root=None):
            del local_dataset_root
            captured_configs.append(family_data_config)
            return {}, episodes, [len(episode.actions) for episode in episodes]

        config = {
            "env_family": "antmaze",
            "prompt_templete_index": ["parallel_full_sensing"],
            "train_data_ratio": 0.5,
            "sampling_seed": 123,
            "antmaze_data_config": {
                "filter_success": True,
                "truncate": True,
                "truncate_holding": 2,
            },
        }

        with mock.patch.object(
            AntMazeDataset,
            "_load_variant_episodes",
            new=classmethod(fake_loader),
        ):
            data = load_variant_data(config, ["umaze"])

        self.assertEqual(captured_configs, [config["antmaze_data_config"]])
        self.assertEqual(data[0].total_steps, 6)
        self.assertEqual(data[0].train_steps + data[0].val_steps, 6)

    def test_pointmaze_data_config_is_passed_to_loader(self):
        episodes = [
            SimpleNamespace(actions=np.zeros((4, 2), dtype=np.float32)),
            SimpleNamespace(actions=np.zeros((2, 2), dtype=np.float32)),
        ]
        captured_configs = []

        def fake_loader(cls, variant, family_data_config=None, local_dataset_root=None):
            del local_dataset_root
            captured_configs.append(family_data_config)
            return {}, episodes, [len(episode.actions) for episode in episodes]

        config = {
            "env_family": "pointmaze",
            "prompt_templete_index": ["parallel_full_sensing"],
            "train_data_ratio": 0.5,
            "sampling_seed": 123,
            "pointmaze_data_config": {
                "truncate": True,
                "truncate_holding": 2,
            },
        }

        with mock.patch.object(
            PointMazeDataset,
            "_load_variant_episodes",
            new=classmethod(fake_loader),
        ):
            data = load_variant_data(config, ["open"])

        self.assertEqual(captured_configs, [config["pointmaze_data_config"]])
        self.assertEqual(data[0].total_steps, 6)
        self.assertEqual(data[0].train_steps + data[0].val_steps, 6)

    def test_local_dataset_root_is_passed_to_loader(self):
        episodes = [
            SimpleNamespace(actions=np.zeros((4, 2), dtype=np.float32)),
            SimpleNamespace(actions=np.zeros((2, 2), dtype=np.float32)),
        ]
        captured_roots = []

        def fake_loader(cls, variant, family_data_config=None, local_dataset_root=None):
            del family_data_config
            captured_roots.append(local_dataset_root)
            return {}, episodes, [len(episode.actions) for episode in episodes]

        config = {
            "env_family": "pointmaze",
            "prompt_templete_index": ["parallel_full_sensing"],
            "train_data_ratio": 0.5,
            "sampling_seed": 123,
            "local_dataset_root": "/scratch/local_datasets_v2",
        }

        with mock.patch.object(
            PointMazeDataset,
            "_load_variant_episodes",
            new=classmethod(fake_loader),
        ):
            data = load_variant_data(config, ["open"])

        self.assertEqual(captured_roots, ["/scratch/local_datasets_v2"])
        self.assertEqual(data[0].total_steps, 6)


if __name__ == "__main__":
    unittest.main()
