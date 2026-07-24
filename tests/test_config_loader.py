import os
import tempfile
import unittest

import yaml

from utils.config_loader import (
    deep_merge_configs,
    load_merged_config,
    resolve_dataset_cache_dir,
)


class ConfigLoaderTests(unittest.TestCase):
    def test_deep_merge_keeps_base_nested_keys_and_replaces_lists(self):
        base = {
            "rollout_worker_num": 1,
            "variants": ["a"],
            "dataloader_config": {
                "num_workers": 4,
                "pin_memory": True,
            },
        }
        override = {
            "variants": ["b", "c"],
            "dataloader_config": {
                "num_workers": 8,
            },
        }

        merged = deep_merge_configs(base, override)

        self.assertEqual(merged["rollout_worker_num"], 1)
        self.assertEqual(merged["variants"], ["b", "c"])
        self.assertEqual(
            merged["dataloader_config"],
            {"num_workers": 8, "pin_memory": True},
        )
        self.assertEqual(base["variants"], ["a"])
        self.assertEqual(base["dataloader_config"]["num_workers"], 4)

    def test_later_config_can_override_with_null(self):
        merged = deep_merge_configs({"experiment_id": "old"}, {"experiment_id": None})
        self.assertIsNone(merged["experiment_id"])

    def test_load_merged_config_uses_later_files_as_higher_priority(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "base.yaml")
            override_path = os.path.join(tmpdir, "override.yaml")
            with open(base_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {
                        "record_video": False,
                        "dataloader_config": {
                            "num_workers": 2,
                            "pin_memory": True,
                        },
                    },
                    f,
                )
            with open(override_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {
                        "record_video": True,
                        "dataloader_config": {
                            "num_workers": 6,
                        },
                    },
                    f,
                )

            merged = load_merged_config([base_path, override_path])

        self.assertTrue(merged["record_video"])
        self.assertEqual(
            merged["dataloader_config"],
            {"num_workers": 6, "pin_memory": True},
        )

    def test_later_config_can_delete_base_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "base.yaml")
            override_path = os.path.join(tmpdir, "override.yaml")
            with open(base_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {
                        "action_token_mode": "parallel_gaussian",
                        "gaussian_log_std_init": -1.0,
                        "dataloader_config": {
                            "num_workers": 0,
                            "pin_memory": False,
                        },
                    },
                    f,
                )
            with open(override_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {
                        "config_delete_keys": [
                            "gaussian_log_std_init",
                            "dataloader_config.pin_memory",
                        ],
                        "action_token_mode": "simple_mtp_bin",
                    },
                    f,
                )

            merged = load_merged_config([base_path, override_path])

        self.assertEqual(merged["action_token_mode"], "simple_mtp_bin")
        self.assertNotIn("gaussian_log_std_init", merged)
        self.assertEqual(merged["dataloader_config"], {"num_workers": 0})

    def test_complete_v2_cache_config_overrides_legacy_cache_dir(self):
        config = {
            "dataset_cache_dir": "/legacy/cache",
            "dataset_cache_v2_root": "/scratch/cache-root",
            "dataset_cache_v2_dir": "pointmaze/scale16",
        }

        self.assertEqual(
            resolve_dataset_cache_dir(config),
            os.path.join("/scratch/cache-root", "pointmaze/scale16"),
        )

    def test_partial_v2_cache_config_falls_back_to_legacy_cache_dir(self):
        self.assertEqual(
            resolve_dataset_cache_dir(
                {
                    "dataset_cache_dir": "/legacy/cache",
                    "dataset_cache_v2_root": "/scratch/cache-root",
                }
            ),
            "/legacy/cache",
        )
        self.assertIsNone(
            resolve_dataset_cache_dir({"dataset_cache_v2_dir": "pointmaze/scale16"})
        )

    def test_v2_cache_dir_must_be_relative(self):
        with self.assertRaisesRegex(ValueError, "must be a relative path"):
            resolve_dataset_cache_dir(
                {
                    "dataset_cache_v2_root": "/scratch/cache-root",
                    "dataset_cache_v2_dir": "/other/cache",
                }
            )

    def test_complete_v2_cache_config_requires_string_paths(self):
        with self.assertRaisesRegex(ValueError, "dataset_cache_v2_root"):
            resolve_dataset_cache_dir(
                {
                    "dataset_cache_v2_root": 1,
                    "dataset_cache_v2_dir": "pointmaze/scale16",
                }
            )
        with self.assertRaisesRegex(ValueError, "dataset_cache_v2_dir"):
            resolve_dataset_cache_dir(
                {
                    "dataset_cache_v2_root": "/scratch/cache-root",
                    "dataset_cache_v2_dir": 1,
                }
            )

    def test_v2_cache_config_is_resolved_after_layered_merge(self):
        merged = deep_merge_configs(
            {"dataset_cache_v2_root": "/scratch/cache-root"},
            {
                "dataset_cache_dir": "/legacy/cache",
                "dataset_cache_v2_dir": "antmaze/scale16",
            },
        )

        self.assertEqual(
            resolve_dataset_cache_dir(merged),
            os.path.join("/scratch/cache-root", "antmaze/scale16"),
        )


if __name__ == "__main__":
    unittest.main()
