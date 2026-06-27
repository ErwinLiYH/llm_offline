import os
import tempfile
import unittest

import yaml

from utils.config_loader import deep_merge_configs, load_merged_config


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


if __name__ == "__main__":
    unittest.main()
