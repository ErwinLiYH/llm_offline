import os
import re
import unittest
from dataclasses import replace

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  registers PointMaze envs
import numpy as np

from data.pointmaze.dataset import (
    PointMazeBuildConfig,
    PointMazeDataset,
    _partition_episode_indices,
)
from data.pointmaze import formatting
from data.pointmaze.formatting import _obs_xy_to_row_col
from data.pointmaze.variants import POINTMAZE_VARIANTS
from utils.prompt_loader import load_template_map, render_template


class PointMazeFormattingTest(unittest.TestCase):
    def test_remote_prompt_maps_match_registered_env_maps(self):
        for variant, meta in POINTMAZE_VARIANTS.items():
            if meta.get("varient_type") != "remote":
                continue

            with self.subTest(variant=variant):
                env = gym.make(meta["env_id"])
                try:
                    self.assertEqual(
                        meta["prompt_vars"]["maze_map"],
                        env.unwrapped.maze.maze_map,
                    )
                finally:
                    env.close()

    def test_remote_cell_conversion_matches_env_cell_centers(self):
        for variant, meta in POINTMAZE_VARIANTS.items():
            if meta.get("varient_type") != "remote":
                continue

            with self.subTest(variant=variant):
                env = gym.make(meta["env_id"])
                try:
                    maze = env.unwrapped.maze
                    prompt_map = meta["prompt_vars"]["maze_map"]
                    for row, row_values in enumerate(prompt_map):
                        for col, value in enumerate(row_values):
                            if value == 1:
                                continue

                            xy = maze.cell_rowcol_to_xy(np.array([row, col]))
                            self.assertEqual(
                                _obs_xy_to_row_col(
                                    float(xy[0]),
                                    float(xy[1]),
                                    prompt_map,
                                    maze_size_scaling=maze.maze_size_scaling,
                                ),
                                (row, col),
                            )
                finally:
                    env.close()

    def test_large_log_coordinate_maps_to_video_cell(self):
        maze_map = POINTMAZE_VARIANTS["large"]["prompt_vars"]["maze_map"]

        self.assertEqual(
            _obs_xy_to_row_col(
                -1.5023,
                0.6034,
                maze_map,
            ),
            (3, 4),
        )

    def test_wall_cell_observation_snaps_to_nearest_free_cell(self):
        maze_map = POINTMAZE_VARIANTS["large"]["prompt_vars"]["maze_map"]

        row, col = _obs_xy_to_row_col(
            -1.5,
            -0.49,
            maze_map,
        )

        self.assertEqual((row, col), (5, 4))
        self.assertNotEqual(maze_map[row][col], 1)

    def test_format_obs_splits_location_and_wall_sensing(self):
        prompt_vars = POINTMAZE_VARIANTS["large"]["prompt_vars"]
        obs = {
            "observation": np.array([-1.5023, 0.6034, 0.1, -0.2], dtype=np.float32),
            "desired_goal": np.array([-1.5, -0.49], dtype=np.float32),
        }

        payload = formatting.format_obs(obs, prompt_vars)

        self.assertIn("obs_text", payload)
        self.assertIn("location_sensing_en", payload)
        self.assertIn("location_sensing_zh", payload)
        self.assertIn("wall_sensing_en", payload)
        self.assertIn("wall_sensing_zh", payload)
        self.assertNotIn("map_sensing_en", payload)
        self.assertNotIn("map_sensing_zh", payload)

        self.assertIn("Current cell: row 4, column 5.", payload["location_sensing_en"])
        self.assertIn("Goal cell: row 6, column 5.", payload["location_sensing_en"])
        self.assertIn("Rows and columns", payload["location_sensing_en"])
        self.assertNotIn("Neighboring cells", payload["location_sensing_en"])

        self.assertIn("Neighboring cells:", payload["wall_sensing_en"])
        for label in ("up=", "down=", "left=", "right="):
            self.assertIn(label, payload["wall_sensing_en"])
        self.assertNotIn("Current cell", payload["wall_sensing_en"])
        self.assertNotIn("Goal cell", payload["wall_sensing_en"])

        self.assertIn("当前位置格子：第 4 行，第 5 列。", payload["location_sensing_zh"])
        self.assertIn("目标格子：第 6 行，第 5 列。", payload["location_sensing_zh"])
        self.assertNotIn("相邻格子", payload["location_sensing_zh"])
        self.assertIn("相邻格子：", payload["wall_sensing_zh"])
        for label in ("上=", "下=", "左=", "右="):
            self.assertIn(label, payload["wall_sensing_zh"])

    def test_pointmaze_templates_render_with_split_sensing_fields(self):
        prompt_vars = POINTMAZE_VARIANTS["large"]["prompt_vars"]
        obs = {
            "observation": np.array([-1.5023, 0.6034, 0.1, -0.2], dtype=np.float32),
            "desired_goal": np.array([-1.5, -0.49], dtype=np.float32),
        }
        obs_payload = formatting.format_obs(obs, prompt_vars)
        history_payload = formatting.format_history([], prompt_vars)

        for name, template in load_template_map("pointmaze").items():
            with self.subTest(template=name):
                self.assertNotIn("map_sensing", template)
                rendered = render_template(
                    template,
                    prompt_vars,
                    **obs_payload,
                    **history_payload,
                )
                self.assertNotIn("map_sensing", rendered)

    def test_dataset_cache_path_uses_compact_signature_hash(self):
        config = PointMazeBuildConfig(
            variant="large",
            split="train",
            tokenizer_name_or_path="Qwen/Qwen3.5-0.8B",
            max_length=512,
            num_workers=1,
            cache_dir="/tmp/pointmaze-cache",
            max_data_num=None,
            dataset_partition_count=1,
            dataset_partition_index=None,
            prompt_template_count=1,
            prompt_templete_index=["0"],
            train_data_ratio=0.9,
            episode_keep_num=None,
            balance_variant_episode_count=False,
            balanced_train_episode_count=None,
            sampling_seed=0,
            family_data_config=None,
            local_dataset_root=None,
            history_num=0,
            history_stride=1,
            wall_sensing_version="v3",
            map_sensing_boundary_risk_threshold=0.10,
            action_token_mode="text",
            action_num_bins=50,
            action_bin_min=-1.0,
            action_bin_max=1.0,
            new_token=False,
            action_dim=2,
            action_token_schema_hash="text",
            progress_interval_seconds=5.0,
        )

        cache_path = PointMazeDataset._cache_path(config)
        cache_name = os.path.basename(cache_path)
        payload = PointMazeDataset._cache_signature_payload(config)

        self.assertRegex(cache_name, re.compile(r"^[0-9a-f]{32}\.pkl$"))
        self.assertEqual(payload["variant"], "large")
        self.assertEqual(payload["prompt_names"], ["0"])
        self.assertNotIn("max_data_num", payload)
        self.assertNotIn("dataset_partition_count", payload)
        self.assertNotIn("source_hashes", payload)

        partition0 = replace(
            config,
            dataset_partition_count=2,
            dataset_partition_index=0,
        )
        partition1 = replace(
            config,
            dataset_partition_count=2,
            dataset_partition_index=1,
        )
        val_partition0 = replace(partition0, split="val")
        val_partition1 = replace(partition1, split="val")

        partition_payload = PointMazeDataset._cache_signature_payload(partition0)
        self.assertEqual(partition_payload["split"], "train")
        self.assertEqual(partition_payload["dataset_partition_count"], 2)
        self.assertEqual(partition_payload["dataset_partition_index"], 0)
        val_partition_payload = PointMazeDataset._cache_signature_payload(val_partition0)
        self.assertEqual(val_partition_payload["split"], "val")
        self.assertNotIn("dataset_partition_count", val_partition_payload)
        self.assertNotIn("dataset_partition_index", val_partition_payload)
        self.assertNotEqual(PointMazeDataset._cache_path(partition0), cache_path)
        self.assertNotEqual(PointMazeDataset._cache_path(partition0), PointMazeDataset._cache_path(partition1))
        self.assertNotEqual(PointMazeDataset._cache_path(partition0), PointMazeDataset._cache_path(val_partition0))
        self.assertNotEqual(PointMazeDataset._cache_path(val_partition0), cache_path)
        self.assertEqual(PointMazeDataset._cache_path(val_partition0), PointMazeDataset._cache_path(val_partition1))

    def test_episode_partitioning_is_deterministic_and_non_overlapping(self):
        indices = list(range(17))
        partitions = [
            _partition_episode_indices(
                indices,
                variant="large",
                split="train",
                sampling_seed=123,
                partition_count=4,
                partition_index=idx,
            )
            for idx in range(4)
        ]
        flattened = sorted(item for partition in partitions for item in partition)

        self.assertEqual(flattened, indices)
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(
            partitions[0],
            _partition_episode_indices(
                indices,
                variant="large",
                split="train",
                sampling_seed=123,
                partition_count=4,
                partition_index=0,
            ),
        )

    def test_pointmaze_action_dim_and_continuous_collate(self):
        self.assertEqual(PointMazeDataset.get_action_dim(["large", "medium"]), 2)

        dataset = PointMazeDataset(
            "large",
            "train",
            [
                {
                    "input_ids": [1, 2],
                    "attention_mask": [1, 1],
                    "labels": [-100, -100],
                    "action_bin_labels": [-1, -1],
                    "action_values": [0.25, -0.5],
                },
                {
                    "input_ids": [3],
                    "attention_mask": [1],
                    "labels": [-100],
                    "action_bin_labels": [-1],
                    "action_values": [1.0, 0.0],
                },
            ],
        )

        batch = PointMazeDataset.collate_fn([dataset[0], dataset[1]])

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 2))
        self.assertEqual(tuple(batch["action_values"].shape), (2, 2))
        np.testing.assert_allclose(batch["action_values"][0].numpy(), [0.25, -0.5])


if __name__ == "__main__":
    unittest.main()
