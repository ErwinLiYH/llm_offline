import tempfile
import unittest
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  registers AntMaze envs
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from data.antmaze import formatting
from data.antmaze.dataset import AntMazeDataset
from data.antmaze.variants import ANTMAZE_VARIANTS
from data.base_dataset import DatasetBuildRequest
from data.registry import get_action_dim, resolve_variant_env_spec
from utils.maze_sensing import _neighbor_status
from utils.prompt_loader import load_template_map, render_template
from utils.variant_selection import get_available_variants


def _fake_episodes():
    episodes = []
    for episode_idx in range(2):
        observations = {
            "observation": np.zeros((3, 27), dtype=np.float32),
            "achieved_goal": np.asarray(
                [[episode_idx, 0.0], [episode_idx, 0.5], [episode_idx, 1.0]],
                dtype=np.float32,
            ),
            "desired_goal": np.asarray(
                [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]],
                dtype=np.float32,
            ),
        }
        actions = np.full((2, 8), 0.1 * (episode_idx + 1), dtype=np.float32)
        episodes.append(SimpleNamespace(observations=observations, actions=actions))
    return episodes


def _make_test_tokenizer(path: str):
    backend = Tokenizer(
        WordLevel(
            vocab={
                "<unk>": 0,
                "<pad>": 1,
                "<eos>": 2,
                "user": 3,
                "assistant": 4,
            },
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }} <eos> "
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    tokenizer.save_pretrained(path)
    return tokenizer


class AntMazeSupportTest(unittest.TestCase):
    def test_corner_risk_distinguishes_new_corners_from_continuous_walls(self):
        row = 3
        col = 3
        cases = [
            ("up-left", -1, 0, 0, -1, -1, -1, -0.45, 0.0),
            ("up-right", -1, 0, 0, 1, -1, 1, 0.45, 0.0),
            ("down-left", 1, 0, 0, -1, 1, -1, -0.45, 0.0),
            ("down-right", 1, 0, 0, 1, 1, 1, 0.45, 0.0),
            ("left-top", 0, -1, -1, 0, -1, -1, 0.0, 0.45),
            ("left-bottom", 0, -1, 1, 0, 1, -1, 0.0, -0.45),
            ("right-top", 0, 1, -1, 0, -1, 1, 0.0, 0.45),
            ("right-bottom", 0, 1, 1, 0, 1, 1, 0.0, -0.45),
        ]

        for (
            name,
            d_row,
            d_col,
            side_d_row,
            side_d_col,
            diagonal_d_row,
            diagonal_d_col,
            x,
            y,
        ) in cases:
            with self.subTest(case=name):
                maze_map = [[0 for _ in range(7)] for _ in range(7)]
                maze_map[row + diagonal_d_row][col + diagonal_d_col] = 1

                self.assertEqual(
                    _neighbor_status(
                        maze_map,
                        row,
                        col,
                        d_row,
                        d_col,
                        x=x,
                        y=y,
                    ),
                    "wall",
                )

                maze_map[row + side_d_row][col + side_d_col] = 1
                self.assertEqual(
                    _neighbor_status(
                        maze_map,
                        row,
                        col,
                        d_row,
                        d_col,
                        x=x,
                        y=y,
                    ),
                    "free",
                )

                maze_map[row + side_d_row][col + side_d_col] = 0
                maze_map[row + diagonal_d_row][col + diagonal_d_col] = 0
                self.assertEqual(
                    _neighbor_status(
                        maze_map,
                        row,
                        col,
                        d_row,
                        d_col,
                        x=x,
                        y=y,
                    ),
                    "free",
                )

                maze_map[row + diagonal_d_row][col + diagonal_d_col] = 1
                self.assertEqual(
                    _neighbor_status(
                        maze_map,
                        row,
                        col,
                        d_row,
                        d_col,
                        x=0.0,
                        y=0.0,
                    ),
                    "free",
                )

    def test_direct_neighbor_wall_still_blocks_all_directions(self):
        row = 3
        col = 3
        for name, d_row, d_col in (
            ("up", -1, 0),
            ("down", 1, 0),
            ("left", 0, -1),
            ("right", 0, 1),
        ):
            with self.subTest(direction=name):
                maze_map = [[0 for _ in range(7)] for _ in range(7)]
                maze_map[row + d_row][col + d_col] = 1
                self.assertEqual(
                    _neighbor_status(
                        maze_map,
                        row,
                        col,
                        d_row,
                        d_col,
                        x=0.0,
                        y=0.0,
                    ),
                    "wall",
                )

    def test_registry_contains_official_d4rl_variants(self):
        self.assertEqual(
            get_available_variants("antmaze"),
            [
                "umaze",
                "umaze-diverse",
                "medium-play",
                "medium-diverse",
                "large-play",
                "large-diverse",
            ],
        )
        self.assertEqual(get_action_dim("antmaze", ["umaze", "large-diverse"]), 8)

    def test_registered_eval_env_matches_d4rl_observation_contract(self):
        _meta, env_id, env_kwargs = resolve_variant_env_spec("antmaze", "umaze")
        env = gym.make(env_id, **env_kwargs)
        try:
            obs, _ = env.reset(seed=0)
            self.assertEqual(obs["observation"].shape, (27,))
            self.assertEqual(obs["achieved_goal"].shape, (2,))
            self.assertEqual(obs["desired_goal"].shape, (2,))
            self.assertEqual(env.action_space.shape, (8,))
        finally:
            env.close()

    def test_training_map_cell_conversion_matches_registered_env_geometry(self):
        for variant, meta in ANTMAZE_VARIANTS.items():
            with self.subTest(variant=variant):
                env = gym.make(meta["env_id"])
                try:
                    maze = env.unwrapped.maze
                    prompt_vars = meta["prompt_vars"]
                    self.assertEqual(
                        [
                            [cell == 1 for cell in row]
                            for row in prompt_vars["maze_map"]
                        ],
                        [
                            [cell == 1 for cell in row]
                            for row in maze.maze_map
                        ],
                    )
                    for row, row_values in enumerate(maze.maze_map):
                        for col, value in enumerate(row_values):
                            if value == 1:
                                continue
                            xy = maze.cell_rowcol_to_xy(np.asarray([row, col]))
                            self.assertEqual(
                                formatting.obs_xy_to_row_col(
                                    float(xy[0]),
                                    float(xy[1]),
                                    prompt_vars["maze_map"],
                                    maze_size_scaling=prompt_vars["maze_size_scaling"],
                                ),
                                (row, col),
                            )
                finally:
                    env.close()

    def test_eval_sensing_uses_instantiated_official_eval_map(self):
        for variant in ANTMAZE_VARIANTS:
            with self.subTest(variant=variant):
                meta, env_id, env_kwargs = resolve_variant_env_spec(
                    "antmaze",
                    variant,
                )
                env = gym.make(env_id, **env_kwargs)
                try:
                    prompt_vars = formatting.prepare_eval_prompt_vars(
                        meta["prompt_vars"],
                        env,
                    )
                    maze = env.unwrapped.maze
                    self.assertEqual(prompt_vars["maze_map"], maze.maze_map)
                    self.assertEqual(
                        prompt_vars["maze_size_scaling"],
                        maze.maze_size_scaling,
                    )
                    for row, row_values in enumerate(maze.maze_map):
                        for col, value in enumerate(row_values):
                            if value == 1:
                                continue
                            xy = maze.cell_rowcol_to_xy(np.asarray([row, col]))
                            self.assertEqual(
                                formatting.obs_xy_to_row_col(
                                    float(xy[0]),
                                    float(xy[1]),
                                    prompt_vars["maze_map"],
                                    maze_size_scaling=prompt_vars["maze_size_scaling"],
                                ),
                                (row, col),
                            )
                finally:
                    env.close()

        meta, env_id, env_kwargs = resolve_variant_env_spec("antmaze", "umaze")
        env = gym.make(env_id, **env_kwargs)
        try:
            prompt_vars = formatting.prepare_eval_prompt_vars(meta["prompt_vars"], env)
            maze = env.unwrapped.maze
            current_xy = maze.cell_rowcol_to_xy(np.asarray([1, 3]))
            goal_xy = maze.cell_rowcol_to_xy(np.asarray([3, 3]))
            obs = {
                "observation": np.zeros(27, dtype=np.float32),
                "achieved_goal": current_xy,
                "desired_goal": goal_xy,
            }

            payload = formatting.format_obs(obs, prompt_vars)

            self.assertEqual(prompt_vars["maze_map"], maze.maze_map)
            self.assertIn(
                "Current cell: row 2, column 4. Goal cell: row 4, column 4.",
                payload["location_sensing_en"],
            )
            self.assertEqual(
                payload["wall_sensing_en"],
                "Neighboring cells: up=wall, down=wall, left=free, right=wall.",
            )
        finally:
            env.close()

    def test_action_format_parse_and_validation(self):
        action = np.asarray(
            [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8],
            dtype=np.float32,
        )
        text = formatting.format_action(action)
        parsed, success = formatting.parse_action(text)

        self.assertEqual(text, "-10,20,-30,40,-50,60,-70,80")
        self.assertTrue(success)
        np.testing.assert_allclose(parsed, action)
        self.assertTrue(formatting.validate_action(parsed))
        self.assertFalse(formatting.validate_action(np.zeros(2, dtype=np.float32)))
        self.assertFalse(formatting.parse_action("1,2,3,4,5,6,7,8,9")[1])

    def test_all_prompts_render(self):
        prompt_vars = ANTMAZE_VARIANTS["medium-diverse"]["prompt_vars"]
        obs = {
            "observation": np.zeros(27, dtype=np.float32),
            "achieved_goal": np.asarray([1.0, 2.0], dtype=np.float32),
            "desired_goal": np.asarray([3.0, 4.0], dtype=np.float32),
        }
        obs_payload = formatting.format_obs(obs, prompt_vars)
        history_payload = formatting.format_history([], prompt_vars)

        for key in (
            "location_sensing_en",
            "location_sensing_zh",
            "wall_sensing_en",
            "wall_sensing_zh",
        ):
            self.assertIn(key, obs_payload)

        for name, template in load_template_map("antmaze").items():
            with self.subTest(template=name):
                rendered = render_template(
                    template,
                    prompt_vars,
                    **obs_payload,
                    **history_payload,
                )
                self.assertIn("Ant quadruped", rendered)
                self.assertIn("Torso xy", rendered)

                expects_location = (
                    name == "0"
                    or "full_sensing" in name
                    or "loca_sensing" in name
                )
                expects_wall = name == "0" or "full_sensing" in name or "wall_sensing" in name
                self.assertEqual("Location sensing:" in rendered, expects_location)
                self.assertEqual("Current cell:" in rendered, expects_location)
                self.assertEqual("Wall sensing:" in rendered, expects_wall)
                self.assertEqual("Neighboring cells:" in rendered, expects_wall)

    def test_bin_and_parallel_prompt_families_have_four_sensing_variants(self):
        names = set(load_template_map("antmaze"))
        self.assertEqual(
            {name for name in names if name.startswith("bin_")},
            {
                "bin_full_sensing",
                "bin_loca_sensing",
                "bin_wall_sensing",
                "bin_no_sensing",
            },
        )
        self.assertEqual(
            {name for name in names if name.startswith("parallel_")},
            {
                "parallel_full_sensing",
                "parallel_loca_sensing",
                "parallel_wall_sensing",
                "parallel_no_sensing",
            },
        )
        self.assertNotIn("bin", names)
        self.assertNotIn("parallel", names)

    def test_history_includes_torso_cell(self):
        prompt_vars = ANTMAZE_VARIANTS["umaze"]["prompt_vars"]
        history = formatting.format_history(
            [
                {
                    "steps_ago": 2,
                    "observation": np.asarray([4.0, 4.0], dtype=np.float32),
                    "action_text": "0,0,0,0,0,0,0,0",
                }
            ],
            prompt_vars,
        )

        self.assertIn("torso_xy=(4.0000, 4.0000)", history["history_block_en"])
        self.assertIn("cell=row 2, column 4", history["history_block_en"])

    def test_synthetic_continuous_tokenization_pipeline(self):
        episodes = _fake_episodes()

        def fake_loader(_cls, variant):
            return ANTMAZE_VARIANTS[variant], episodes, [2, 2]

        with tempfile.TemporaryDirectory() as tokenizer_dir:
            tokenizer = _make_test_tokenizer(tokenizer_dir)
            requests = [
                DatasetBuildRequest(
                    variant="umaze",
                    split=split,
                    tokenizer=tokenizer,
                    tokenizer_name_or_path=tokenizer_dir,
                    max_length=1024,
                    num_workers=1,
                    prompt_templete_index=["parallel_full_sensing"],
                    train_data_ratio=0.5,
                    episode_keep_num=2,
                    sampling_seed=0,
                    action_token_mode="parallel_l1",
                    action_dim=8,
                )
                for split in ("train", "val")
            ]
            with mock.patch.object(
                AntMazeDataset,
                "_load_variant_episodes",
                new=classmethod(fake_loader),
            ):
                train_dataset, val_dataset = AntMazeDataset.build_batch(requests)

        self.assertEqual(len(train_dataset), 2)
        self.assertEqual(len(val_dataset), 2)
        for dataset in (train_dataset, val_dataset):
            sample = dataset[0]
            self.assertEqual(tuple(sample["action_values"].shape), (8,))
            self.assertTrue(bool((sample["labels"] == -100).all()))
            self.assertTrue(bool((sample["action_bin_labels"] == -1).all()))

    def test_synthetic_bin_tokenization_marks_eight_action_positions(self):
        episodes = _fake_episodes()

        def fake_loader(_cls, variant):
            return ANTMAZE_VARIANTS[variant], episodes, [2, 2]

        with tempfile.TemporaryDirectory() as tokenizer_dir:
            tokenizer = _make_test_tokenizer(tokenizer_dir)
            requests = [
                DatasetBuildRequest(
                    variant="umaze",
                    split=split,
                    tokenizer=tokenizer,
                    tokenizer_name_or_path=tokenizer_dir,
                    max_length=1024,
                    num_workers=1,
                    prompt_templete_index=["bin_full_sensing"],
                    train_data_ratio=0.5,
                    episode_keep_num=2,
                    sampling_seed=0,
                    action_token_mode="bin",
                    action_num_bins=5,
                    new_token=True,
                    action_dim=8,
                )
                for split in ("train", "val")
            ]
            with mock.patch.object(
                AntMazeDataset,
                "_load_variant_episodes",
                new=classmethod(fake_loader),
            ):
                train_dataset, val_dataset = AntMazeDataset.build_batch(requests)

        for dataset in (train_dataset, val_dataset):
            sample = dataset[0]
            self.assertEqual(int((sample["action_bin_labels"] >= 0).sum()), 8)

    def test_segment_shard_cache_uses_flat_samples_and_metadata(self):
        episodes = _fake_episodes()
        segment = {
            "segment_key": 7,
            "episode_idx": 0,
            "start_t": 1,
            "end_t": 2,
            "episode_len": 2,
            "step_count": 1,
            "sample_count": 1,
            "variant": "umaze",
        }
        payload = {
            "episode_idx": 0,
            "observations": episodes[0].observations,
            "actions": episodes[0].actions,
        }

        with tempfile.TemporaryDirectory() as tokenizer_dir, tempfile.TemporaryDirectory() as cache_dir:
            tokenizer = _make_test_tokenizer(tokenizer_dir)
            request = DatasetBuildRequest(
                variant="umaze",
                split="train",
                tokenizer=tokenizer,
                tokenizer_name_or_path=tokenizer_dir,
                max_length=1024,
                num_workers=1,
                cache_dir=cache_dir,
                dataset_partition_count=2,
                dataset_partition_index=0,
                episode_segments=[segment],
                episode_payloads=[payload],
                partition_plan_hash="plan-test",
                prompt_templete_index=["parallel_full_sensing"],
                action_token_mode="parallel_l1",
                action_dim=8,
            )
            dataset = AntMazeDataset.build_batch([request])[0]

            cache_paths = list(Path(cache_dir).glob("*.pkl"))
            self.assertEqual(len(cache_paths), 1)
            with open(cache_paths[0], "rb") as f:
                cache = pickle.load(f)

        self.assertEqual(len(dataset), 1)
        self.assertIn("samples", cache)
        self.assertNotIn("episodes", cache)
        self.assertEqual(len(cache["samples"]), 1)
        self.assertEqual(cache["metadata"]["episode_segments"], [segment])
        self.assertEqual(cache["metadata"]["partition_plan_hash"], "plan-test")


if __name__ == "__main__":
    unittest.main()
