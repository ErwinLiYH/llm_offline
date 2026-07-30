import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import h5py
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from crossmaze.seed_map import SeedMapSpec, generate_seed_map
from data.base_dataset import DatasetBuildRequest
from data.pointmaze.dataset import PointMazeDataset
from data.seed_map_corpus import (
    append_minari_shard_to_seed_map_corpus,
    build_seed_map_episode_selection,
    create_seed_map_corpus,
    finalize_seed_map_corpus,
    normalize_seed_ranges,
    resolve_seed_map_selection,
    seed_map_dataset_name,
)


def _write_fake_shard(root: Path, episode_count: int, action_dim: int = 2):
    path = root / "data" / "main_data.hdf5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        for episode_index in range(episode_count):
            group = handle.create_group(f"episode_{episode_index}")
            observations = group.create_group("observations")
            observations.create_dataset(
                "observation",
                data=np.zeros((4, 4), dtype=np.float32),
            )
            observations.create_dataset(
                "achieved_goal",
                data=np.zeros((4, 2), dtype=np.float32),
            )
            observations.create_dataset(
                "desired_goal",
                data=np.ones((4, 2), dtype=np.float32),
            )
            group.create_dataset(
                "actions",
                data=np.zeros((3, action_dim), dtype=np.float32),
            )
            group.create_dataset("rewards", data=np.zeros(3, dtype=np.float32))


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


class SeedMapCorpusTests(unittest.TestCase):
    def test_dataset_name_uses_half_open_range(self):
        self.assertEqual(seed_map_dataset_name(0, 1000, 10), "0-1000-10")

    def test_ranges_reject_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlapping"):
            normalize_seed_ranges(
                [[0, 5], [4, 8]],
                default_start=0,
                default_end=10,
            )

    def test_create_append_finalize_and_select(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "0-3-2"
            create_seed_map_corpus(
                root,
                env_family="pointmaze",
                reward_type="sparse",
                seed_start=0,
                seed_end=3,
                trajectories_per_seed=2,
                seed_map_spec=SeedMapSpec(
                    size_mode="fixed",
                    fixed_rows=7,
                    fixed_cols=7,
                ),
            )
            for map_seed in range(3):
                shard = Path(tmp) / f"shard-{map_seed}"
                _write_fake_shard(shard, 2)
                append_minari_shard_to_seed_map_corpus(
                    root,
                    shard,
                    map_seed=map_seed,
                    maze_map=generate_seed_map(
                        map_seed,
                        SeedMapSpec(
                            size_mode="fixed",
                            fixed_rows=7,
                            fixed_cols=7,
                        ),
                    ),
                    trajectory_start_index=0,
                    trajectory_count=2,
                )
            manifest = finalize_seed_map_corpus(root)
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["total_episodes"], 6)

            selection = resolve_seed_map_selection(
                {
                    "enabled": True,
                    "dataset_path": str(root),
                    "seed_ranges": [[0, 3]],
                    "seed_count": 2,
                    "trajectories_per_seed": 1,
                    "selection_seed": 9,
                    "split_unit": "seed",
                },
                env_family="pointmaze",
            )
            self.assertEqual(len(selection.selected_seeds), 2)
            self.assertEqual(len(selection.selected_keys), 2)
            split = build_seed_map_episode_selection(
                selection,
                train_data_ratio=0.5,
            )
            self.assertEqual(split["train_episode_count"], 1)
            self.assertEqual(split["val_episode_count"], 1)
            train_seed = split["records"][split["train_indices"][0]]["map_seed"]
            val_seed = split["records"][split["val_indices"][0]]["map_seed"]
            self.assertNotEqual(train_seed, val_seed)

    def test_incomplete_corpus_cannot_be_selected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "0-2-1"
            create_seed_map_corpus(
                root,
                env_family="pointmaze",
                reward_type="sparse",
                seed_start=0,
                seed_end=2,
                trajectories_per_seed=1,
                seed_map_spec=None,
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                resolve_seed_map_selection(
                    {
                        "enabled": True,
                        "dataset_path": str(root),
                        "trajectories_per_seed": 1,
                    },
                    env_family="pointmaze",
                )

    def test_resume_removes_unindexed_interrupted_hdf5_group(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "0-1-1"
            kwargs = {
                "env_family": "pointmaze",
                "reward_type": "sparse",
                "seed_start": 0,
                "seed_end": 1,
                "trajectories_per_seed": 1,
                "seed_map_spec": None,
            }
            create_seed_map_corpus(root, **kwargs)
            with h5py.File(root / "data" / "main_data.hdf5", "a") as handle:
                handle.create_group("episode_0")

            resumed = create_seed_map_corpus(root, **kwargs)

            self.assertFalse(resumed["complete"])
            with h5py.File(root / "data" / "main_data.hdf5", "r") as handle:
                self.assertNotIn("episode_0", handle)

    def test_pointmaze_dataset_builds_from_synthetic_source_tag(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "0-2-1"
            tokenizer_path = Path(tmp) / "tokenizer"
            tokenizer = _make_test_tokenizer(str(tokenizer_path))
            spec = SeedMapSpec(
                size_mode="fixed",
                fixed_rows=7,
                fixed_cols=7,
            )
            create_seed_map_corpus(
                root,
                env_family="pointmaze",
                reward_type="sparse",
                seed_start=0,
                seed_end=2,
                trajectories_per_seed=1,
                seed_map_spec=spec,
            )
            for map_seed in range(2):
                shard = Path(tmp) / f"dataset-shard-{map_seed}"
                _write_fake_shard(shard, 1)
                append_minari_shard_to_seed_map_corpus(
                    root,
                    shard,
                    map_seed=map_seed,
                    maze_map=generate_seed_map(map_seed, spec),
                    trajectory_start_index=0,
                    trajectory_count=1,
                )
            finalize_seed_map_corpus(root)
            section = {
                "enabled": True,
                "dataset_path": str(root),
                "seed_ranges": [[0, 2]],
                "seed_count": 2,
                "trajectories_per_seed": 1,
                "selection_seed": 3,
                "split_unit": "seed",
            }
            selection = resolve_seed_map_selection(
                section,
                env_family="pointmaze",
            )
            common = {
                "variant": selection.source_tag,
                "tokenizer": tokenizer,
                "tokenizer_name_or_path": str(tokenizer_path),
                "num_workers": 1,
                "prompt_templete_index": ["0"],
                "train_data_ratio": 0.5,
                "seed_map_selection": section,
            }
            train_dataset, val_dataset = PointMazeDataset.build_batch(
                [
                    DatasetBuildRequest(split="train", **common),
                    DatasetBuildRequest(split="val", **common),
                ]
            )

            self.assertEqual(len(train_dataset), 3)
            self.assertEqual(len(val_dataset), 3)


if __name__ == "__main__":
    unittest.main()
