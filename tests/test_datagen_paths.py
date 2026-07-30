import argparse
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from utils.datagen_paths import (
    add_datagen_path_args,
    configure_minari_temporary_root,
    publish_new_dataset,
    remove_temporary_dataset,
    resolve_final_dataset_path,
    resolve_temporary_dataset_root,
    temporary_dataset_merge_path,
)
from utils.seed_map_config import (
    add_seed_map_generation_args,
    resolve_seed_map_generation_path,
)


class DataGenerationPathTest(unittest.TestCase):
    def test_cli_exposes_final_and_temporary_roots_with_compatibility_aliases(self):
        parser = argparse.ArgumentParser()
        add_datagen_path_args(parser)
        add_seed_map_generation_args(parser)

        canonical = parser.parse_args(
            [
                "--dataset-root",
                "/final",
                "--temporary-dataset-root",
                "/temporary",
            ]
        )
        self.assertEqual(canonical.dataset_root, Path("/final"))
        self.assertEqual(
            canonical.temporary_dataset_root,
            Path("/temporary"),
        )

        aliases = parser.parse_args(
            [
                "--seed-map-dataset-root",
                "/legacy-final",
                "--temp-dataset-root",
                "/short-temporary",
            ]
        )
        self.assertEqual(aliases.dataset_root, Path("/legacy-final"))
        self.assertEqual(
            aliases.temporary_dataset_root,
            Path("/short-temporary"),
        )

    def test_final_dataset_root_preserves_derived_leaf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            default = parent / "old" / "pointmaze-local-layout-01-v0"
            configured_parent = parent / "new"
            exact = configured_parent / default.name

            self.assertEqual(
                resolve_final_dataset_path(default, None),
                default.resolve(),
            )
            self.assertEqual(
                resolve_final_dataset_path(default, configured_parent),
                exact.resolve(),
            )
            self.assertEqual(
                resolve_final_dataset_path(default, exact),
                exact.resolve(),
            )

    def test_seed_map_final_root_uses_canonical_dataset_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                dataset_root=Path(temp_dir),
                seed_map_start=0,
                seed_map_end=100,
                seed_map_trajectories_per_seed=50,
            )
            resolved = resolve_seed_map_generation_path(
                args,
                env_family="antmaze",
                reward_type="sparse",
                repo_root=Path("/unused"),
                spec=SimpleNamespace(
                    size_mode="random",
                    version="v1",
                    min_size=9,
                    max_size=13,
                ),
            )
            self.assertEqual(
                resolved,
                Path(temp_dir).resolve()
                / "antmaze-seed-map-v1-random-size9-13-sparse"
                / "0-100-50",
            )
            namespace = (
                Path(temp_dir)
                / "antmaze-seed-map-v1-random-size9-13-sparse"
            )
            args.dataset_root = namespace
            self.assertEqual(
                resolve_seed_map_generation_path(
                    args,
                    env_family="antmaze",
                    reward_type="sparse",
                    repo_root=Path("/unused"),
                    spec=SimpleNamespace(
                        size_mode="random",
                        version="v1",
                        min_size=9,
                        max_size=13,
                    ),
                ),
                namespace.resolve() / "0-100-50",
            )
            exact = namespace / "0-100-50"
            args.dataset_root = exact
            self.assertEqual(
                resolve_seed_map_generation_path(
                    args,
                    env_family="antmaze",
                    reward_type="sparse",
                    repo_root=Path("/unused"),
                    spec=SimpleNamespace(
                        size_mode="random",
                        version="v1",
                        min_size=9,
                        max_size=13,
                    ),
                ),
                exact.resolve(),
            )

    def test_temporary_root_defaults_to_tmpdir_and_configures_minari(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ,
            {"TMPDIR": temp_dir},
        ):
            expected = Path(temp_dir).resolve()
            self.assertEqual(resolve_temporary_dataset_root(None), expected)
            with mock.patch.dict(os.environ, {}, clear=False):
                self.assertEqual(
                    configure_minari_temporary_root(expected),
                    expected,
                )
                self.assertEqual(os.environ["MINARI_DATASETS_PATH"], str(expected))

    def test_temporary_cleanup_requires_matching_dataset_leaf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_id = "pointmaze-shard-test-v0"
            dataset_path = root / dataset_id
            dataset_path.mkdir()
            (dataset_path / "marker").write_text("temporary", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "path leaf differs"):
                remove_temporary_dataset("different-id-v0", dataset_path)
            self.assertTrue(dataset_path.exists())

            remove_temporary_dataset(dataset_id, dataset_path)
            self.assertFalse(dataset_path.exists())

    def test_temporary_merge_path_is_inside_configured_root_and_is_cleaned(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "shards"
            with temporary_dataset_merge_path(
                root,
                label="pointmaze seed map 7",
            ) as merged_path:
                self.assertEqual(merged_path.name, "dataset")
                self.assertEqual(merged_path.parent.parent, root.resolve())
                (merged_path / "data").mkdir(parents=True)
                (merged_path / "data" / "marker").write_text(
                    "merged",
                    encoding="utf-8",
                )
                workspace = merged_path.parent
                self.assertTrue(workspace.exists())

            self.assertFalse(workspace.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_publish_new_dataset_exposes_only_completed_transfer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            staged = root / "temporary" / "merged"
            final = root / "final" / "dataset-v0"
            (staged / "data").mkdir(parents=True)
            (staged / "data" / "marker").write_text("ready", encoding="utf-8")

            publish_new_dataset(staged, final)

            self.assertEqual(
                (final / "data" / "marker").read_text(encoding="utf-8"),
                "ready",
            )
            self.assertEqual(
                list(final.parent.glob(f".{final.name}.publishing-*")),
                [],
            )
            with self.assertRaises(FileExistsError):
                publish_new_dataset(staged, final)


if __name__ == "__main__":
    unittest.main()
