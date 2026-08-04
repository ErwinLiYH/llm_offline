import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import local_pointmaze_gen


class PointMazeDataGenerationTest(unittest.TestCase):
    def test_cli_defaults_hard_sample_max_path_len_to_40(self):
        with mock.patch("sys.argv", ["local_pointmaze_gen.py"]):
            args = local_pointmaze_gen.parse_args()

        self.assertEqual(args.hard_sample_max_path_len, 40)

    def test_hard_sample_pair_space_uses_legacy_generation_difficulty(self):
        maze_map = [
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
        pair_space, total = local_pointmaze_gen._build_hard_sample_pair_space(
            maze_map,
            [(1, 1), (1, 3), (2, 2)],
            hard_sample_alpha=2.0,
        )
        bounded_top_pairs, bounded_total = (
            local_pointmaze_gen._build_hard_sample_pair_space(
                maze_map,
                [(1, 1), (1, 3), (2, 2)],
                hard_sample_alpha=2.0,
                hard_sample_top_n=2,
                hard_sample_max_path_len=2,
            )
        )
        hard_pair = next(
            pair
            for pair in pair_space
            if pair["start_cell"] == (1, 1) and pair["goal_cell"] == (1, 3)
        )

        self.assertEqual(total, len(pair_space))
        self.assertEqual(hard_pair["path_len"], 4)
        self.assertEqual(hard_pair["away_steps"], 1)
        self.assertAlmostEqual(hard_pair["difficulty"], 0.625)
        self.assertAlmostEqual(
            max(pair["sample_weight"] for pair in pair_space)
            / min(pair["sample_weight"] for pair in pair_space),
            3.0,
        )
        self.assertEqual(bounded_total, total)
        self.assertEqual(len(bounded_top_pairs), 2)
        self.assertTrue(
            all(pair["path_len"] <= 2 for pair in bounded_top_pairs)
        )

    def test_hard_sample_generation_summary_records_saved_difficulty(self):
        pair_space = [
            {
                "start_cell": (1, 1),
                "goal_cell": (1, 3),
                "path_len": 4,
                "away_steps": 1,
                "away_frac": 0.25,
                "difficulty": 0.625,
                "sample_weight": 2.0,
            }
        ]
        episode_record = local_pointmaze_gen._difficulty_record_for_episode(
            pair_space[0],
            episode_index=0,
            attempts_for_pair=2,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            local_pointmaze_gen._write_generation_summary(
                dataset_root=Path(temp_dir),
                variant="local-layoutV2-01",
                reward_type="dense",
                target_episodes=3,
                final_episodes=3,
                seed=42,
                max_episode_steps=100,
                post_success_hold_steps=0,
                post_success_hold_noise_std=0.0,
                hard_sample=True,
                hard_retry=5,
                hard_sample_alpha=1.0,
                hard_sample_top_n=400,
                hard_sample_max_path_len=40,
                hard_pair_space=pair_space,
                hard_pair_space_total=10,
                shard_results=[
                    {
                        "attempted_episodes": 2,
                        "collected_steps": 20,
                        "hard_pairs_sampled": 1,
                        "hard_pairs_succeeded": 1,
                        "hard_pairs_exhausted": 0,
                        "hard_failed_attempts": 1,
                        "episode_difficulty": [episode_record],
                    }
                ],
            )
            summary = json.loads(
                (Path(temp_dir) / "generation_summary.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(summary["hard_sample"])
        self.assertEqual(summary["reward_type"], "dense")
        self.assertEqual(summary["hard_pair_space_total"], 10)
        self.assertEqual(summary["hard_pair_space_used"], 1)
        self.assertEqual(summary["hard_sample_max_path_len"], 40)
        self.assertEqual(summary["hard_failed_attempts"], 1)
        self.assertEqual(summary["episode_difficulty"][0]["difficulty"], 0.625)
        self.assertEqual(summary["episode_difficulty"][0]["attempts_for_pair"], 2)

    def test_main_validates_and_forwards_hard_sample_options(self):
        base_args = {
            "variants": ["local-layoutV2-01"],
            "num_workers": 1,
            "target_episodes": 1,
            "overwrite": False,
            "seed": 42,
            "reward_type": "dense",
            "max_episode_steps": 100,
            "post_success_hold_steps": 0,
            "post_success_hold_noise_std": 0.0,
            "hard_sample": True,
            "hard_retry": 5,
            "hard_sample_alpha": 1.0,
            "hard_sample_top_n": 400,
            "hard_sample_max_path_len": 40,
            "dataset_root": Path("/tmp/point-final"),
            "temporary_dataset_root": Path("/tmp/point-shards"),
        }
        invalid_cases = [
            ({"hard_retry": -1}, "--hard-retry must be >= 0"),
            ({"hard_sample_alpha": -0.1}, "--hard-sample-alpha must be >= 0"),
            ({"hard_sample_top_n": -1}, "--hard-sample-top-n must be >= 0"),
            (
                {"hard_sample_max_path_len": -1},
                "--hard-sample-max-path-len must be >= 0",
            ),
        ]
        for overrides, error in invalid_cases:
            with self.subTest(overrides=overrides), mock.patch(
                "local_pointmaze_gen.parse_args",
                return_value=SimpleNamespace(**{**base_args, **overrides}),
            ):
                with self.assertRaisesRegex(ValueError, error):
                    local_pointmaze_gen.main()

        with mock.patch(
            "local_pointmaze_gen.parse_args",
            return_value=SimpleNamespace(**base_args),
        ), mock.patch("local_pointmaze_gen.generate_variant") as generate_variant:
            local_pointmaze_gen.main()

        kwargs = generate_variant.call_args.kwargs
        self.assertTrue(kwargs["hard_sample"])
        self.assertEqual(kwargs["hard_retry"], 5)
        self.assertEqual(kwargs["hard_sample_alpha"], 1.0)
        self.assertEqual(kwargs["hard_sample_top_n"], 400)
        self.assertEqual(kwargs["hard_sample_max_path_len"], 40)
        self.assertEqual(kwargs["reward_type"], "dense")
        self.assertEqual(kwargs["dataset_root_override"], Path("/tmp/point-final"))
        self.assertEqual(
            kwargs["temporary_dataset_root"],
            Path("/tmp/point-shards"),
        )

    def test_seed_map_manifest_records_default_max_path_len(self):
        spec = local_pointmaze_gen.normalize_seed_map_spec(
            {
                "version": "v1",
                "size_mode": "fixed",
                "fixed_rows": 9,
                "fixed_cols": 9,
            }
        )
        args = SimpleNamespace(
            reward_type="sparse",
            temporary_dataset_root=Path("/tmp/point-shards"),
            seed=42,
            max_episode_steps=100,
            post_success_hold_steps=0,
            post_success_hold_noise_std=0.0,
            hard_sample=True,
            hard_retry=5,
            hard_sample_alpha=0.0,
            hard_sample_top_n=200,
            hard_sample_max_path_len=40,
            seed_map_start=1,
            seed_map_end=2,
            seed_map_trajectories_per_seed=1,
            overwrite=False,
        )
        with mock.patch(
            "local_pointmaze_gen.validate_seed_map_generation_args",
            return_value=spec,
        ), mock.patch(
            "local_pointmaze_gen.resolve_temporary_dataset_root",
            return_value=Path("/tmp/point-shards"),
        ), mock.patch(
            "local_pointmaze_gen.resolve_seed_map_generation_path",
            return_value=Path("/tmp/point-corpus"),
        ), mock.patch(
            "local_pointmaze_gen.create_seed_map_corpus",
            return_value={"complete": True},
        ) as create_corpus:
            local_pointmaze_gen.generate_seed_map_corpus(args)

        collection_config = create_corpus.call_args.kwargs["collection_config"]
        self.assertEqual(collection_config["hard_sample_max_path_len"], 40)

    def test_seed_map_worker_applies_max_path_len_before_collection(self):
        maze_map = [
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
        spec = local_pointmaze_gen.normalize_seed_map_spec(
            {
                "version": "v1",
                "size_mode": "fixed",
                "fixed_rows": 5,
                "fixed_cols": 5,
            }
        )
        with mock.patch(
            "local_pointmaze_gen.generate_seed_map",
            return_value=maze_map,
        ), mock.patch(
            "local_pointmaze_gen.build_seed_map_env_paras",
            return_value={"max_episode_steps": 100},
        ), mock.patch(
            "local_pointmaze_gen._collect_shard",
            return_value={"dataset_id": "test", "path": "/tmp/test"},
        ) as collect_shard:
            result = local_pointmaze_gen._collect_seed_map_shard_from_kwargs(
                {
                    "map_seed": 7,
                    "seed_map_spec": spec.to_dict(),
                    "reward_type": "sparse",
                    "max_episode_steps": 100,
                    "hard_sample": True,
                    "hard_sample_alpha": 0.0,
                    "hard_sample_top_n": 2,
                    "hard_sample_max_path_len": 2,
                }
            )

        pairs = collect_shard.call_args.kwargs["hard_pair_space"]
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(pair["path_len"] <= 2 for pair in pairs))
        self.assertGreater(result["hard_pair_space_total"], len(pairs))

    def test_hard_sample_scripts_expose_default_max_path_len(self):
        sbatch_dir = Path(__file__).resolve().parents[1] / "sbatch"
        script_names = (
            "dataGen.point.hard.sh",
            "dataGen.point.hard.slurm",
            "dataGen.point.hard.seedmap.sh",
            "dataGen.point.hard.seedmap.slurm",
        )
        for script_name in script_names:
            with self.subTest(script=script_name):
                script = (sbatch_dir / script_name).read_text(encoding="utf-8")
                for flag in (
                    "--hard-sample",
                    "--hard-retry",
                    "--hard-sample-alpha",
                    "--hard-sample-top-n",
                    "--hard-sample-max-path-len",
                    "--reward-type",
                ):
                    self.assertIn(flag, script)
                self.assertIn(
                    "HARD_SAMPLE_MAX_PATH_LEN=${HARD_SAMPLE_MAX_PATH_LEN:-40}",
                    script,
                )

        fixed_slurm = (sbatch_dir / "dataGen.point.hard.slurm").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --array=0-11%4", fixed_slurm)
        self.assertIn("local-layoutV2-12", fixed_slurm)


if __name__ == "__main__":
    unittest.main()
