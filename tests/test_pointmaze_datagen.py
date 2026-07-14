import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import local_pointmaze_gen


class PointMazeDataGenerationTest(unittest.TestCase):
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
        }
        invalid_cases = [
            ({"hard_retry": -1}, "--hard-retry must be >= 0"),
            ({"hard_sample_alpha": -0.1}, "--hard-sample-alpha must be >= 0"),
            ({"hard_sample_top_n": -1}, "--hard-sample-top-n must be >= 0"),
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
        self.assertEqual(kwargs["reward_type"], "dense")

    def test_hard_sample_sbatch_exposes_all_controls(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "sbatch"
            / "dataGen.point.hard.slurm"
        ).read_text(encoding="utf-8")

        for flag in (
            "--hard-sample",
            "--hard-retry",
            "--hard-sample-alpha",
            "--hard-sample-top-n",
            "--reward-type",
        ):
            self.assertIn(flag, script)
        self.assertIn("#SBATCH --array=0-11%4", script)
        self.assertIn("local-layoutV2-12", script)


if __name__ == "__main__":
    unittest.main()
