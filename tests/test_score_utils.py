import json
import tempfile
import unittest

import gymnasium_robotics  # noqa: F401  registers PointMaze envs

from utils.pointmaze_score import (
    build_local_pointmaze_score_env_spec,
    build_remote_pointmaze_score_env_spec,
    get_remote_pointmaze_reference,
    load_and_validate_local_reference,
    local_reference_path,
    make_pointmaze_score_env,
    normalize_score,
    normalize_score_std,
)


LOCAL_LAYOUT_07_FREE_GOAL_CELL = [6, 6]
LOCAL_LAYOUT_07_WALL_CELL = [7, 7]


class PointMazeScoreUtilsTest(unittest.TestCase):
    def test_normalized_score_formula(self):
        self.assertAlmostEqual(normalize_score(15.0, 10.0, 20.0), 50.0)
        self.assertAlmostEqual(normalize_score_std(2.0, 10.0, 20.0), 20.0)

    def test_remote_reference_lookup(self):
        ref = get_remote_pointmaze_reference("open")
        self.assertAlmostEqual(ref["ref_min_score"], 7.199999809265137)
        self.assertAlmostEqual(ref["ref_max_score"], 229.86000061035156)
        self.assertEqual(ref["reference_source"], "minari_d4rl_metadata")
        with self.assertRaises(ValueError):
            get_remote_pointmaze_reference("local-layout-07")

    def test_remote_official_env_horizon_and_kwargs(self):
        spec = build_remote_pointmaze_score_env_spec("medium-dense")
        self.assertEqual(spec.env_id, "PointMaze_MediumDense-v3")
        self.assertEqual(spec.max_episode_steps, 600)
        self.assertEqual(spec.env_kwargs["reward_type"], "dense")
        self.assertTrue(spec.env_kwargs["continuing_task"])
        self.assertFalse(spec.env_kwargs["reset_target"])

        env = make_pointmaze_score_env(spec)
        try:
            self.assertEqual(env.spec.max_episode_steps, 600)
            self.assertTrue(env.unwrapped.continuing_task)
            self.assertFalse(env.unwrapped.reset_target)
        finally:
            env.close()

    def test_local_goal_cell_validation_and_reference_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "local_reference_root": tmpdir,
                "local_eval_maps": {
                    "local-layout-07": {"goal_cell": LOCAL_LAYOUT_07_FREE_GOAL_CELL}
                },
            }
            spec = build_local_pointmaze_score_env_spec("local-layout-07", config)
            goal_count = sum(
                1 for row in spec.env_kwargs["maze_map"] for value in row if value == "g"
            )
            self.assertEqual(goal_count, 1)
            self.assertEqual(spec.goal_cell, LOCAL_LAYOUT_07_FREE_GOAL_CELL)

            path = local_reference_path(config, "local-layout-07")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ref_min_score": 1.0,
                        "ref_max_score": 2.0,
                        "env_fingerprint": spec.env_fingerprint,
                    },
                    f,
                )
            reference, loaded_path = load_and_validate_local_reference(
                config=config,
                variant="local-layout-07",
                score_env_spec=spec,
            )
            self.assertEqual(loaded_path, path)
            self.assertEqual(reference["ref_min_score"], 1.0)

    def test_local_score_horizon_default_and_override(self):
        default_config = {
            "local_eval_maps": {
                "local-layout-07": {"goal_cell": LOCAL_LAYOUT_07_FREE_GOAL_CELL}
            },
        }
        default_spec = build_local_pointmaze_score_env_spec(
            "local-layout-07",
            default_config,
        )
        self.assertEqual(default_spec.max_episode_steps, 624)

        override_config = {
            "local_eval_maps": {
                "local-layout-07": {
                    "goal_cell": LOCAL_LAYOUT_07_FREE_GOAL_CELL,
                    "max_episode_steps": 600,
                }
            },
        }
        override_spec = build_local_pointmaze_score_env_spec(
            "local-layout-07",
            override_config,
        )
        self.assertEqual(override_spec.max_episode_steps, 600)

    def test_local_score_reward_type_override_changes_fingerprint(self):
        base_config = {
            "local_eval_maps": {
                "local-layout-07": {"goal_cell": LOCAL_LAYOUT_07_FREE_GOAL_CELL}
            },
        }
        sparse_spec = build_local_pointmaze_score_env_spec(
            "local-layout-07",
            base_config,
        )
        dense_spec = build_local_pointmaze_score_env_spec(
            "local-layout-07",
            {**base_config, "reward_type": "dense"},
        )

        self.assertEqual(sparse_spec.reward_type, "sparse")
        self.assertEqual(dense_spec.reward_type, "dense")
        self.assertEqual(dense_spec.env_kwargs["reward_type"], "dense")
        self.assertNotEqual(sparse_spec.env_fingerprint, dense_spec.env_fingerprint)

    def test_local_reference_missing_and_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "local_reference_root": tmpdir,
                "local_eval_maps": {
                    "local-layout-07": {"goal_cell": LOCAL_LAYOUT_07_FREE_GOAL_CELL}
                },
            }
            spec = build_local_pointmaze_score_env_spec("local-layout-07", config)
            with self.assertRaises(FileNotFoundError):
                load_and_validate_local_reference(
                    config=config,
                    variant="local-layout-07",
                    score_env_spec=spec,
                )

            path = local_reference_path(config, "local-layout-07")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ref_min_score": 1.0,
                        "ref_max_score": 2.0,
                        "env_fingerprint": "wrong",
                    },
                    f,
                )
            with self.assertRaises(ValueError):
                load_and_validate_local_reference(
                    config=config,
                    variant="local-layout-07",
                    score_env_spec=spec,
                )

    def test_local_goal_cell_must_be_free(self):
        config = {
            "local_eval_maps": {
                "local-layout-07": {"goal_cell": LOCAL_LAYOUT_07_WALL_CELL}
            },
        }
        with self.assertRaises(ValueError):
            build_local_pointmaze_score_env_spec("local-layout-07", config)


if __name__ == "__main__":
    unittest.main()
