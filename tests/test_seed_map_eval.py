import unittest

from crossmaze.eval_position import (
    build_seed_map_eval_position_config,
    resolve_seed_map_eval_position_mode,
    select_seed_map_eval_position,
)
from crossmaze.seed_map import generate_seed_map
from utils.seed_map_eval import (
    resolve_seed_map_eval_selection,
    seed_map_eval_target,
)


class SeedMapEvalTests(unittest.TestCase):
    def test_selection_is_deterministic_and_supports_disjoint_ranges(self):
        section = {
            "enabled": True,
            "seed_ranges": [[0, 10], [20, 25]],
            "seed_count": 4,
            "selection_seed": 7,
            "seed_map_size_mode": "random",
            "seed_map_min_size": 5,
            "seed_map_max_size": 9,
        }
        first = resolve_seed_map_eval_selection(
            section,
            env_family="pointmaze",
        )
        second = resolve_seed_map_eval_selection(
            section,
            env_family="pointmaze",
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first.selected_seeds), 4)
        self.assertTrue(
            all(0 <= seed < 10 or 20 <= seed < 25 for seed in first.selected_seeds)
        )
        target = seed_map_eval_target(
            {"resolved_seed_map_eval": first.to_dict()},
            first.selected_variants[0],
        )
        self.assertEqual(target["map_seed"], first.selected_seeds[0])
        self.assertNotIn("episodes_per_seed", target)
        self.assertNotIn("episodes_per_seed", first.to_dict())
        self.assertRegex(first.selection_tag, r"^seed-map-eval-n4-[0-9a-f]{10}$")

    def test_selection_rejects_legacy_episode_count(self):
        with self.assertRaisesRegex(ValueError, "Unknown seed_map_eval fields"):
            resolve_seed_map_eval_selection(
                {
                    "enabled": True,
                    "seed_ranges": [[0, 1]],
                    "episodes_per_seed": 3,
                    "seed_map_size_mode": "fixed",
                    "seed_map_fixed_rows": 5,
                    "seed_map_fixed_cols": 5,
                },
                env_family="pointmaze",
            )

    def test_eval_defaults_to_random_start_goal_for_antmaze(self):
        self.assertEqual(
            resolve_seed_map_eval_position_mode({}),
            "random-start-goal",
        )
        with self.assertRaisesRegex(ValueError, "no registered canonical pair"):
            resolve_seed_map_eval_position_mode(
                {"eval_start_goal_mode": "fix-start-goal"}
            )

    def test_hard_sample_pool_is_deterministic(self):
        maze_map = generate_seed_map(
            3,
            {
                "size_mode": "fixed",
                "fixed_rows": 5,
                "fixed_cols": 5,
            },
        )
        config = {
            "eval_start_goal_mode": "hard-sample",
            "eval_hard_sample_top_n": 4,
            "eval_hard_sample_alpha": 1.0,
        }
        first = build_seed_map_eval_position_config(
            "antmaze",
            3,
            maze_map,
            seed=9,
            config=config,
        )
        second = build_seed_map_eval_position_config(
            "antmaze",
            3,
            maze_map,
            seed=9,
            config=config,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first["start_goal_list"]), 4)
        record = select_seed_map_eval_position(
            "antmaze",
            3,
            episode_index=0,
            seed=9,
            position_config=first,
            config=config,
        )
        self.assertEqual(record["source"], "hard_sample")


if __name__ == "__main__":
    unittest.main()
