import unittest

from crossmaze import compute_topology_difficulty, topology_difficulty_config
from crossmaze.topology_difficulty import (
    _canonical_undirected_path,
    contrast_normalized_component,
    compute_top_level_path_difficulty,
    compute_top_level_path_metrics,
    logistic_final_score,
    maximal_contiguous_paths,
)
from crossmaze.variants import POINTMAZE_ENV_FACTS


class TopLevelPathMetricsTest(unittest.TestCase):
    def test_reversal_is_canonicalized(self):
        forward = ((1, 1), (1, 2), (2, 2))
        reverse = tuple(reversed(forward))
        self.assertEqual(
            _canonical_undirected_path(forward),
            _canonical_undirected_path(reverse),
        )

    def test_contiguous_subpaths_are_removed(self):
        longest = ((1, 1), (1, 2), (1, 3), (1, 4))
        paths = (
            _canonical_undirected_path(longest),
            _canonical_undirected_path(longest[:3]),
            _canonical_undirected_path(longest[1:]),
            _canonical_undirected_path(longest[1:3]),
        )
        self.assertEqual(maximal_contiguous_paths(paths), (longest,))

    def test_chain_has_one_top_level_path(self):
        maze_map = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
        metrics = compute_top_level_path_metrics(maze_map)
        self.assertEqual(metrics.unordered_pair_count, 3)
        self.assertEqual(metrics.contained_path_count, 2)
        self.assertEqual(metrics.top_level_path_count, 1)
        self.assertEqual(metrics.mean_top_level_path_len, 2.0)
        self.assertEqual(metrics.mean_top_level_turns, 0.0)
        self.assertEqual(metrics.mean_top_level_detour_ratio, 0.0)
        self.assertEqual(metrics.map_area, 15)

    def test_t_junction_has_three_leaf_to_leaf_top_paths(self):
        maze_map = [
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
        metrics = compute_top_level_path_metrics(maze_map)
        self.assertEqual(metrics.top_level_path_count, 3)
        self.assertEqual(metrics.mean_top_level_path_len, 2.0)
        self.assertAlmostEqual(metrics.mean_top_level_turns, 2.0 / 3.0)
        self.assertEqual(metrics.raw_degree_3_cells, 1)
        self.assertEqual(metrics.corridor_t_junctions, 1)
        self.assertEqual(metrics.corridor_cross_junctions, 0)

    def test_open_room_degree_cells_are_not_corridor_junctions(self):
        metrics = compute_top_level_path_metrics(
            POINTMAZE_ENV_FACTS["open"]["maze_map"]
        )
        self.assertGreater(metrics.raw_degree_3_cells, 0)
        self.assertGreater(metrics.raw_degree_4_cells, 0)
        self.assertEqual(metrics.corridor_t_junctions, 0)
        self.assertEqual(metrics.corridor_cross_junctions, 0)

    def test_cross_requires_all_diagonals_to_be_walls(self):
        corridor_cross = [
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
        ]
        open_corner_cross = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
        ]
        corridor_metrics = compute_top_level_path_metrics(corridor_cross)
        open_corner_metrics = compute_top_level_path_metrics(open_corner_cross)
        self.assertEqual(corridor_metrics.corridor_cross_junctions, 1)
        self.assertEqual(open_corner_metrics.raw_degree_4_cells, 1)
        self.assertEqual(open_corner_metrics.corridor_cross_junctions, 0)

    def test_umaze_top_path_detour_ratio(self):
        metrics = compute_top_level_path_metrics(
            POINTMAZE_ENV_FACTS["umaze"]["maze_map"]
        )
        self.assertEqual(metrics.top_level_path_count, 1)
        self.assertAlmostEqual(metrics.mean_top_level_detour_ratio, 2.0 / 3.0)

    def test_v3_component_contrast(self):
        self.assertEqual(contrast_normalized_component(0.0), 0.0)
        self.assertEqual(contrast_normalized_component(0.5), 0.5)
        self.assertEqual(contrast_normalized_component(1.0), 1.0)
        self.assertAlmostEqual(
            contrast_normalized_component(0.2),
            1.0 - contrast_normalized_component(0.8),
        )
        self.assertLess(contrast_normalized_component(0.2), 0.2)
        self.assertGreater(contrast_normalized_component(0.8), 0.8)
        with self.assertRaisesRegex(ValueError, "gamma must be positive"):
            contrast_normalized_component(0.5, gamma=0.0)

    def test_v3_logistic_midpoint(self):
        self.assertEqual(logistic_final_score(45.0), 50.0)

    def test_v3_3_config_and_reference_scores(self):
        config = topology_difficulty_config()
        self.assertEqual(config["version"], "v3.3")
        self.assertEqual(config["component_contrast_gamma"], 1.5)
        self.assertEqual(config["final_score_logistic_midpoint"], 45.0)
        self.assertEqual(config["final_score_logistic_temperature"], 10.0)
        self.assertEqual(
            config["weights"],
            {
                "path_count": 0.05,
                "mean_path_len": 0.25,
                "mean_turns": 0.15,
                "mean_detour_exposure": 0.25,
                "map_area": 0.20,
                "t_junctions": 0.05,
                "cross_junctions": 0.05,
            },
        )

        open_difficulty = compute_top_level_path_difficulty(
            compute_top_level_path_metrics(
                POINTMAZE_ENV_FACTS["open"]["maze_map"]
            )
        )
        umaze_difficulty = compute_top_level_path_difficulty(
            compute_top_level_path_metrics(
                POINTMAZE_ENV_FACTS["umaze"]["maze_map"]
            )
        )
        self.assertEqual(open_difficulty.raw_score, 14.14)
        self.assertEqual(open_difficulty.score, 4.37)
        self.assertEqual(umaze_difficulty.raw_score, 19.82)
        self.assertEqual(umaze_difficulty.score, 7.46)

    def test_public_crossmaze_one_call_api(self):
        result = compute_topology_difficulty(
            POINTMAZE_ENV_FACTS["open"]["maze_map"]
        )
        self.assertEqual(result.version, "v3.3")
        self.assertEqual(result.raw_score, 14.14)
        self.assertEqual(result.score, 4.37)
        self.assertEqual(result.metrics.top_level_path_count, 17)
        self.assertGreater(result.difficulty.path_count_component, 0.0)
        self.assertEqual(result.to_dict()["difficulty"]["version"], "v3.3")

    def test_generated_antmaze_static_difficulty_bands(self):
        from crossmaze.variants import ANTMAZE_ENV_FACTS
        from utils.maze_metrics import choose_farthest_free_pair, compute_maze_metrics

        expected_bands = {
            **{
                f"test-layout-{index:02d}": (40.0, 50.0)
                for index in range(5, 9)
            },
            **{
                f"test-layout-{index:02d}": (57.0, 63.0)
                for index in range(9, 11)
            },
        }
        max_rows = max(
            len(facts["maze_map"]) for facts in ANTMAZE_ENV_FACTS.values()
        )
        max_cols = max(
            len(facts["maze_map"][0]) for facts in ANTMAZE_ENV_FACTS.values()
        )
        self.assertEqual((max_rows, max_cols), (12, 16))

        for name, (lower, upper) in expected_bands.items():
            with self.subTest(variant=name):
                facts = ANTMAZE_ENV_FACTS[name]
                maze_map = facts["maze_map"]
                start, goal = choose_farthest_free_pair(maze_map)
                metrics = compute_maze_metrics(maze_map, start=start, goal=goal)
                self.assertGreaterEqual(metrics.static_difficulty, lower)
                self.assertLessEqual(metrics.static_difficulty, upper)
                self.assertLessEqual(len(maze_map), max_rows)
                self.assertLessEqual(len(maze_map[0]), max_cols)
                self.assertEqual(facts["env_paras"]["max_episode_steps"], 1000)

    def test_generated_pointmaze_v2_static_difficulty_bands(self):
        from utils.maze_metrics import choose_farthest_free_pair, compute_maze_metrics

        expected_bands = {
            "test-layoutV2-07": (40.0, 50.0),
            "test-layoutV2-08": (60.0, 70.0),
            "test-layoutV2-09": (60.0, 70.0),
            "test-layoutV2-10": (78.0, 83.0),
        }
        max_rows = max(
            len(facts["maze_map"]) for facts in POINTMAZE_ENV_FACTS.values()
        )
        max_cols = max(
            len(facts["maze_map"][0]) for facts in POINTMAZE_ENV_FACTS.values()
        )
        self.assertEqual((max_rows, max_cols), (15, 15))

        for name, (lower, upper) in expected_bands.items():
            with self.subTest(variant=name):
                facts = POINTMAZE_ENV_FACTS[name]
                maze_map = facts["maze_map"]
                start, goal = choose_farthest_free_pair(maze_map)
                metrics = compute_maze_metrics(maze_map, start=start, goal=goal)
                self.assertGreaterEqual(metrics.static_difficulty, lower)
                self.assertLessEqual(metrics.static_difficulty, upper)
                self.assertLessEqual(len(maze_map), max_rows)
                self.assertLessEqual(len(maze_map[0]), max_cols)
                self.assertEqual(
                    facts["env_paras"]["max_episode_steps"],
                    6 * len(maze_map) * len(maze_map[0]),
                )

    def test_pointmaze_v3_3_ordering_contract(self):
        names = [
            "open",
            "umaze",
            "medium",
            "large",
            *[f"local-layoutV2-{index:02d}" for index in range(1, 13)],
            *[f"test-layoutV2-{index:02d}" for index in range(1, 11)],
        ]
        scores = {
            name: compute_top_level_path_difficulty(
                compute_top_level_path_metrics(
                    POINTMAZE_ENV_FACTS[name]["maze_map"]
                )
            ).score
            for name in names
        }
        other_scores = [
            score
            for name, score in scores.items()
            if name not in {"open", "umaze"}
        ]
        self.assertLess(scores["open"], scores["umaze"])
        self.assertLess(scores["umaze"], min(other_scores))


if __name__ == "__main__":
    unittest.main()
