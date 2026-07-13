"""Long-term tests for the CrossMaze environment package boundary."""

import json
import math
import os
import subprocess
import sys
import unittest

import numpy as np

from data.antmaze.variants import ANTMAZE_VARIANTS
from data.pointmaze.variants import POINTMAZE_VARIANTS


class CrossMazePackageTest(unittest.TestCase):
    def test_eval_pointmaze_obs_schema_and_static_layout(self):
        import crossmaze

        env = crossmaze.make(
            "pointmaze",
            "umaze",
            mode="eval",
            config={"wall_sensing_version": "v5"},
        )
        try:
            obs, _info = env.reset(seed=7)
            state = obs[crossmaze.CROSSMAZE_OBS_KEY]
            json.dumps(state)
            for key in (
                "maze_map",
                "maze_size_scaling",
                "maze_shape",
                "position_xy",
                "goal_xy",
                "position_cell",
                "goal_cell",
                "neighbor_status",
                "wall_sensing_version",
                "map_sensing_boundary_risk_threshold",
            ):
                self.assertIn(key, state)
            self.assertEqual(state["wall_sensing_version"], "v5")
            self.assertEqual(state["map_sensing_boundary_risk_threshold"], 0.10)
            self.assertEqual(
                set(state["neighbor_status"]), {"up", "down", "left", "right"}
            )
            pv = POINTMAZE_VARIANTS["umaze"]["prompt_vars"]
            self.assertEqual(
                state["maze_map"], [list(row) for row in pv["maze_map"]]
            )
            self.assertIn("observation", obs)
            self.assertIn("desired_goal", obs)

            obs2, _r, _t, _tr, _i = env.step(env.action_space.sample())
            self.assertIn(crossmaze.CROSSMAZE_OBS_KEY, obs2)

            from utils.maze_sensing import compute_sensing_state

            meta = dict(pv)
            meta["wall_sensing_version"] = "v5"
            expected = compute_sensing_state(
                obs2["observation"].astype(np.float32),
                obs2["desired_goal"].astype(np.float32),
                meta,
            )
            for key, value in expected.items():
                self.assertEqual(obs2[crossmaze.CROSSMAZE_OBS_KEY][key], value, key)
        finally:
            env.close()

    def test_eval_antmaze_uses_live_layout(self):
        import crossmaze

        env = crossmaze.make("antmaze", "umaze", mode="eval", config={})
        try:
            live_map = [list(row) for row in env.unwrapped.maze.maze_map]
            obs, _info = env.reset(seed=3)
            state = obs[crossmaze.CROSSMAZE_OBS_KEY]
            self.assertEqual(state["maze_map"], live_map)
            self.assertEqual(
                state["maze_size_scaling"],
                float(env.unwrapped.maze.maze_size_scaling),
            )
            json.dumps(state)
        finally:
            env.close()

    def test_score_mode_uses_static_variant_map_without_goal_marks(self):
        import crossmaze

        env = crossmaze.make("pointmaze", "medium", mode="score", config={})
        try:
            obs, _info = env.reset(seed=11)
            state = obs[crossmaze.CROSSMAZE_OBS_KEY]
            flat = [cell for row in state["maze_map"] for cell in row]
            self.assertNotIn("g", flat)
            self.assertNotIn("r", flat)
            pv = POINTMAZE_VARIANTS["medium"]["prompt_vars"]
            self.assertEqual(
                state["maze_map"], [list(row) for row in pv["maze_map"]]
            )
        finally:
            env.close()

    def test_score_mode_rejects_antmaze(self):
        import crossmaze

        with self.assertRaises(ValueError):
            crossmaze.make("antmaze", "umaze", mode="score", config={})

    def test_score_helper_registers_gymnasium_robotics_envs(self):
        from crossmaze.score import (
            build_pointmaze_score_env_spec,
            make_pointmaze_score_env,
        )

        spec = build_pointmaze_score_env_spec("umaze", {})
        env = make_pointmaze_score_env(spec)
        try:
            obs, _info = env.reset(seed=0)
            self.assertIn("observation", obs)
            self.assertIn("desired_goal", obs)
        finally:
            env.close()

    def test_assert_meta_consistent(self):
        import crossmaze
        from utils.sensing_config import apply_sensing_config_to_prompt_vars

        env = crossmaze.make("pointmaze", "umaze", mode="eval", config={})
        try:
            pv = apply_sensing_config_to_prompt_vars(
                POINTMAZE_VARIANTS["umaze"]["prompt_vars"], {}
            )
            env.assert_meta_consistent(pv)
            bad = dict(pv)
            bad["wall_sensing_version"] = "v5"
            with self.assertRaises(ValueError):
                env.assert_meta_consistent(bad)
        finally:
            env.close()

    def test_import_order_is_cycle_free(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for code in (
            "import crossmaze; import data.registry; assert callable(crossmaze.make)",
            "import data.registry; import crossmaze; assert callable(crossmaze.make)",
            "import crossmaze.layout; import data.registry",
            "from crossmaze import make; import data.registry; assert callable(make)",
        ):
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode, 0, msg=f"{code!r} failed: {result.stderr}"
            )


class MapUnificationTest(unittest.TestCase):
    """AntMaze eval uses collection maps + coordinate start/goal, no r/g maps."""

    def test_no_marker_cells_anywhere_in_variant_registries(self):
        for family_variants in (POINTMAZE_VARIANTS, ANTMAZE_VARIANTS):
            for variant, meta in family_variants.items():
                maps = [meta["prompt_vars"]["maze_map"]]
                if "env_paras" in meta:
                    maps.append(meta["env_paras"]["maze_map"])
                if "env_kwargs" in meta:
                    maps.append(meta["env_kwargs"]["maze_map"])
                for maze_map in maps:
                    cells = {cell for row in maze_map for cell in row}
                    self.assertTrue(
                        cells <= {0, 1}, msg=f"{variant}: non-binary cells {cells}"
                    )

    def test_antmaze_eval_maps_match_collection_maps(self):
        for variant, meta in ANTMAZE_VARIANTS.items():
            env_map = (
                meta["env_kwargs"]["maze_map"]
                if "env_kwargs" in meta
                else meta["env_paras"]["maze_map"]
            )
            self.assertEqual(env_map, meta["prompt_vars"]["maze_map"], msg=variant)

    def test_antmaze_eval_cells_are_free_and_official_mapping_holds(self):
        from crossmaze.eval_position import EVAL_POSITIONS

        for variant, meta in ANTMAZE_VARIANTS.items():
            maze_map = meta["prompt_vars"]["maze_map"]
            position_config = EVAL_POSITIONS["antmaze"][variant]
            self.assertIn("fix_start_goal", position_config, msg=variant)
            self.assertEqual(
                position_config["fix_start_goal"]["start_cell"],
                meta["eval_reset_cell"],
                msg=variant,
            )
            self.assertEqual(
                position_config["fix_start_goal"]["goal_cell"],
                meta["eval_goal_cell"],
                msg=variant,
            )
            self.assertIsInstance(
                position_config["fix_start_goal"]["difficulty"],
                float,
                msg=variant,
            )
            for key in ("eval_reset_cell", "eval_goal_cell"):
                row, col = meta[key]
                self.assertEqual(
                    maze_map[row][col], 0, msg=f"{variant} {key} is not free"
                )
        for variant in ("umaze", "umaze-diverse"):
            self.assertEqual(ANTMAZE_VARIANTS[variant]["eval_reset_cell"], [1, 1])
            self.assertEqual(ANTMAZE_VARIANTS[variant]["eval_goal_cell"], [3, 1])
        for variant in ("medium-play", "medium-diverse"):
            self.assertEqual(ANTMAZE_VARIANTS[variant]["eval_reset_cell"], [1, 1])
            self.assertEqual(ANTMAZE_VARIANTS[variant]["eval_goal_cell"], [6, 6])
        for variant in ("large-play", "large-diverse"):
            self.assertEqual(ANTMAZE_VARIANTS[variant]["eval_reset_cell"], [1, 1])
            self.assertEqual(ANTMAZE_VARIANTS[variant]["eval_goal_cell"], [7, 9])

    def test_repo_variants_derive_from_crossmaze_env_facts(self):
        from crossmaze.variants import ANTMAZE_ENV_FACTS, POINTMAZE_ENV_FACTS
        from crossmaze.eval_position import eval_reset_options

        self.assertEqual(set(POINTMAZE_ENV_FACTS), set(POINTMAZE_VARIANTS))
        self.assertEqual(set(ANTMAZE_ENV_FACTS), set(ANTMAZE_VARIANTS))
        for variant, facts in POINTMAZE_ENV_FACTS.items():
            meta = POINTMAZE_VARIANTS[variant]
            self.assertEqual(facts["maze_map"], meta["prompt_vars"]["maze_map"])
            if facts["kind"] == "local":
                self.assertEqual(facts["env_paras"], meta["env_paras"])
        for variant, facts in ANTMAZE_ENV_FACTS.items():
            meta = ANTMAZE_VARIANTS[variant]
            self.assertEqual(facts["maze_map"], meta["prompt_vars"]["maze_map"])
            self.assertNotIn("eval_reset_cell", facts)
            self.assertNotIn("eval_goal_cell", facts)
            reset_options = eval_reset_options("antmaze", variant)
            self.assertEqual(reset_options["reset_cell"], meta["eval_reset_cell"])
            self.assertEqual(reset_options["goal_cell"], meta["eval_goal_cell"])

    def test_pointmaze_eval_position_tables_use_top_hard_pairs(self):
        from crossmaze.eval_position import (
            HARD_SAMPLE_MODE,
            build_eval_start_goal_pair_space,
            get_eval_position_config,
            _hard_sample_pool_size,
        )
        from crossmaze.variants import POINTMAZE_ENV_FACTS

        seed = 17
        hard_config = {
            "eval_start_goal_mode": HARD_SAMPLE_MODE,
            "eval_hard_sample_top_percent": 0.2,
            "eval_hard_sample_alpha": 0.0,
        }
        for variant, facts in POINTMAZE_ENV_FACTS.items():
            with self.subTest(variant=variant):
                table = get_eval_position_config(
                    "pointmaze",
                    variant,
                    seed=seed,
                    config=hard_config,
                )["start_goal_list"]
                candidate_cells = [
                    (row_idx, col_idx)
                    for row_idx, row in enumerate(facts["maze_map"])
                    for col_idx, cell in enumerate(row)
                    if cell != 1
                ]
                pair_space, _total = build_eval_start_goal_pair_space(
                    facts["maze_map"],
                    candidate_cells,
                    hard_sample_alpha=0.0,
                )
                hard_pool = pair_space[
                    -_hard_sample_pool_size(
                        len(pair_space),
                        top_percent=0.2,
                        top_n=None,
                    ):
                ]
                hard_by_pair = {
                    (record["start_cell"], record["goal_cell"]): record
                    for record in hard_pool
                }

                self.assertEqual(len(table), min(100, len(hard_pool)))
                for record in table:
                    start_cell = tuple(record["start_cell"])
                    goal_cell = tuple(record["goal_cell"])
                    self.assertIn((start_cell, goal_cell), hard_by_pair)
                    self.assertEqual(facts["maze_map"][start_cell[0]][start_cell[1]], 0)
                    self.assertEqual(facts["maze_map"][goal_cell[0]][goal_cell[1]], 0)
                    self.assertAlmostEqual(
                        record["difficulty"],
                        hard_by_pair[(start_cell, goal_cell)]["difficulty"],
                    )

    def test_hard_sample_builds_only_the_requested_variant(self):
        from unittest.mock import patch

        import crossmaze.eval_position as eval_position
        from crossmaze.variants import POINTMAZE_ENV_FACTS

        hard_config = {
            "eval_start_goal_mode": eval_position.HARD_SAMPLE_MODE,
            "eval_hard_sample_top_percent": 0.2,
        }
        variant = "local-layoutV2-12"
        requested_map = POINTMAZE_ENV_FACTS[variant]["maze_map"]
        original_builder = eval_position.build_eval_start_goal_pair_space
        built_maps = []

        def recording_builder(maze_map, *args, **kwargs):
            built_maps.append(maze_map)
            return original_builder(maze_map, *args, **kwargs)

        eval_position._hard_sample_eval_position_for_settings.cache_clear()
        eval_position._eval_pair_space_for_variant.cache_clear()
        try:
            with patch.object(
                eval_position,
                "build_eval_start_goal_pair_space",
                side_effect=recording_builder,
            ):
                first = eval_position.get_eval_position_config(
                    "pointmaze",
                    variant,
                    seed=17,
                    config=hard_config,
                )
                second = eval_position.get_eval_position_config(
                    "pointmaze",
                    variant,
                    seed=17,
                    config=hard_config,
                )

            self.assertEqual(first, second)
            self.assertEqual(len(built_maps), 1)
            self.assertIs(built_maps[0], requested_map)
        finally:
            eval_position._hard_sample_eval_position_for_settings.cache_clear()
            eval_position._eval_pair_space_for_variant.cache_clear()

    def test_hard_sample_without_episode_index_skips_pair_building(self):
        from unittest.mock import patch

        import crossmaze.eval_position as eval_position

        hard_config = {
            "eval_start_goal_mode": eval_position.HARD_SAMPLE_MODE,
            "eval_hard_sample_top_percent": 0.2,
        }
        eval_position._hard_sample_eval_position_for_settings.cache_clear()
        eval_position._eval_pair_space_for_variant.cache_clear()
        with patch.object(
            eval_position,
            "build_eval_start_goal_pair_space",
            side_effect=AssertionError("hard pair table should not be built"),
        ):
            self.assertIsNone(
                eval_position.select_eval_position(
                    "pointmaze",
                    "local-layoutV2-12",
                    episode_index=None,
                    seed=17,
                    config=hard_config,
                )
            )
        with self.assertRaises(ValueError):
            eval_position.select_eval_position(
                "pointmaze",
                "local-layoutV2-12",
                episode_index=None,
                seed=17,
                config={"eval_start_goal_mode": eval_position.HARD_SAMPLE_MODE},
            )

    def test_path_difficulty_ignores_equivalent_shortest_route_as_distractor(self):
        from crossmaze.eval_position import build_eval_start_goal_pair_space

        maze_map = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ]
        cells = [(1, 1), (1, 2), (2, 1), (2, 2)]
        records, _ = build_eval_start_goal_pair_space(
            maze_map,
            cells,
            hard_sample_alpha=0.0,
        )
        diagonal = next(
            record
            for record in records
            if record["start_cell"] == (1, 1)
            and record["goal_cell"] == (2, 2)
        )

        self.assertEqual(diagonal["path_len"], 2)
        self.assertEqual(diagonal["distractor_point_count"], 0)
        self.assertEqual(diagonal["distractor_exit_count"], 0)
        self.assertEqual(diagonal["branch_score"], 0.0)

        top_count = max(1, math.ceil(len(records) * 0.1))
        expected_map_difficulty = sum(
            record["difficulty"] for record in records[-top_count:]
        ) / top_count
        self.assertEqual(diagonal["map_difficulty_path_count"], top_count)
        self.assertAlmostEqual(
            diagonal["map_difficulty"],
            expected_map_difficulty,
        )

    def test_hard_sample_pool_payload_contains_map_and_path_difficulty(self):
        from crossmaze.eval_position import (
            HARD_SAMPLE_MODE,
            get_eval_position_pool_payload,
        )

        payload = get_eval_position_pool_payload(
            "pointmaze",
            "umaze",
            seed=17,
            config={
                "eval_start_goal_mode": HARD_SAMPLE_MODE,
                "eval_hard_sample_top_n": 5,
            },
        )

        self.assertEqual(payload["difficulty_version"], "v2")
        self.assertEqual(payload["difficulty_config"]["length_scale"], 20.0)
        self.assertEqual(payload["map_difficulty_top_fraction"], 0.1)
        self.assertEqual(payload["selected_pair_count"], 5)
        self.assertEqual(len(payload["start_goal_list"]), 5)
        components = payload["start_goal_list"][0]["difficulty_components"]
        for key in (
            "length_score",
            "branch_score",
            "detour_score",
            "distractor_point_count",
            "distractor_exit_count",
        ):
            self.assertIn(key, components)

    def test_eval_position_selection_is_seeded_permutation_cycle(self):
        from crossmaze.eval_position import (
            HARD_SAMPLE_MODE,
            get_eval_position_config,
            select_eval_position,
        )

        variant = "umaze"
        hard_config = {
            "eval_start_goal_mode": HARD_SAMPLE_MODE,
            "eval_hard_sample_top_percent": 0.2,
        }
        table = get_eval_position_config(
            "pointmaze",
            variant,
            seed=17,
            config=hard_config,
        )["start_goal_list"]
        count = len(table)
        seed = 17
        first_cycle = [
            select_eval_position(
                "pointmaze",
                variant,
                idx,
                seed,
                config=hard_config,
            )["index"]
            for idx in range(count)
        ]
        second_read = [
            select_eval_position(
                "pointmaze",
                variant,
                idx,
                seed,
                config=hard_config,
            )["index"]
            for idx in range(count)
        ]

        self.assertEqual(first_cycle, second_read)
        self.assertEqual(len(set(first_cycle)), count)
        for idx in range(count):
            self.assertEqual(
                select_eval_position(
                    "pointmaze",
                    variant,
                    idx + count,
                    seed,
                    config=hard_config,
                )["index"],
                first_cycle[idx],
            )

        fixed = select_eval_position("antmaze", "umaze", 999, seed)
        self.assertEqual(fixed["source"], "fix_start_goal")
        self.assertEqual(fixed["index"], 0)

    def test_pointmaze_eval_seed_controls_hard_pair_table(self):
        from crossmaze.eval_position import HARD_SAMPLE_MODE, get_eval_position_config
        from crossmaze.variants import POINTMAZE_ENV_FACTS

        hard_config = {
            "eval_start_goal_mode": HARD_SAMPLE_MODE,
            "eval_hard_sample_top_percent": 0.2,
        }
        changed_variants = []
        for variant in POINTMAZE_ENV_FACTS:
            table_a = get_eval_position_config(
                "pointmaze",
                variant,
                seed=17,
                config=hard_config,
            )["start_goal_list"]
            table_b = get_eval_position_config(
                "pointmaze",
                variant,
                seed=17,
                config=hard_config,
            )["start_goal_list"]
            table_c = get_eval_position_config(
                "pointmaze",
                variant,
                seed=18,
                config=hard_config,
            )["start_goal_list"]
            self.assertEqual(table_a, table_b, msg=variant)
            if table_a != table_c:
                changed_variants.append(variant)

        self.assertTrue(changed_variants)

    def test_eval_position_modes_and_validation(self):
        from crossmaze.eval_position import (
            FIX_START_GOAL_MODE,
            HARD_SAMPLE_MODE,
            RANDOM_START_GOAL_MODE,
            eval_position_count,
            eval_position_selection_policy,
            get_eval_position_config,
            resolve_eval_position_mode,
            select_eval_position,
        )

        self.assertEqual(resolve_eval_position_mode("antmaze", {}), FIX_START_GOAL_MODE)
        self.assertEqual(
            resolve_eval_position_mode("pointmaze", {}),
            RANDOM_START_GOAL_MODE,
        )
        self.assertIsNone(get_eval_position_config("pointmaze", "umaze"))
        self.assertIsNone(select_eval_position("pointmaze", "umaze", 0, 1))
        self.assertEqual(eval_position_count("pointmaze", "umaze"), 0)
        self.assertEqual(
            eval_position_selection_policy("pointmaze", "umaze"),
            "env_default_random",
        )

        hard_top_n_config = {
            "eval_start_goal_mode": HARD_SAMPLE_MODE,
            "eval_hard_sample_top_n": 5,
            "eval_hard_sample_alpha": 2.0,
        }
        table = get_eval_position_config(
            "pointmaze",
            "umaze",
            seed=3,
            config=hard_top_n_config,
        )["start_goal_list"]
        self.assertEqual(len(table), 5)
        record = select_eval_position(
            "pointmaze",
            "umaze",
            0,
            3,
            config=hard_top_n_config,
        )
        self.assertEqual(record["source"], "hard_sample")

        with self.assertRaises(ValueError):
            get_eval_position_config(
                "pointmaze",
                "umaze",
                config={"eval_start_goal_mode": FIX_START_GOAL_MODE},
            )
        with self.assertRaises(ValueError):
            get_eval_position_config(
                "pointmaze",
                "umaze",
                config={"eval_start_goal_mode": HARD_SAMPLE_MODE},
            )
        with self.assertRaises(ValueError):
            get_eval_position_config(
                "pointmaze",
                "umaze",
                config={
                    "eval_start_goal_mode": HARD_SAMPLE_MODE,
                    "eval_hard_sample_top_percent": 0.2,
                    "eval_hard_sample_top_n": 10,
                },
            )

    def test_antmaze_eval_reset_places_start_and_goal_at_recorded_cells(self):
        import crossmaze

        meta = ANTMAZE_VARIANTS["umaze"]
        env = crossmaze.make("antmaze", "umaze", mode="eval", config={})
        try:
            self.assertEqual(
                env.default_reset_options,
                {
                    "reset_cell": meta["eval_reset_cell"],
                    "goal_cell": meta["eval_goal_cell"],
                },
            )
            for seed in (0, 5):
                obs, _info = env.reset(seed=seed)
                state = obs[crossmaze.CROSSMAZE_OBS_KEY]
                self.assertEqual(state["position_cell"], meta["eval_reset_cell"])
                self.assertEqual(state["goal_cell"], meta["eval_goal_cell"])

            obs_a, _ = env.reset(seed=42)
            obs_b, _ = env.reset(seed=42)
            np.testing.assert_allclose(obs_a["achieved_goal"], obs_b["achieved_goal"])
            np.testing.assert_allclose(obs_a["desired_goal"], obs_b["desired_goal"])

            override = {
                "reset_cell": np.asarray([3, 3], dtype=np.int64),
                "goal_cell": np.asarray([1, 3], dtype=np.int64),
            }
            obs_c, _ = env.reset(seed=1, options=override)
            state_c = obs_c[crossmaze.CROSSMAZE_OBS_KEY]
            self.assertEqual(state_c["position_cell"], [3, 3])
            self.assertEqual(state_c["goal_cell"], [1, 3])
        finally:
            env.close()

    def test_crossmaze_package_is_standalone(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        code = (
            "import sys\n"
            "import crossmaze\n"
            "from crossmaze.variants import eval_reset_options\n"
            "from crossmaze.eval_position import select_eval_position\n"
            "from crossmaze.score import build_pointmaze_score_env_spec\n"
            "assert eval_reset_options('antmaze', 'umaze') == "
            "{'reset_cell': [1, 1], 'goal_cell': [3, 1]}\n"
            "assert eval_reset_options('pointmaze', 'umaze') is None\n"
            "assert select_eval_position('pointmaze', 'umaze', 0, 1) is None\n"
            "hard_config = {'eval_start_goal_mode': 'hard-sample', "
            "'eval_hard_sample_top_percent': 0.2}\n"
            "assert select_eval_position('pointmaze', 'umaze', 0, 1, "
            "config=hard_config)['source'] == 'hard_sample'\n"
            "env = crossmaze.make('pointmaze', 'umaze', mode='eval', config={})\n"
            "obs, _ = env.reset(seed=0)\n"
            "assert crossmaze.CROSSMAZE_OBS_KEY in obs\n"
            "env.close()\n"
            "spec = build_pointmaze_score_env_spec('umaze', {})\n"
            "assert spec.env_fingerprint\n"
            "bad = [m for m in sys.modules"
            " if m in ('data', 'utils') or m.startswith(('data.', 'utils.'))]\n"
            "assert not bad, bad\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
