"""Behavior-preservation tests for the CrossMaze environment extraction.

The golden fixture was captured from the pre-refactor code by
`tests/fixtures/capture_crossmaze_golden.py`. Every assertion here pins
byte-identical sensing/prompt text and unchanged dataset cache signatures.
"""
import json
import os
import unittest

import numpy as np

from data.antmaze import formatting as am_fmt
from data.antmaze.dataset import AntMazeDataset
from data.pointmaze import formatting as pm_fmt
from data.pointmaze.dataset import DatasetBuildRequest, PointMazeDataset
from data.antmaze.variants import ANTMAZE_VARIANTS
from data.pointmaze.variants import POINTMAZE_VARIANTS
from utils.eval_rollout import render_policy_prompt
from utils.maze_sensing import (
    _state_matches_meta,
    build_sensing,
    compute_sensing_state,
    render_sensing_text,
)
from utils.prompt_loader import load_named_templates
from utils.sensing_config import apply_sensing_config_to_prompt_vars

GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "crossmaze_sensing_golden.json"
)

_FAMILY_FORMATTERS = {"pointmaze": pm_fmt, "antmaze": am_fmt}
_FAMILY_VARIANTS = {"pointmaze": POINTMAZE_VARIANTS, "antmaze": ANTMAZE_VARIANTS}


def _load_golden():
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


def _case_meta(case):
    prompt_vars = _FAMILY_VARIANTS[case["family"]][case["variant"]]["prompt_vars"]
    meta = dict(prompt_vars)
    meta["wall_sensing_version"] = case["version"]
    meta["map_sensing_boundary_risk_threshold"] = case.get("threshold", 0.10)
    return meta


def _make_obs(family, x, y, gx, gy):
    if family == "pointmaze":
        return {
            "observation": np.array([x, y, 0.1234, -0.5678], dtype=np.float64),
            "desired_goal": np.array([gx, gy], dtype=np.float64),
        }
    state = np.linspace(-0.9, 0.9, 27).astype(np.float64)
    return {
        "observation": state,
        "achieved_goal": np.array([x, y], dtype=np.float64),
        "desired_goal": np.array([gx, gy], dtype=np.float64),
    }


def _history_entry(family, fmt, x, y, k):
    obs = _make_obs(family, x, y, 0.0, 0.0)
    stored = fmt.format_history_observation(obs)
    if family == "pointmaze":
        action_text = f"{10 + k},-{20 + k}"
    else:
        action_text = ",".join(str((7 * k + i) % 100 - 50) for i in range(8))
    return {"observation": stored, "action_text": action_text}


class BuildSensingGoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden = _load_golden()

    def test_build_sensing_matches_golden(self):
        for case in self.golden["build_sensing"]:
            meta = _case_meta(case)
            out = build_sensing(
                np.array([case["x"], case["y"]], dtype=np.float64),
                np.array([case["gx"], case["gy"]], dtype=np.float64),
                meta,
            )
            self.assertEqual(
                out,
                case["output"],
                msg=(
                    f"{case['family']}/{case['variant']} {case['version']}"
                    f" thr={case['threshold']} {case['pos_label']}"
                    f" ({case['x']}, {case['y']})"
                ),
            )

    def test_build_sensing_is_render_of_compute(self):
        for case in self.golden["build_sensing"][::17]:
            meta = _case_meta(case)
            position = np.array([case["x"], case["y"]], dtype=np.float64)
            goal = np.array([case["gx"], case["gy"]], dtype=np.float64)
            state = compute_sensing_state(position, goal, meta)
            self.assertEqual(render_sensing_text(state), case["output"])
            # structured state stays JSON friendly for wrapper observations
            json.dumps(state)
            self.assertEqual(state["wall_sensing_version"], case["version"])
            self.assertEqual(
                state["map_sensing_boundary_risk_threshold"], case["threshold"]
            )

    def test_state_matches_meta(self):
        case = self.golden["build_sensing"][0]
        meta = _case_meta(case)
        state = compute_sensing_state(
            np.array([case["x"], case["y"]], dtype=np.float64),
            np.array([case["gx"], case["gy"]], dtype=np.float64),
            meta,
        )
        enriched = dict(state)
        enriched["maze_map"] = [list(row) for row in meta["maze_map"]]
        enriched["maze_size_scaling"] = float(meta.get("maze_size_scaling", 1.0))
        self.assertTrue(_state_matches_meta(enriched, meta))

        # any drift must force a fallback to recomputation
        self.assertFalse(_state_matches_meta(state, meta))  # no maze_map key
        other_version = dict(enriched)
        other_version["wall_sensing_version"] = (
            "v5" if state["wall_sensing_version"] != "v5" else "v4"
        )
        self.assertFalse(_state_matches_meta(other_version, meta))
        other_threshold = dict(enriched)
        other_threshold["map_sensing_boundary_risk_threshold"] = 0.42
        self.assertFalse(_state_matches_meta(other_threshold, meta))
        other_scaling = dict(enriched)
        other_scaling["maze_size_scaling"] = enriched["maze_size_scaling"] + 1.0
        self.assertFalse(_state_matches_meta(other_scaling, meta))
        other_map = dict(enriched)
        other_map["maze_map"] = [list(row) for row in meta["maze_map"]]
        other_map["maze_map"][0][0] = 0
        self.assertFalse(_state_matches_meta(other_map, meta))
        self.assertFalse(_state_matches_meta(None, meta))


class FormatObsGoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden = _load_golden()

    def test_format_obs_matches_golden(self):
        for case in self.golden["format_obs"]:
            fmt = _FAMILY_FORMATTERS[case["family"]]
            meta = _case_meta(case)
            obs = _make_obs(
                case["family"], case["x"], case["y"], case["gx"], case["gy"]
            )
            payload = fmt.format_obs(obs, meta)
            self.assertEqual(
                payload,
                case["payload"],
                msg=(
                    f"{case['family']}/{case['variant']} {case['version']}"
                    f" {case['pos_label']}"
                ),
            )


class PromptGoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden = _load_golden()

    def test_render_policy_prompt_matches_golden(self):
        from utils.maze_sensing import _cell_center_xy

        for case in self.golden["prompts"]:
            family = case["family"]
            fmt = _FAMILY_FORMATTERS[family]
            pv = _FAMILY_VARIANTS[family][case["variant"]]["prompt_vars"]
            maze_map = pv["maze_map"]
            scaling = float(pv.get("maze_size_scaling", 1.0))
            rows, cols = len(maze_map), len(maze_map[0])
            free = [
                (r, c)
                for r in range(rows)
                for c in range(cols)
                if maze_map[r][c] != 1
            ]
            template = load_named_templates(family, ["parallel_full_sensing"])[0]
            prompt_vars = apply_sensing_config_to_prompt_vars(
                pv,
                {
                    "wall_sensing_version": case["version"],
                    "map_sensing_boundary_risk_threshold": 0.10,
                },
            )
            px, py = _cell_center_xy(*free[0], rows, cols, scaling)
            g_row, g_col = free[len(free) // 2]
            gx, gy = _cell_center_xy(g_row, g_col, rows, cols, scaling)
            obs = _make_obs(family, px + 0.03 * scaling, py - 0.04 * scaling, gx, gy)
            buffer = [
                _history_entry(family, fmt, px + 0.01 * k * scaling, py, k)
                for k in range(3)
            ]
            prompt = render_policy_prompt(
                formatter=fmt,
                template=template,
                prompt_vars=prompt_vars,
                obs=obs,
                history_buffer=buffer,
                history_num=case["history_num"],
                history_stride=1,
            )
            self.assertEqual(
                prompt,
                case["prompt"],
                msg=f"{family}/{case['variant']} {case['version']}"
                f" history={case['history_num']}",
            )


class FormatterFastPathTest(unittest.TestCase):
    """format_obs must render byte-identically with and without wrapper state."""

    @classmethod
    def setUpClass(cls):
        cls.golden = _load_golden()

    def _enriched(self, case, obs, meta):
        fam_pos = {
            "pointmaze": obs["observation"].astype(np.float32),
            "antmaze": np.asarray(obs.get("achieved_goal"), dtype=np.float32)
            if case["family"] == "antmaze"
            else None,
        }[case["family"]]
        goal = np.asarray(obs["desired_goal"], dtype=np.float32)
        state = compute_sensing_state(fam_pos, goal, meta)
        enriched = dict(obs)
        enriched["crossmaze"] = {
            "maze_map": [list(row) for row in meta["maze_map"]],
            "maze_size_scaling": float(meta.get("maze_size_scaling", 1.0)),
            "maze_shape": [len(meta["maze_map"]), len(meta["maze_map"][0])],
            **state,
        }
        return enriched

    def test_enriched_obs_renders_identically(self):
        for case in self.golden["format_obs"]:
            fmt = _FAMILY_FORMATTERS[case["family"]]
            meta = _case_meta(case)
            obs = _make_obs(
                case["family"], case["x"], case["y"], case["gx"], case["gy"]
            )
            enriched = self._enriched(case, obs, meta)
            self.assertEqual(
                fmt.format_obs(enriched, meta),
                case["payload"],
                msg=f"{case['family']} {case['version']} {case['pos_label']}",
            )

    def test_mismatched_state_falls_back_to_recompute(self):
        for case in self.golden["format_obs"][:4]:
            fmt = _FAMILY_FORMATTERS[case["family"]]
            meta = _case_meta(case)
            obs = _make_obs(
                case["family"], case["x"], case["y"], case["gx"], case["gy"]
            )
            enriched = self._enriched(case, obs, meta)
            # corrupt the attached state: wrong version and wrong neighbors
            bad_state = dict(enriched["crossmaze"])
            bad_state["wall_sensing_version"] = (
                "v1" if case["version"] != "v1" else "v2"
            )
            bad_state["neighbor_status"] = {
                "up": "wall",
                "down": "wall",
                "left": "wall",
                "right": "wall",
            }
            enriched["crossmaze"] = bad_state
            self.assertEqual(fmt.format_obs(enriched, meta), case["payload"])


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
            # original obs keys must be preserved untouched
            self.assertIn("observation", obs)
            self.assertIn("desired_goal", obs)
            obs2, _r, _t, _tr, _i = env.step(env.action_space.sample())
            self.assertIn(crossmaze.CROSSMAZE_OBS_KEY, obs2)

            # sensing state must match what the formatter-side path computes
            from utils.maze_sensing import compute_sensing_state as css

            meta = dict(pv)
            meta["wall_sensing_version"] = "v5"
            expected = css(
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
        import subprocess
        import sys

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


class EndToEndPromptParityTest(unittest.TestCase):
    """Wrapper-enriched vs legacy obs must render byte-identical prompts."""

    def _run_family(self, family, variant):
        import crossmaze
        from crossmaze.layout import (
            live_env_layout_overrides,
            static_layout_from_prompt_vars,
        )
        from crossmaze.wrapper import CrossMazeEnv

        fmt = _FAMILY_FORMATTERS[family]
        base_pv = _FAMILY_VARIANTS[family][variant]["prompt_vars"]
        template = load_named_templates(family, ["parallel_full_sensing"])[0]

        outer = crossmaze.make(family, variant, mode="eval", config={})
        inner = outer.env
        try:
            if family == "antmaze":
                layout = live_env_layout_overrides(inner)
                base_pv = dict(base_pv)
                base_pv.update(layout)
            else:
                layout = static_layout_from_prompt_vars(base_pv)
            for version in ("v1", "v2", "v3", "v4", "v5"):
                sensing = {"wall_sensing_version": version}
                env = CrossMazeEnv(
                    inner, env_family=family, layout=layout, sensing_config=sensing
                )
                prompt_vars = apply_sensing_config_to_prompt_vars(base_pv, sensing)
                env.assert_meta_consistent(prompt_vars)
                obs, _info = env.reset(seed=123)
                env.action_space.seed(123)
                history_buffer = []
                for _step in range(20):
                    for history_num in (0, 2):
                        kwargs = dict(
                            formatter=fmt,
                            template=template,
                            prompt_vars=prompt_vars,
                            history_buffer=history_buffer,
                            history_num=history_num,
                            history_stride=1,
                        )
                        enriched_prompt = render_policy_prompt(obs=obs, **kwargs)
                        stripped = {
                            k: v for k, v in obs.items() if k != "crossmaze"
                        }
                        legacy_prompt = render_policy_prompt(obs=stripped, **kwargs)
                        self.assertEqual(
                            enriched_prompt,
                            legacy_prompt,
                            msg=f"{family}/{variant} {version} history={history_num}",
                        )
                    action = env.action_space.sample()
                    history_buffer.append(
                        {
                            "observation": fmt.format_history_observation(obs),
                            "action_text": fmt.format_action(action),
                        }
                    )
                    obs, _r, terminated, truncated, _i = env.step(action)
                    if terminated or truncated:
                        obs, _info = env.reset(seed=124)
                        history_buffer = []
        finally:
            outer.close()

    def test_pointmaze_prompt_parity(self):
        self._run_family("pointmaze", "umaze")

    def test_antmaze_prompt_parity(self):
        self._run_family("antmaze", "umaze")


class CacheSignatureGoldenTest(unittest.TestCase):
    # The 2026-07 map unification replaced AntMaze r/g-marked eval maps with
    # plain collection maps plus coordinate eval cells. The variant metadata
    # embedded in the cache signature changed, so AntMaze signatures shifted
    # once (tokenized content is unchanged; old caches are simply rebuilt).
    # PointMaze variant dicts are byte-identical, so its golden hash still holds.
    ANTMAZE_UMAZE_TEXT_SIGNATURE = "df5ef1221a71347000e9f5e29c780ac6"

    @classmethod
    def setUpClass(cls):
        cls.golden = _load_golden()

    def _signature_hash(self, dataset_cls, variant, action_dim):
        request = DatasetBuildRequest(
            variant=variant,
            split="train",
            tokenizer=None,
            tokenizer_name_or_path="crossmaze-golden-tokenizer",
            max_length=1024,
            prompt_templete_index=["parallel_full_sensing"],
            action_token_mode="text",
            action_dim=action_dim,
        )
        config = dataset_cls._normalize_request(request)
        return dataset_cls._cache_signature_hash(config)

    def test_pointmaze_cache_signature_hash_unchanged(self):
        self.assertEqual(
            self._signature_hash(PointMazeDataset, "open", 2),
            self.golden["cache_signatures"]["pointmaze_open_text"],
        )

    def test_antmaze_cache_signature_hash_pinned_after_map_unification(self):
        current = self._signature_hash(AntMazeDataset, "umaze", 8)
        self.assertEqual(current, self.ANTMAZE_UMAZE_TEXT_SIGNATURE)
        self.assertNotEqual(
            current,
            self.golden["cache_signatures"]["antmaze_umaze_text"],
            msg="AntMaze signature intentionally shifted with the map unification",
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
            self.assertEqual(
                env_map, meta["prompt_vars"]["maze_map"], msg=variant
            )

    def test_antmaze_eval_cells_are_free_and_official_mapping_holds(self):
        for variant, meta in ANTMAZE_VARIANTS.items():
            maze_map = meta["prompt_vars"]["maze_map"]
            for key in ("eval_reset_cell", "eval_goal_cell"):
                row, col = meta[key]
                self.assertEqual(
                    maze_map[row][col], 0, msg=f"{variant} {key} is not free"
                )
        # umaze eval cells are the mirror image of the official r=(1,3)/g=(3,3)
        for variant in ("umaze", "umaze-diverse"):
            self.assertEqual(ANTMAZE_VARIANTS[variant]["eval_reset_cell"], [1, 1])
            self.assertEqual(ANTMAZE_VARIANTS[variant]["eval_goal_cell"], [3, 1])
        # medium/large walls match the official eval maps, cells carry over
        for variant in ("medium-play", "medium-diverse"):
            self.assertEqual(ANTMAZE_VARIANTS[variant]["eval_reset_cell"], [1, 1])
            self.assertEqual(ANTMAZE_VARIANTS[variant]["eval_goal_cell"], [6, 6])
        for variant in ("large-play", "large-diverse"):
            self.assertEqual(ANTMAZE_VARIANTS[variant]["eval_reset_cell"], [1, 1])
            self.assertEqual(ANTMAZE_VARIANTS[variant]["eval_goal_cell"], [7, 9])

    def test_repo_variants_derive_from_crossmaze_env_facts(self):
        from crossmaze.variants import ANTMAZE_ENV_FACTS, POINTMAZE_ENV_FACTS

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
            self.assertEqual(list(facts["eval_reset_cell"]), meta["eval_reset_cell"])
            self.assertEqual(list(facts["eval_goal_cell"]), meta["eval_goal_cell"])

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
            # same seed reproduces the same noisy start/goal positions
            obs_a, _ = env.reset(seed=42)
            obs_b, _ = env.reset(seed=42)
            np.testing.assert_allclose(
                obs_a["achieved_goal"], obs_b["achieved_goal"]
            )
            np.testing.assert_allclose(
                obs_a["desired_goal"], obs_b["desired_goal"]
            )
            # explicit caller options still override the defaults
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
        import subprocess
        import sys

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        code = (
            "import sys\n"
            "import crossmaze\n"
            "from crossmaze.variants import ENV_FACTS, eval_env_spec, eval_reset_options\n"
            "from crossmaze.score import build_pointmaze_score_env_spec\n"
            "assert eval_reset_options('antmaze', 'umaze') == "
            "{'reset_cell': [1, 1], 'goal_cell': [3, 1]}\n"
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
