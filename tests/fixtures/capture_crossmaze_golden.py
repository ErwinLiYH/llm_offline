"""Capture golden fixtures for the CrossMaze extraction refactor (run BEFORE any edit).

Outputs tests/fixtures/crossmaze_sensing_golden.json with:
- build_sensing outputs across families x versions x thresholds x positions
- format_obs payloads
- full render_policy_prompt strings (history 0 and 2)
- cache signature payload hashes for fixed text-mode requests
"""
import json
import sys
import numpy as np

sys.path.insert(0, "/home/u6mx/yl1118.u6mx/llm_offline")

from utils.maze_sensing import build_sensing, _cell_bounds, _cell_center_xy  # noqa: E402
from utils.sensing_config import apply_sensing_config_to_prompt_vars  # noqa: E402
from utils.prompt_loader import load_named_templates  # noqa: E402
from utils.eval_rollout import render_policy_prompt  # noqa: E402
from data.pointmaze.variants import POINTMAZE_VARIANTS  # noqa: E402
from data.antmaze.variants import ANTMAZE_VARIANTS  # noqa: E402
from data.pointmaze import formatting as pm_fmt  # noqa: E402
from data.antmaze import formatting as am_fmt  # noqa: E402
from data.pointmaze.dataset import PointMazeDataset, DatasetBuildRequest  # noqa: E402
from data.antmaze.dataset import AntMazeDataset  # noqa: E402

FAMILIES = {
    "pointmaze": {
        "variants": ["medium", "local-EXturn"],
        "fmt": pm_fmt,
        "variant_registry": POINTMAZE_VARIANTS,
    },
    "antmaze": {
        "variants": ["medium-play"],
        "fmt": am_fmt,
        "variant_registry": ANTMAZE_VARIANTS,
    },
}
VERSIONS = ["v1", "v2", "v3", "v4", "v5"]
THRESHOLDS = [0.10, 0.25]


def battery_positions(maze_map, scaling, threshold):
    """Deterministic battery of xy points: centers, near-boundary (in/at/out of
    threshold*scaling), wall-snap points, map-edge clip points."""
    rows, cols = len(maze_map), len(maze_map[0])
    free = [(r, c) for r in range(rows) for c in range(cols) if maze_map[r][c] != 1]
    walls = [(r, c) for r in range(rows) for c in range(cols) if maze_map[r][c] == 1]
    pts = []
    thr = threshold * scaling
    # sample up to 8 spread free cells deterministically
    step = max(1, len(free) // 8)
    sample = free[::step][:8]
    for r, c in sample:
        cx, cy = _cell_center_xy(r, c, rows, cols, scaling)
        left_x, right_x, bottom_y, top_y = _cell_bounds(r, c, rows, cols, scaling)
        pts.append(("center", cx, cy))
        for label, x, y in (
            ("near_left_in", left_x + 0.5 * thr, cy),
            ("near_left_at", left_x + thr, cy),
            ("near_left_out", left_x + 1.5 * thr, cy),
            ("near_right_in", right_x - 0.5 * thr, cy),
            ("near_top_in", cx, top_y - 0.5 * thr),
            ("near_bottom_in", cx, bottom_y + 0.5 * thr),
            ("corner_in", left_x + 0.5 * thr, top_y - 0.5 * thr),
        ):
            pts.append((label, x, y))
    # wall-snap: centers of a couple of wall cells (interior ones)
    for r, c in walls[1: len(walls): max(1, len(walls) // 3)][:3]:
        cx, cy = _cell_center_xy(r, c, rows, cols, scaling)
        pts.append(("wall_snap", cx, cy))
    # outside-map clip
    pts.append(("clip_out", -(cols) * scaling, (rows) * scaling))
    return pts


def goal_points(maze_map, scaling):
    rows, cols = len(maze_map), len(maze_map[0])
    free = [(r, c) for r in range(rows) for c in range(cols) if maze_map[r][c] != 1]
    g1 = _cell_center_xy(*free[len(free) // 2], rows, cols, scaling)
    walls = [(r, c) for r in range(rows) for c in range(cols) if maze_map[r][c] == 1]
    g2 = _cell_center_xy(*walls[len(walls) // 2], rows, cols, scaling)  # snap-exercising goal
    return [("goal_free", *g1), ("goal_wall_snap", *g2)]


def make_obs(family, x, y, gx, gy):
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


def history_entry(family, fmt, x, y, k):
    obs = make_obs(family, x, y, 0.0, 0.0)
    stored = fmt.format_history_observation(obs)
    if family == "pointmaze":
        action_text = f"{10 + k},-{20 + k}"
    else:
        action_text = ",".join(str((7 * k + i) % 100 - 50) for i in range(8))
    return {"observation": stored, "action_text": action_text}


def main():
    golden = {"build_sensing": [], "format_obs": [], "prompts": [], "cache_signatures": {}}

    for family, spec in FAMILIES.items():
        fmt = spec["fmt"]
        for variant in spec["variants"]:
            pv = spec["variant_registry"][variant]["prompt_vars"]
            maze_map = pv["maze_map"]
            scaling = float(pv.get("maze_size_scaling", 1.0))
            goals = goal_points(maze_map, scaling)
            for version in VERSIONS:
                for threshold in THRESHOLDS:
                    meta = dict(pv)
                    meta["wall_sensing_version"] = version
                    meta["map_sensing_boundary_risk_threshold"] = threshold
                    for plabel, x, y in battery_positions(maze_map, scaling, threshold):
                        glabel, gx, gy = goals[0] if plabel != "center" else goals[1]
                        out = build_sensing(
                            np.array([x, y], dtype=np.float64),
                            np.array([gx, gy], dtype=np.float64),
                            meta,
                        )
                        golden["build_sensing"].append(
                            {
                                "family": family,
                                "variant": variant,
                                "version": version,
                                "threshold": threshold,
                                "pos_label": plabel,
                                "x": x,
                                "y": y,
                                "gx": gx,
                                "gy": gy,
                                "output": out,
                            }
                        )

            # format_obs payloads (default sensing + v5) on a subset of points
            for version in ("v3", "v5"):
                meta = dict(pv)
                meta["wall_sensing_version"] = version
                meta["map_sensing_boundary_risk_threshold"] = 0.10
                for plabel, x, y in battery_positions(maze_map, scaling, 0.10)[:12]:
                    glabel, gx, gy = goals[0]
                    obs = make_obs(family, x, y, gx, gy)
                    payload = fmt.format_obs(obs, meta)
                    golden["format_obs"].append(
                        {
                            "family": family,
                            "variant": variant,
                            "version": version,
                            "pos_label": plabel,
                            "x": x,
                            "y": y,
                            "gx": gx,
                            "gy": gy,
                            "payload": payload,
                        }
                    )

            # full prompts, history 0 and 2
            template = load_named_templates(family, ["parallel_full_sensing"])[0]
            for version in ("v3", "v5"):
                prompt_vars = apply_sensing_config_to_prompt_vars(
                    pv,
                    {
                        "wall_sensing_version": version,
                        "map_sensing_boundary_risk_threshold": 0.10,
                    },
                )
                rows, cols = len(maze_map), len(maze_map[0])
                free = [
                    (r, c)
                    for r in range(rows)
                    for c in range(cols)
                    if maze_map[r][c] != 1
                ]
                px, py = _cell_center_xy(*free[0], rows, cols, scaling)
                gx, gy = goal_points(maze_map, scaling)[0][1:]
                obs = make_obs(family, px + 0.03 * scaling, py - 0.04 * scaling, gx, gy)
                for history_num in (0, 2):
                    buffer = [
                        history_entry(family, fmt, px + 0.01 * k * scaling, py, k)
                        for k in range(3)
                    ]
                    prompt = render_policy_prompt(
                        formatter=fmt,
                        template=template,
                        prompt_vars=prompt_vars,
                        obs=obs,
                        history_buffer=buffer,
                        history_num=history_num,
                        history_stride=1,
                    )
                    golden["prompts"].append(
                        {
                            "family": family,
                            "variant": variant,
                            "version": version,
                            "history_num": history_num,
                            "prompt": prompt,
                        }
                    )

    # cache signature hashes (text mode, fixed tokenizer path, no tokenizer object)
    for name, dataset_cls, variant, action_dim in (
        ("pointmaze_open_text", PointMazeDataset, "open", 2),
        ("antmaze_umaze_text", AntMazeDataset, "umaze", 8),
    ):
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
        golden["cache_signatures"][name] = dataset_cls._cache_signature_hash(config)

    out_path = "/home/u6mx/yl1118.u6mx/llm_offline/tests/fixtures/crossmaze_sensing_golden.json"
    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(golden, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(
        f"wrote {out_path}: build_sensing={len(golden['build_sensing'])} "
        f"format_obs={len(golden['format_obs'])} prompts={len(golden['prompts'])} "
        f"cache_signatures={golden['cache_signatures']}"
    )


if __name__ == "__main__":
    main()
