"""Public entry point building CrossMaze-wrapped PointMaze/AntMaze envs.

The gymnasium imports live inside `make()` so importing the `crossmaze`
package stays lightweight for tokenization workers. `make()` only uses
CrossMaze-internal modules (variants, layout, score), so the package works
standalone without the training repo.
"""

from crossmaze.wrapper import CrossMazeEnv


def _apply_video_env_kwargs(config: dict, env_kwargs: dict) -> dict:
    resolved = dict(env_kwargs)
    if not bool(config.get("record_video", False)):
        return resolved
    if resolved.get("render_mode") != "rgb_array":
        resolved["render_mode"] = "rgb_array"
    return resolved


def make(
    env_family: str,
    variant: str,
    *,
    mode: str = "eval",
    config: dict | None = None,
) -> CrossMazeEnv:
    """Build a CrossMaze env for eval or (PointMaze-only) official scoring.

    `config` is the eval/score run config; it supplies `reward_type`,
    `env_kwargs` overrides, `record_video` (forces `render_mode: rgb_array`),
    and the sensing keys (`wall_sensing_version`,
    `map_sensing_boundary_risk_threshold`). `reward_type` accepts `sparse` or
    `dense` and is configured only at the top level.

    Eval envs are built on the same maps the offline data was collected on.
    Variants with fixed evaluation start/goal cells (all AntMaze variants)
    apply them automatically through `reset(options=...)` unless the caller
    passes explicit reset options. Score mode (PointMaze-only) keeps the
    official goal-marked eval maps; local score envs honor reward overrides,
    while remote official variants keep their registered reward type. Its
    sensing layout is the plain variant map, not the goal-marked score map.
    """
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401 registers maze environments

    from crossmaze.families import SUPPORTED_ENV_FAMILIES
    from crossmaze.layout import live_env_layout_overrides
    from crossmaze.reward import resolve_reward_type
    from crossmaze.variants import (
        eval_env_spec,
        eval_reset_options,
        get_env_facts,
    )

    config = dict(config or {})
    if env_family not in SUPPORTED_ENV_FAMILIES:
        raise ValueError(
            f"Unsupported env_family for CrossMaze: {env_family!r}. "
            f"Supported: {list(SUPPORTED_ENV_FAMILIES)}"
        )

    facts = get_env_facts(env_family, variant)
    reset_options = None
    if mode == "score":
        if env_family != "pointmaze":
            raise ValueError("CrossMaze score mode is PointMaze-only")
        from crossmaze.score import (
            build_pointmaze_score_env_spec,
            make_pointmaze_score_env,
        )

        score_env_spec = build_pointmaze_score_env_spec(variant, config)
        env = make_pointmaze_score_env(
            score_env_spec,
            render_mode="rgb_array" if config.get("record_video", False) else None,
        )
        layout = {
            "maze_map": facts["maze_map"],
            "maze_size_scaling": facts["maze_size_scaling"],
        }
    elif mode == "eval":
        env_id, env_kwargs = eval_env_spec(env_family, variant)
        env_kwargs.update(dict(config.get("env_kwargs") or {}))
        env_kwargs["reward_type"] = resolve_reward_type(
            config,
            default=facts["reward_type"],
        )
        env_kwargs = _apply_video_env_kwargs(config, env_kwargs)
        env = gym.make(env_id, **env_kwargs)
        reset_options = eval_reset_options(env_family, variant, config=config)
        if env_family == "antmaze":
            layout = live_env_layout_overrides(env)
        else:
            layout = {
                "maze_map": facts["maze_map"],
                "maze_size_scaling": facts["maze_size_scaling"],
            }
    else:
        raise ValueError(f"Unknown CrossMaze mode: {mode!r}. Use 'eval' or 'score'.")

    return CrossMazeEnv(
        env,
        env_family=env_family,
        layout=layout,
        sensing_config=config,
        default_reset_options=reset_options,
    )
