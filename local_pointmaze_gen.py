"""Generate local PointMaze Minari datasets through Farama's official scripts."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import os
import shutil
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  registers PointMaze envs
import minari
import numpy as np
from minari import DataCollector, MinariDataset, StepDataCallback
from minari.storage.local import get_dataset_path

from data.pointmaze.variants import (
    POINTMAZE_VARIANTS,
    get_pointmaze_variant_type,
    resolve_local_dataset_path,
)

OFFICIAL_POINTMAZE_DIR = (
    Path(__file__).resolve().parent
    / "third_party"
    / "minari-dataset-generation-scripts"
    / "scripts"
    / "pointmaze"
)


def _load_official_pointmaze_modules():
    if not OFFICIAL_POINTMAZE_DIR.exists():
        raise RuntimeError(
            "Official Farama PointMaze generator submodule is missing. "
            "Run: git submodule update --init --recursive"
        )
    official_path = str(OFFICIAL_POINTMAZE_DIR)
    if official_path not in sys.path:
        sys.path.insert(0, official_path)
    controller_module = importlib.import_module("controller")
    return controller_module


_OFFICIAL_CONTROLLER = _load_official_pointmaze_modules()
WaypointController = _OFFICIAL_CONTROLLER.WaypointController


class PointMazeStepDataCallback(StepDataCallback):
    """Record PointMaze state and optionally split episodes at first success."""

    truncate_on_success = True

    def __call__(
        self,
        env,
        obs,
        info,
        action=None,
        rew=None,
        terminated=None,
        truncated=None,
    ):
        step_data = super().__call__(env, obs, info, action, rew, terminated, truncated)
        info_key = "info" if "info" in step_data else "infos"
        truncated_key = "truncated" if "truncated" in step_data else "truncations"
        if self.truncate_on_success and step_data[info_key].get("success", False):
            step_data[truncated_key] = True
        step_data[info_key]["qpos"] = obs["observation"][:2]
        step_data[info_key]["qvel"] = obs["observation"][2:]
        step_data[info_key]["goal"] = obs["desired_goal"]
        return step_data


class PointMazeHoldStepDataCallback(PointMazeStepDataCallback):
    """PointMaze callback that lets the collector keep recording after success."""

    truncate_on_success = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--target-episodes", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=1000000)
    parser.add_argument(
        "--post-success-hold-steps",
        type=int,
        default=0,
        help="Extra steps to record after first reaching the goal in each local episode.",
    )
    parser.add_argument(
        "--post-success-hold-noise-std",
        type=float,
        default=0.0,
        help="Gaussian action noise std used only during the post-success hold phase.",
    )
    return parser.parse_args()


def _existing_episode_count(dataset_root: Path) -> int:
    data_path = dataset_root / "data"
    if not data_path.exists():
        return 0
    return int(MinariDataset(data_path).total_episodes)


def _make_env(env_paras: dict, max_episode_steps: int):
    env_paras = dict(env_paras)
    env_id = env_paras.pop("id")
    env_paras["max_episode_steps"] = max_episode_steps
    return gym.make(env_id, **env_paras)


def _official_requirements() -> list[str]:
    requirements_path = OFFICIAL_POINTMAZE_DIR / "requirements.txt"
    return [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _minari_version_specifier() -> str:
    version_parts = minari.__version__.split(".")
    if len(version_parts) < 2:
        return f"~={minari.__version__}"
    return f"~={version_parts[0]}.{version_parts[1]}"


def _official_description(env_id: str) -> str:
    description_path = OFFICIAL_POINTMAZE_DIR / "description.md"
    return description_path.read_text(encoding="utf-8").format(env_id=env_id)


def _create_dataset(collector: DataCollector, **kwargs):
    supported_keys = inspect.signature(collector.create_dataset).parameters
    return collector.create_dataset(
        **{key: value for key, value in kwargs.items() if key in supported_keys}
    )


def _clip_action(action: np.ndarray) -> np.ndarray:
    return np.clip(action, -1, 1).astype(np.float32)


def _waypoint_action(controller: WaypointController, obs: dict) -> np.ndarray:
    action = controller.compute_action(obs)
    action += np.random.randn(*action.shape) * 0.5
    return _clip_action(action)


def _hold_action(obs: dict, noise_std: float) -> np.ndarray:
    action = 10.0 * (obs["desired_goal"] - obs["achieved_goal"]) - obs["observation"][2:]
    if noise_std > 0:
        action += np.random.randn(*action.shape) * noise_std
    return _clip_action(action)


def _collect_shard(
    *,
    variant: str,
    env_paras: dict,
    target_episodes: int,
    seed: int,
    worker_index: int,
    max_episode_steps: int,
    post_success_hold_steps: int,
    post_success_hold_noise_std: float,
) -> tuple[str, str]:
    # Keep shard IDs namespace-free so Minari does not recursively scan the
    # shared dataset root while parallel workers create/delete temporary dirs.
    dataset_id = f"pointmaze-{variant}-shard-{uuid.uuid4().hex[:12]}-v0"
    collect_env_paras = dict(env_paras)
    if post_success_hold_steps > 0:
        collect_env_paras["reset_target"] = False
    env = _make_env(collect_env_paras, max_episode_steps=max_episode_steps)
    collector = DataCollector(
        env,
        step_data_callback=(
            PointMazeHoldStepDataCallback
            if post_success_hold_steps > 0
            else PointMazeStepDataCallback
        ),
        record_infos=True,
    )
    np.random.seed(seed)
    controller = WaypointController(maze=env.unwrapped.maze, maze_solver="QIteration")
    obs, _ = collector.reset(seed=seed)
    steps = 0
    episodes = 0
    successful_episodes = 0
    holding = False
    hold_steps_remaining = 0

    while episodes < target_episodes:
        action = (
            _hold_action(obs, post_success_hold_noise_std)
            if holding
            else _waypoint_action(controller, obs)
        )
        obs, _, terminated, truncated, info = collector.step(action)
        steps += 1

        if post_success_hold_steps <= 0:
            if info.get("success", False):
                episodes += 1
                successful_episodes += 1
            elif terminated or truncated:
                obs, _ = collector.reset()
            continue

        if terminated or truncated:
            if holding or info.get("success", False):
                episodes += 1
                successful_episodes += 1
            obs, _ = collector.reset()
            holding = False
            hold_steps_remaining = 0
            continue

        if holding:
            hold_steps_remaining -= 1
            if hold_steps_remaining <= 0:
                episodes += 1
                successful_episodes += 1
                obs, _ = collector.reset()
                holding = False
                hold_steps_remaining = 0
            continue

        if info.get("success", False):
            holding = True
            hold_steps_remaining = post_success_hold_steps

    eval_env_paras = dict(env_paras)
    eval_env_id = eval_env_paras.pop("id")
    eval_env_paras["max_episode_steps"] = max_episode_steps
    eval_env_paras["continuing_task"] = True
    eval_env_paras["reset_target"] = False
    eval_env = gym.make(eval_env_id, **eval_env_paras)
    eval_controller = WaypointController(maze=eval_env.unwrapped.maze, maze_solver="QIteration")

    _create_dataset(
        collector,
        dataset_id=dataset_id,
        eval_env=eval_env,
        expert_policy=eval_controller.compute_action,
        num_episodes_average_score=1,
        algorithm_name="QIteration",
        code_permalink="https://github.com/Farama-Foundation/minari-dataset-generation-scripts",
        author="local_pointmaze_gen.py",
        author_email="",
        description=(
            _official_description(eval_env_id)
            + f"\n\nLocal wrapper variant={variant}, worker={worker_index}, "
            f"target_episodes={target_episodes}, collected_steps={steps}, seed={seed}, "
            f"post_success_hold_steps={post_success_hold_steps}, "
            f"post_success_hold_noise_std={post_success_hold_noise_std}."
        ),
        requirements=_official_requirements(),
        minari_version=_minari_version_specifier(),
    )
    eval_env.close()
    collector.close()
    print(
        f"[local-pointmaze-gen] {variant} worker={worker_index}: shard={dataset_id}, "
        f"target_episodes={target_episodes}, successful_episodes={successful_episodes}, "
        f"collected_steps={steps}, seed={seed}, "
        f"post_success_hold_steps={post_success_hold_steps}, "
        f"post_success_hold_noise_std={post_success_hold_noise_std}."
    )
    return dataset_id, str(get_dataset_path(dataset_id))


def _collect_shard_from_kwargs(kwargs: dict) -> tuple[str, str]:
    return _collect_shard(**kwargs)


def _copy_dataset_root(src_root: Path, dst_root: Path):
    if dst_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset path: {dst_root}")
    dst_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_root, dst_root)


def _merge_shard_into_final(shard_root: Path, final_root: Path):
    if not (final_root / "data").exists():
        _copy_dataset_root(shard_root, final_root)
        return
    final_dataset = MinariDataset(final_root / "data")
    shard_dataset = MinariDataset(shard_root / "data")
    final_dataset.storage.update_from_storage(shard_dataset.storage)


def _cleanup_dataset_id(dataset_id: str):
    try:
        minari.delete_dataset(dataset_id)
    except FileNotFoundError:
        pass


def generate_variant(
    variant: str,
    *,
    num_workers: int,
    target_episodes: int,
    overwrite: bool,
    seed: int,
    max_episode_steps: int,
    post_success_hold_steps: int,
    post_success_hold_noise_std: float,
):
    if variant not in POINTMAZE_VARIANTS:
        raise ValueError(f"Unknown PointMaze variant: {variant!r}")
    meta = POINTMAZE_VARIANTS[variant]
    if get_pointmaze_variant_type(meta) != "local":
        raise ValueError(f"Variant {variant!r} is not local and cannot be generated")

    dataset_root = resolve_local_dataset_path(meta["dataset_path"])
    if post_success_hold_steps > 0 and dataset_root.exists() and not overwrite:
        raise ValueError(
            f"{variant} already has a local dataset at {dataset_root}. "
            "Use --overwrite when generating post-success hold data, so old "
            "goal-arrival-only episodes are not mixed with hold episodes."
        )
    if overwrite and dataset_root.exists():
        shutil.rmtree(dataset_root)

    existing_episodes = _existing_episode_count(dataset_root)
    if existing_episodes >= target_episodes:
        print(
            f"[local-pointmaze-gen] {variant}: existing_episodes={existing_episodes} "
            f">= target_episodes={target_episodes}; skipping"
        )
        return

    deficit = target_episodes - existing_episodes
    worker_count = max(1, min(num_workers, deficit))
    shard_targets = [deficit // worker_count] * worker_count
    for idx in range(deficit % worker_count):
        shard_targets[idx] += 1

    print(
        f"[local-pointmaze-gen] {variant}: existing_episodes={existing_episodes}, "
        f"target_episodes={target_episodes}, deficit={deficit}, workers={worker_count}, "
        f"post_success_hold_steps={post_success_hold_steps}, "
        f"post_success_hold_noise_std={post_success_hold_noise_std}"
    )
    shard_specs = []
    variant_seed_offset = int.from_bytes(hashlib.sha256(variant.encode("utf-8")).digest()[:4], "big")
    for worker_index, shard_target in enumerate(shard_targets):
        shard_seed = seed + 1009 * worker_index + variant_seed_offset
        shard_specs.append(
            {
                "variant": variant,
                "env_paras": meta["env_paras"],
                "target_episodes": shard_target,
                "seed": shard_seed,
                "worker_index": worker_index,
                "max_episode_steps": max_episode_steps,
                "post_success_hold_steps": post_success_hold_steps,
                "post_success_hold_noise_std": post_success_hold_noise_std,
            }
        )

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        shard_results = list(executor.map(_collect_shard_from_kwargs, shard_specs))

    try:
        for dataset_id, shard_path in shard_results:
            shard_root = Path(shard_path)
            print(f"[local-pointmaze-gen] {variant}: merging shard {dataset_id}")
            _merge_shard_into_final(shard_root, dataset_root)
    finally:
        for dataset_id, _ in shard_results:
            _cleanup_dataset_id(dataset_id)

    final_episodes = _existing_episode_count(dataset_root)
    print(
        f"[local-pointmaze-gen] {variant}: final_episodes={final_episodes}, "
        f"dataset_path={dataset_root}"
    )


def main():
    args = parse_args()
    if args.target_episodes < 1:
        raise ValueError("--target-episodes must be >= 1")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.post_success_hold_steps < 0:
        raise ValueError("--post-success-hold-steps must be >= 0")
    if args.post_success_hold_noise_std < 0:
        raise ValueError("--post-success-hold-noise-std must be >= 0")
    for variant in args.variants:
        generate_variant(
            variant,
            num_workers=args.num_workers,
            target_episodes=args.target_episodes,
            overwrite=args.overwrite,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            post_success_hold_steps=args.post_success_hold_steps,
            post_success_hold_noise_std=args.post_success_hold_noise_std,
        )


if __name__ == "__main__":
    main()
