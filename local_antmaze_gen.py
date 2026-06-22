"""Generate local AntMaze Minari datasets with the official waypoint+SAC stack."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import os
import shutil
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  registers AntMaze envs
import minari
import numpy as np
from minari import DataCollector, MinariDataset, StepDataCallback
from minari.storage.local import get_dataset_path
from stable_baselines3 import SAC

from data.antmaze.variants import (
    ANTMAZE_VARIANTS,
    get_antmaze_variant_type,
    resolve_local_dataset_path,
)


OFFICIAL_ANTMAZE_DIR = (
    Path(__file__).resolve().parent
    / "third_party"
    / "minari-dataset-generation-scripts"
    / "scripts"
    / "D4RL"
    / "antmaze"
)
OFFICIAL_SCRIPTS_DIR = OFFICIAL_ANTMAZE_DIR.parents[1]
DEFAULT_POLICY_FILE = OFFICIAL_ANTMAZE_DIR / "GoalReachAnt_model.zip"


def _load_official_antmaze_controller():
    if not OFFICIAL_ANTMAZE_DIR.exists():
        raise RuntimeError(
            "Official Farama AntMaze generator submodule is missing. "
            "Run: git submodule update --init --recursive"
        )
    scripts_path = str(OFFICIAL_SCRIPTS_DIR)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    controller_path = OFFICIAL_ANTMAZE_DIR / "controller.py"
    spec = importlib.util.spec_from_file_location(
        "official_antmaze_controller",
        controller_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Failed to load official AntMaze controller from {controller_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_OFFICIAL_CONTROLLER = _load_official_antmaze_controller()
WaypointController = _OFFICIAL_CONTROLLER.WaypointController


class AntMazeStepDataCallback(StepDataCallback):
    """Record AntMaze state and optionally split episodes at first success."""

    truncate_on_success = False

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
        success = bool(step_data[info_key].get("success", False))
        step_data[info_key] = {"success": success}
        if self.truncate_on_success and success:
            step_data[truncated_key] = True

        step_data[info_key]["qpos"] = np.concatenate(
            [obs["achieved_goal"], obs["observation"][:13]]
        )
        step_data[info_key]["qvel"] = obs["observation"][13:]
        step_data[info_key]["goal"] = obs["desired_goal"]
        return step_data


class AntMazeSuccessStepDataCallback(AntMazeStepDataCallback):
    truncate_on_success = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--target-episodes", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Override each local variant's configured AntMaze horizon.",
    )
    parser.add_argument(
        "--policy-file",
        type=Path,
        default=DEFAULT_POLICY_FILE,
        help="Stable-Baselines3 SAC goal-reaching policy zip.",
    )
    parser.add_argument(
        "--maze-solver",
        type=str,
        default="QIteration",
        choices=("QIteration", "DFS"),
        help="Discrete planner used to generate local waypoints.",
    )
    parser.add_argument(
        "--action-noise",
        type=float,
        default=0.2,
        help="Gaussian action noise std added to SAC actions before clipping.",
    )
    parser.add_argument(
        "--truncate-on-success",
        action="store_true",
        help="End a collected episode as soon as info['success'] becomes true.",
    )
    return parser.parse_args()


def _existing_episode_count(dataset_root: Path) -> int:
    data_path = dataset_root / "data"
    if not data_path.exists():
        return 0
    return int(MinariDataset(data_path).total_episodes)


def _make_env(env_paras: dict, max_episode_steps: int | None):
    env_paras = dict(env_paras)
    env_id = env_paras.pop("id")
    if max_episode_steps is not None:
        env_paras["max_episode_steps"] = int(max_episode_steps)
    return gym.make(env_id, **env_paras)


def _official_requirements() -> list[str]:
    requirements_path = OFFICIAL_ANTMAZE_DIR / "requirements.txt"
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


def _create_dataset(collector: DataCollector, **kwargs):
    supported_keys = inspect.signature(collector.create_dataset).parameters
    return collector.create_dataset(
        **{key: value for key, value in kwargs.items() if key in supported_keys}
    )


def _wrap_maze_obs(obs: dict, waypoint_xy: np.ndarray) -> np.ndarray:
    goal_direction = waypoint_xy - obs["achieved_goal"]
    return np.concatenate([obs["observation"], goal_direction])


def _clip_action(action: np.ndarray) -> np.ndarray:
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _collect_shard(
    *,
    variant: str,
    env_paras: dict,
    collection_env_paras: dict,
    target_episodes: int,
    seed: int,
    worker_index: int,
    max_episode_steps: int | None,
    policy_file: str,
    maze_solver: str,
    action_noise: float,
    truncate_on_success: bool,
) -> tuple[str, str]:
    dataset_id = f"local/antmaze-{variant}-shard-{uuid.uuid4().hex[:12]}-v0"
    env = _make_env(collection_env_paras, max_episode_steps=max_episode_steps)
    collector = DataCollector(
        env,
        step_data_callback=(
            AntMazeSuccessStepDataCallback
            if truncate_on_success
            else AntMazeStepDataCallback
        ),
        record_infos=True,
    )
    np.random.seed(seed)
    model = SAC.load(policy_file)

    def action_callback(obs: dict, waypoint_xy: np.ndarray) -> np.ndarray:
        return model.predict(_wrap_maze_obs(obs, waypoint_xy))[0]

    waypoint_controller = WaypointController(
        maze=env.unwrapped.maze,
        model_callback=action_callback,
        maze_solver=maze_solver,
    )
    obs, _ = collector.reset(seed=seed)
    reset_seed = seed
    steps = 0
    episodes = 0
    successful_episodes = 0
    episode_success = False

    while episodes < target_episodes:
        action = waypoint_controller.compute_action(obs)
        if action_noise > 0:
            action = action + np.random.randn(*action.shape) * float(action_noise)
        action = _clip_action(action)

        obs, _, terminated, truncated, info = collector.step(action)
        steps += 1
        step_success = bool(info.get("success", False))
        episode_success = episode_success or step_success
        if truncate_on_success and step_success:
            truncated = True

        if terminated or truncated:
            episodes += 1
            if episode_success:
                successful_episodes += 1
            reset_seed += 1
            obs, _ = collector.reset(seed=reset_seed)
            episode_success = False

    eval_env = _make_env(env_paras, max_episode_steps=max_episode_steps)
    eval_controller = WaypointController(
        maze=eval_env.unwrapped.maze,
        model_callback=action_callback,
        maze_solver=maze_solver,
    )

    _create_dataset(
        collector,
        dataset_id=dataset_id,
        eval_env=eval_env,
        expert_policy=eval_controller.compute_action,
        num_episodes_average_score=1,
        algorithm_name=f"{maze_solver}+SAC",
        code_permalink="https://github.com/Farama-Foundation/minari-dataset-generation-scripts",
        author="local_antmaze_gen.py",
        author_email="",
        description=(
            f"Local AntMaze wrapper variant={variant}, worker={worker_index}, "
            f"target_episodes={target_episodes}, collected_steps={steps}, seed={seed}, "
            f"success_episodes={successful_episodes}, action_noise={action_noise}, "
            f"truncate_on_success={truncate_on_success}."
        ),
        requirements=_official_requirements(),
        minari_version=_minari_version_specifier(),
    )
    eval_env.close()
    collector.close()
    print(
        f"[local-antmaze-gen] {variant} worker={worker_index}: shard={dataset_id}, "
        f"target_episodes={target_episodes}, successful_episodes={successful_episodes}, "
        f"collected_steps={steps}, seed={seed}, action_noise={action_noise}, "
        f"truncate_on_success={truncate_on_success}."
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
    max_episode_steps: int | None,
    policy_file: Path,
    maze_solver: str,
    action_noise: float,
    truncate_on_success: bool,
):
    if variant not in ANTMAZE_VARIANTS:
        raise ValueError(f"Unknown AntMaze variant: {variant!r}")
    meta = ANTMAZE_VARIANTS[variant]
    if get_antmaze_variant_type(meta) != "local":
        raise ValueError(f"Variant {variant!r} is not local and cannot be generated")

    dataset_root = resolve_local_dataset_path(meta["dataset_path"])
    if overwrite and dataset_root.exists():
        shutil.rmtree(dataset_root)

    existing_episodes = _existing_episode_count(dataset_root)
    if existing_episodes >= target_episodes:
        print(
            f"[local-antmaze-gen] {variant}: existing_episodes={existing_episodes} "
            f">= target_episodes={target_episodes}; skipping"
        )
        return

    deficit = target_episodes - existing_episodes
    worker_count = max(1, min(num_workers, deficit))
    shard_targets = [deficit // worker_count] * worker_count
    for idx in range(deficit % worker_count):
        shard_targets[idx] += 1

    policy_file = policy_file.expanduser()
    if not policy_file.exists():
        raise FileNotFoundError(f"AntMaze SAC policy file not found: {policy_file}")

    print(
        f"[local-antmaze-gen] {variant}: existing_episodes={existing_episodes}, "
        f"target_episodes={target_episodes}, deficit={deficit}, workers={worker_count}, "
        f"policy_file={policy_file}, maze_solver={maze_solver}, action_noise={action_noise}, "
        f"truncate_on_success={truncate_on_success}"
    )

    collection_env_paras = dict(meta.get("collection_env_paras") or meta["env_paras"])
    eval_env_paras = dict(meta["env_paras"])
    shard_specs = []
    variant_seed_offset = int.from_bytes(
        hashlib.sha256(variant.encode("utf-8")).digest()[:4],
        "big",
    )
    for worker_index, shard_target in enumerate(shard_targets):
        shard_seed = seed + 1009 * worker_index + variant_seed_offset
        shard_specs.append(
            {
                "variant": variant,
                "env_paras": eval_env_paras,
                "collection_env_paras": collection_env_paras,
                "target_episodes": shard_target,
                "seed": shard_seed,
                "worker_index": worker_index,
                "max_episode_steps": max_episode_steps,
                "policy_file": str(policy_file),
                "maze_solver": maze_solver,
                "action_noise": action_noise,
                "truncate_on_success": truncate_on_success,
            }
        )

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        shard_results = list(executor.map(_collect_shard_from_kwargs, shard_specs))

    try:
        for dataset_id, shard_path in shard_results:
            shard_root = Path(shard_path)
            print(f"[local-antmaze-gen] {variant}: merging shard {dataset_id}")
            _merge_shard_into_final(shard_root, dataset_root)
    finally:
        for dataset_id, _ in shard_results:
            _cleanup_dataset_id(dataset_id)

    final_episodes = _existing_episode_count(dataset_root)
    print(
        f"[local-antmaze-gen] {variant}: final_episodes={final_episodes}, "
        f"dataset_path={dataset_root}"
    )


def main():
    args = parse_args()
    if args.target_episodes < 1:
        raise ValueError("--target-episodes must be >= 1")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.action_noise < 0:
        raise ValueError("--action-noise must be >= 0")
    if args.max_episode_steps is not None and args.max_episode_steps < 1:
        raise ValueError("--max-episode-steps must be >= 1")

    for variant in args.variants:
        generate_variant(
            variant,
            num_workers=args.num_workers,
            target_episodes=args.target_episodes,
            overwrite=args.overwrite,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            policy_file=args.policy_file,
            maze_solver=args.maze_solver,
            action_noise=args.action_noise,
            truncate_on_success=args.truncate_on_success,
        )


if __name__ == "__main__":
    main()
