"""Generate local PointMaze Minari datasets through Farama's official scripts."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
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
from minari.data_collector.episode_buffer import EpisodeBuffer
from minari.storage.local import get_dataset_path

from crossmaze.eval_position import (
    build_hard_start_goal_pair_space as _build_hard_sample_pair_space,
)
from crossmaze.reward import REWARD_TYPES, normalize_reward_type
from crossmaze.seed_map import (
    build_seed_map_env_paras,
    generate_seed_map,
    normalize_seed_map_spec,
    stable_seed_map_int,
)
from data.seed_map_corpus import (
    append_minari_shard_to_seed_map_corpus,
    create_seed_map_corpus,
    existing_seed_map_trajectory_indices,
    finalize_seed_map_corpus,
)
from data.pointmaze.variants import (
    POINTMAZE_VARIANTS,
    get_pointmaze_variant_type,
    resolve_local_dataset_path,
)
from utils.datagen_paths import (
    add_datagen_path_args,
    configure_minari_temporary_root,
    publish_new_dataset,
    remove_temporary_dataset,
    resolve_final_dataset_path,
    resolve_temporary_dataset_root,
    temporary_dataset_merge_path,
)
from utils.seed_map_config import (
    add_seed_map_generation_args,
    resolve_seed_map_generation_path,
    validate_seed_map_generation_args,
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
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--target-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reward-type",
        choices=REWARD_TYPES,
        default=None,
        help=(
            "Reward stored in generated transitions. Defaults to the registered "
            "variant reward type; alternate rewards use a separate dataset path."
        ),
    )
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
    parser.add_argument(
        "--hard-sample",
        action="store_true",
        help=(
            "Sample explicit start/goal cell pairs with the legacy data-generation "
            "difficulty bias and save only successful episodes."
        ),
    )
    parser.add_argument(
        "--hard-retry",
        type=int,
        default=5,
        help=(
            "Maximum failed retries for each hard-sampled pair. A pair is tried "
            "at most 1 + hard_retry times."
        ),
    )
    parser.add_argument(
        "--hard-sample-alpha",
        type=float,
        default=1.0,
        help=(
            "Rank-linear difficulty bias for --hard-sample. 0 gives uniform "
            "pair sampling; alpha=1 makes the hardest pair twice as likely as "
            "the easiest pair."
        ),
    )
    parser.add_argument(
        "--hard-sample-top-n",
        type=int,
        default=0,
        help=(
            "Only sample from the top N hardest reachable start/goal pairs after "
            "difficulty sorting. 0 keeps all pairs."
        ),
    )
    add_datagen_path_args(parser)
    add_seed_map_generation_args(parser)
    return parser.parse_args()


def _existing_episode_count(dataset_root: Path) -> int:
    data_path = dataset_root / "data"
    if not data_path.exists():
        return 0
    return int(MinariDataset(data_path).total_episodes)


def _clean_collection_map(maze_map: list[list[object]]) -> list[list[int]]:
    return [[1 if cell == 1 else 0 for cell in row] for row in maze_map]


def _free_cells(maze_map: list[list[object]]) -> list[tuple[int, int]]:
    return [
        (row_idx, col_idx)
        for row_idx, row in enumerate(maze_map)
        for col_idx, cell in enumerate(row)
        if cell != 1
    ]


def _hard_pair_probabilities(pair_space: list[dict]) -> np.ndarray:
    weights = np.asarray(
        [float(pair["sample_weight"]) for pair in pair_space],
        dtype=np.float64,
    )
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(pair_space), 1.0 / len(pair_space), dtype=np.float64)
    return weights / total


def _hard_pair_probability_summary(pair_space: list[dict]) -> dict:
    if not pair_space:
        return {
            "pair_sample_probability_min": None,
            "pair_sample_probability_max": None,
            "pair_sample_probability_max_over_min": None,
        }
    probabilities = _hard_pair_probabilities(pair_space)
    min_probability = float(probabilities.min())
    max_probability = float(probabilities.max())
    return {
        "pair_sample_probability_min": min_probability,
        "pair_sample_probability_max": max_probability,
        "pair_sample_probability_max_over_min": (
            float(max_probability / min_probability)
            if min_probability > 0.0
            else None
        ),
    }


def _stat_triplet(records: list[dict], key: str, prefix: str) -> dict:
    values = [float(record[key]) for record in records if key in record]
    if not values:
        return {
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_mean": None,
        }
    return {
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
        f"{prefix}_mean": float(sum(values) / len(values)),
    }


def _difficulty_record_for_episode(
    pair_record: dict,
    *,
    episode_index: int,
    attempts_for_pair: int,
) -> dict:
    return {
        "episode_index": int(episode_index),
        "start_cell": [
            int(pair_record["start_cell"][0]),
            int(pair_record["start_cell"][1]),
        ],
        "goal_cell": [
            int(pair_record["goal_cell"][0]),
            int(pair_record["goal_cell"][1]),
        ],
        "path_len": int(pair_record["path_len"]),
        "away_steps": int(pair_record["away_steps"]),
        "away_frac": float(pair_record["away_frac"]),
        "difficulty": float(pair_record["difficulty"]),
        "attempts_for_pair": int(attempts_for_pair),
    }


def _episode_data_to_buffer(
    episode: dict,
    *,
    episode_id: int | None = None,
) -> EpisodeBuffer:
    return EpisodeBuffer(
        id=episode_id,
        seed=episode.get("seed"),
        options=episode.get("options"),
        observations=episode["observations"],
        actions=episode["actions"],
        rewards=episode["rewards"],
        terminations=episode["terminations"],
        truncations=episode["truncations"],
        infos=episode.get("infos"),
    )


def _read_collected_episode(
    collector: DataCollector,
    episode_index: int,
) -> EpisodeBuffer:
    episode = next(collector._storage.get_episodes([episode_index]))
    return _episode_data_to_buffer(episode)


def _replace_collector_storage(
    collector: DataCollector,
    selected_episodes: list[EpisodeBuffer],
):
    collector._reset_storage()
    collector._buffer = None
    collector._episode_id = 0
    collector._storage.update_episodes(
        _episode_data_to_buffer(
            {
                "observations": episode.observations,
                "actions": episode.actions,
                "rewards": episode.rewards,
                "terminations": episode.terminations,
                "truncations": episode.truncations,
                "infos": episode.infos,
                "seed": episode.seed,
                "options": episode.options,
            },
            episode_id=idx,
        )
        for idx, episode in enumerate(selected_episodes)
    )
    collector._episode_id = len(selected_episodes)


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
    hard_sample: bool,
    hard_retry: int,
    hard_sample_alpha: float,
    hard_pair_space: list[dict],
    temporary_dataset_root: str,
) -> dict:
    configure_minari_temporary_root(temporary_dataset_root)
    reward_type = normalize_reward_type(env_paras.get("reward_type"))
    # Keep shard IDs namespace-free so Minari does not recursively scan the
    # shared dataset root while parallel workers create/delete temporary dirs.
    dataset_id = (
        f"pointmaze-{variant}-{reward_type}-shard-{uuid.uuid4().hex[:12]}-v0"
    )
    collect_env_paras = dict(env_paras)
    if post_success_hold_steps > 0 or hard_sample:
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
    steps = 0
    attempted_episodes = 0
    successful_episodes = 0
    hard_pairs_sampled = 0
    hard_pairs_succeeded = 0
    hard_pairs_exhausted = 0
    hard_failed_attempts = 0
    episode_difficulty: list[dict] = []

    if hard_sample:
        hard_rng = np.random.default_rng(seed + 31_337)
        hard_probabilities = _hard_pair_probabilities(hard_pair_space)
        selected_episodes: list[EpisodeBuffer] = []
        reset_seed = seed
        while len(selected_episodes) < target_episodes:
            pair_index = int(
                hard_rng.choice(len(hard_pair_space), p=hard_probabilities)
            )
            pair_record = hard_pair_space[pair_index]
            hard_pairs_sampled += 1
            pair_succeeded = False

            for pair_attempt in range(1 + hard_retry):
                attempt_seed = reset_seed
                reset_seed += 1
                controller.waypoint_targets = None
                controller.global_target_xy = np.full(2, np.inf)
                obs, _ = collector.reset(
                    seed=attempt_seed,
                    options={
                        "reset_cell": np.asarray(
                            pair_record["start_cell"],
                            dtype=np.int64,
                        ),
                        "goal_cell": np.asarray(
                            pair_record["goal_cell"],
                            dtype=np.int64,
                        ),
                    },
                )
                holding = False
                hold_steps_remaining = 0
                episode_success = False

                while True:
                    action = (
                        _hold_action(obs, post_success_hold_noise_std)
                        if holding
                        else _waypoint_action(controller, obs)
                    )
                    obs, _, terminated, truncated, info = collector.step(action)
                    steps += 1
                    step_success = bool(info.get("success", False))
                    episode_success = episode_success or step_success

                    if post_success_hold_steps <= 0:
                        if step_success or terminated or truncated:
                            break
                        continue

                    if terminated or truncated:
                        break
                    if holding:
                        hold_steps_remaining -= 1
                        if hold_steps_remaining <= 0:
                            # A manual reset flushes the successful hold episode
                            # into the collector storage before the next attempt.
                            collector.reset(seed=attempt_seed)
                            break
                        continue
                    if step_success:
                        holding = True
                        hold_steps_remaining = post_success_hold_steps

                attempted_episodes += 1
                if episode_success:
                    successful_episodes += 1
                    selected_episodes.append(
                        _read_collected_episode(
                            collector,
                            collector._episode_id - 1,
                        )
                    )
                    episode_difficulty.append(
                        _difficulty_record_for_episode(
                            pair_record,
                            episode_index=len(selected_episodes) - 1,
                            attempts_for_pair=pair_attempt + 1,
                        )
                    )
                    hard_pairs_succeeded += 1
                    pair_succeeded = True
                    break
                hard_failed_attempts += 1

            if not pair_succeeded:
                hard_pairs_exhausted += 1

        _replace_collector_storage(collector, selected_episodes)
    else:
        obs, _ = collector.reset(seed=seed)
        episodes = 0
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
                    attempted_episodes += 1
                    episodes += 1
                    successful_episodes += 1
                elif terminated or truncated:
                    attempted_episodes += 1
                    obs, _ = collector.reset()
                continue

            if terminated or truncated:
                attempted_episodes += 1
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
                    attempted_episodes += 1
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
            f"reward_type={reward_type}, "
            f"target_episodes={target_episodes}, collected_steps={steps}, seed={seed}, "
            f"post_success_hold_steps={post_success_hold_steps}, "
            f"post_success_hold_noise_std={post_success_hold_noise_std}, "
            f"hard_sample={hard_sample}, hard_retry={hard_retry}, "
            f"hard_sample_alpha={hard_sample_alpha}, "
            f"hard_pairs_sampled={hard_pairs_sampled}, "
            f"hard_pairs_succeeded={hard_pairs_succeeded}, "
            f"hard_pairs_exhausted={hard_pairs_exhausted}, "
            f"hard_failed_attempts={hard_failed_attempts}."
        ),
        requirements=_official_requirements(),
        minari_version=_minari_version_specifier(),
    )
    eval_env.close()
    collector.close()
    print(
        f"[local-pointmaze-gen] {variant} worker={worker_index}: shard={dataset_id}, "
        f"target_episodes={target_episodes}, successful_episodes={successful_episodes}, "
        f"reward_type={reward_type}, "
        f"collected_steps={steps}, seed={seed}, "
        f"post_success_hold_steps={post_success_hold_steps}, "
        f"post_success_hold_noise_std={post_success_hold_noise_std}, "
        f"hard_sample={hard_sample}, hard_pairs_sampled={hard_pairs_sampled}, "
        f"hard_pairs_succeeded={hard_pairs_succeeded}, "
        f"hard_pairs_exhausted={hard_pairs_exhausted}, "
        f"hard_failed_attempts={hard_failed_attempts}."
    )
    return {
        "dataset_id": dataset_id,
        "path": str(get_dataset_path(dataset_id)),
        "reward_type": reward_type,
        "target_episodes": int(target_episodes),
        "saved_episodes": int(target_episodes),
        "successful_episodes": int(successful_episodes),
        "attempted_episodes": int(attempted_episodes),
        "collected_steps": int(steps),
        "hard_pairs_sampled": int(hard_pairs_sampled),
        "hard_pairs_succeeded": int(hard_pairs_succeeded),
        "hard_pairs_exhausted": int(hard_pairs_exhausted),
        "hard_failed_attempts": int(hard_failed_attempts),
        "episode_difficulty": episode_difficulty,
    }


def _collect_shard_from_kwargs(kwargs: dict) -> dict:
    return _collect_shard(**kwargs)


def _copy_dataset_root(src_root: Path, dst_root: Path):
    if dst_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset path: {dst_root}")
    dst_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_root, dst_root)


def _merge_minari_dataset(source_root: Path, target_root: Path):
    if not (target_root / "data").exists():
        _copy_dataset_root(source_root, target_root)
        return
    target_dataset = MinariDataset(target_root / "data")
    source_dataset = MinariDataset(source_root / "data")
    target_dataset.storage.update_from_storage(source_dataset.storage)


def _merge_shards_into_temporary_dataset(
    shard_results: list[dict],
    merged_root: Path,
    *,
    label: str,
) -> None:
    for result in shard_results:
        dataset_id = str(result["dataset_id"])
        shard_root = Path(str(result["path"]))
        print(
            f"[local-pointmaze-gen] {label}: merging temporary shard "
            f"{dataset_id} into {merged_root}"
        )
        _merge_minari_dataset(shard_root, merged_root)


def _publish_merged_dataset(merged_root: Path, final_root: Path) -> None:
    if (final_root / "data").exists():
        _merge_minari_dataset(merged_root, final_root)
        return
    publish_new_dataset(merged_root, final_root)


def _cleanup_dataset_id(dataset_id: str, dataset_path: str | Path):
    remove_temporary_dataset(dataset_id, dataset_path)


def _write_generation_summary(
    *,
    dataset_root: Path,
    variant: str,
    reward_type: str,
    target_episodes: int,
    final_episodes: int,
    seed: int,
    max_episode_steps: int,
    post_success_hold_steps: int,
    post_success_hold_noise_std: float,
    hard_sample: bool,
    hard_retry: int,
    hard_sample_alpha: float,
    hard_sample_top_n: int,
    hard_pair_space: list[dict],
    hard_pair_space_total: int,
    shard_results: list[dict],
):
    attempted_episodes = sum(
        int(result.get("attempted_episodes", 0)) for result in shard_results
    )
    collected_steps = sum(
        int(result.get("collected_steps", 0)) for result in shard_results
    )
    hard_pairs_sampled = sum(
        int(result.get("hard_pairs_sampled", 0)) for result in shard_results
    )
    hard_pairs_succeeded = sum(
        int(result.get("hard_pairs_succeeded", 0)) for result in shard_results
    )
    hard_pairs_exhausted = sum(
        int(result.get("hard_pairs_exhausted", 0)) for result in shard_results
    )
    hard_failed_attempts = sum(
        int(result.get("hard_failed_attempts", 0)) for result in shard_results
    )
    shard_episode_difficulty = [
        record
        for result in shard_results
        for record in result.get("episode_difficulty", [])
    ]
    generated_episode_offset = max(0, final_episodes - len(shard_episode_difficulty))
    episode_difficulty = [
        {
            **record,
            "episode_index": int(generated_episode_offset + index),
        }
        for index, record in enumerate(shard_episode_difficulty)
    ]
    summary = {
        "variant": variant,
        "reward_type": normalize_reward_type(reward_type),
        "target_episodes": int(target_episodes),
        "final_episodes": int(final_episodes),
        "seed": int(seed),
        "max_episode_steps": int(max_episode_steps),
        "post_success_hold_steps": int(post_success_hold_steps),
        "post_success_hold_noise_std": float(post_success_hold_noise_std),
        "attempted_episodes": int(attempted_episodes),
        "collected_steps": int(collected_steps),
        "hard_sample": bool(hard_sample),
        "hard_retry": int(hard_retry),
        "hard_sample_alpha": float(hard_sample_alpha),
        "hard_sample_top_n": int(hard_sample_top_n),
        "hard_pair_space_total": int(hard_pair_space_total),
        "hard_pair_space_used": int(len(hard_pair_space)),
        "hard_pairs_sampled": int(hard_pairs_sampled),
        "hard_pairs_succeeded": int(hard_pairs_succeeded),
        "hard_pairs_exhausted": int(hard_pairs_exhausted),
        "hard_failed_attempts": int(hard_failed_attempts),
        "episode_difficulty": episode_difficulty,
    }
    summary.update(_stat_triplet(hard_pair_space, "difficulty", "pair_difficulty"))
    summary.update(_stat_triplet(hard_pair_space, "path_len", "pair_path_len"))
    summary.update(_stat_triplet(hard_pair_space, "away_steps", "pair_away_steps"))
    summary.update(_hard_pair_probability_summary(hard_pair_space))
    summary.update(_stat_triplet(episode_difficulty, "difficulty", "saved_difficulty"))
    summary.update(_stat_triplet(episode_difficulty, "path_len", "saved_path_len"))
    summary.update(_stat_triplet(episode_difficulty, "away_steps", "saved_away_steps"))
    (dataset_root / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_variant(
    variant: str,
    *,
    num_workers: int,
    target_episodes: int,
    overwrite: bool,
    seed: int,
    reward_type: str | None,
    max_episode_steps: int,
    post_success_hold_steps: int,
    post_success_hold_noise_std: float,
    hard_sample: bool = False,
    hard_retry: int = 5,
    hard_sample_alpha: float = 1.0,
    hard_sample_top_n: int = 0,
    dataset_root_override: str | Path | None = None,
    temporary_dataset_root: str | Path | None = None,
):
    if variant not in POINTMAZE_VARIANTS:
        raise ValueError(f"Unknown PointMaze variant: {variant!r}")
    meta = POINTMAZE_VARIANTS[variant]
    if get_pointmaze_variant_type(meta) != "local":
        raise ValueError(f"Variant {variant!r} is not local and cannot be generated")
    if hard_retry < 0:
        raise ValueError("--hard-retry must be >= 0")
    if hard_sample_alpha < 0:
        raise ValueError("--hard-sample-alpha must be >= 0")
    if hard_sample_top_n < 0:
        raise ValueError("--hard-sample-top-n must be >= 0")

    default_reward_type = normalize_reward_type(
        meta["env_paras"].get("reward_type", meta["prompt_vars"]["reward_type"])
    )
    reward_type = normalize_reward_type(reward_type, default=default_reward_type)
    env_paras = dict(meta["env_paras"])
    env_paras["reward_type"] = reward_type
    default_dataset_root = resolve_local_dataset_path(
        meta["dataset_path"],
        reward_type=reward_type,
        default_reward_type=default_reward_type,
    )
    dataset_root = resolve_final_dataset_path(
        default_dataset_root,
        dataset_root_override,
    )
    temporary_dataset_root = resolve_temporary_dataset_root(
        temporary_dataset_root
    )
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

    clean_map = _clean_collection_map(env_paras["maze_map"])
    hard_pair_space: list[dict] = []
    hard_pair_space_total = 0
    if hard_sample:
        hard_pair_space, hard_pair_space_total = _build_hard_sample_pair_space(
            clean_map,
            _free_cells(clean_map),
            hard_sample_alpha,
            hard_sample_top_n,
        )

    print(
        f"[local-pointmaze-gen] {variant}: existing_episodes={existing_episodes}, "
        f"target_episodes={target_episodes}, deficit={deficit}, workers={worker_count}, "
        f"reward_type={reward_type}, "
        f"post_success_hold_steps={post_success_hold_steps}, "
        f"post_success_hold_noise_std={post_success_hold_noise_std}, "
        f"hard_sample={hard_sample}, hard_retry={hard_retry}, "
        f"hard_sample_alpha={hard_sample_alpha}, "
        f"hard_sample_top_n={hard_sample_top_n}, "
        f"hard_pairs={len(hard_pair_space)}/{hard_pair_space_total}, "
        f"dataset_root={dataset_root}, "
        f"temporary_dataset_root={temporary_dataset_root}"
    )
    shard_specs = []
    variant_seed_offset = int.from_bytes(hashlib.sha256(variant.encode("utf-8")).digest()[:4], "big")
    for worker_index, shard_target in enumerate(shard_targets):
        shard_seed = seed + 1009 * worker_index + variant_seed_offset
        shard_specs.append(
            {
                "variant": variant,
                "env_paras": env_paras,
                "target_episodes": shard_target,
                "seed": shard_seed,
                "worker_index": worker_index,
                "max_episode_steps": max_episode_steps,
                "post_success_hold_steps": post_success_hold_steps,
                "post_success_hold_noise_std": post_success_hold_noise_std,
                "hard_sample": hard_sample,
                "hard_retry": hard_retry,
                "hard_sample_alpha": hard_sample_alpha,
                "hard_pair_space": hard_pair_space,
                "temporary_dataset_root": str(temporary_dataset_root),
            }
        )

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        shard_results = list(executor.map(_collect_shard_from_kwargs, shard_specs))

    try:
        with temporary_dataset_merge_path(
            temporary_dataset_root,
            label=f"pointmaze-{variant}",
        ) as merged_root:
            _merge_shards_into_temporary_dataset(
                shard_results,
                merged_root,
                label=variant,
            )
            print(
                f"[local-pointmaze-gen] {variant}: publishing merged dataset "
                f"from {merged_root} to {dataset_root}"
            )
            _publish_merged_dataset(merged_root, dataset_root)
    finally:
        for result in shard_results:
            _cleanup_dataset_id(
                str(result["dataset_id"]),
                str(result["path"]),
            )

    final_episodes = _existing_episode_count(dataset_root)
    _write_generation_summary(
        dataset_root=dataset_root,
        variant=variant,
        reward_type=reward_type,
        target_episodes=target_episodes,
        final_episodes=final_episodes,
        seed=seed,
        max_episode_steps=max_episode_steps,
        post_success_hold_steps=post_success_hold_steps,
        post_success_hold_noise_std=post_success_hold_noise_std,
        hard_sample=hard_sample,
        hard_retry=hard_retry,
        hard_sample_alpha=hard_sample_alpha,
        hard_sample_top_n=hard_sample_top_n,
        hard_pair_space=hard_pair_space,
        hard_pair_space_total=hard_pair_space_total,
        shard_results=shard_results,
    )
    print(
        f"[local-pointmaze-gen] {variant}: final_episodes={final_episodes}, "
        f"dataset_path={dataset_root}"
    )


def _collect_seed_map_shard_from_kwargs(kwargs: dict) -> dict:
    map_seed = int(kwargs.pop("map_seed"))
    seed_map_spec = normalize_seed_map_spec(kwargs.pop("seed_map_spec"))
    reward_type = normalize_reward_type(kwargs.pop("reward_type"))
    max_episode_steps = kwargs.pop("max_episode_steps")
    hard_sample = bool(kwargs.get("hard_sample", False))
    hard_sample_alpha = float(kwargs.get("hard_sample_alpha", 1.0))
    hard_sample_top_n = int(kwargs.pop("hard_sample_top_n", 0))

    maze_map = generate_seed_map(map_seed, seed_map_spec)
    env_paras = build_seed_map_env_paras(
        "pointmaze",
        maze_map=maze_map,
        reward_type=reward_type,
        max_episode_steps=max_episode_steps,
    )
    hard_pair_space = []
    hard_pair_space_total = 0
    if hard_sample:
        hard_pair_space, hard_pair_space_total = _build_hard_sample_pair_space(
            maze_map,
            _free_cells(maze_map),
            hard_sample_alpha,
            hard_sample_top_n,
        )
    result = _collect_shard(
        variant=f"seed-map-{map_seed}",
        env_paras=env_paras,
        max_episode_steps=int(env_paras["max_episode_steps"]),
        hard_pair_space=hard_pair_space,
        **kwargs,
    )
    result.update(
        {
            "map_seed": map_seed,
            "maze_map": maze_map,
            "hard_pair_space_total": hard_pair_space_total,
        }
    )
    return result


def generate_seed_map_corpus(args) -> Path:
    seed_map_spec = validate_seed_map_generation_args(args)
    reward_type = normalize_reward_type(args.reward_type, default="sparse")
    temporary_dataset_root = resolve_temporary_dataset_root(
        getattr(args, "temporary_dataset_root", None)
    )
    repo_root = Path(__file__).resolve().parent
    dataset_root = resolve_seed_map_generation_path(
        args,
        env_family="pointmaze",
        reward_type=reward_type,
        repo_root=repo_root,
        spec=seed_map_spec,
    )
    collection_config = {
        "collection_seed": int(args.seed),
        "max_episode_steps": int(args.max_episode_steps),
        "post_success_hold_steps": int(args.post_success_hold_steps),
        "post_success_hold_noise_std": float(args.post_success_hold_noise_std),
        "hard_sample": bool(args.hard_sample),
        "hard_retry": int(args.hard_retry),
        "hard_sample_alpha": float(args.hard_sample_alpha),
        "hard_sample_top_n": int(args.hard_sample_top_n),
    }
    manifest = create_seed_map_corpus(
        dataset_root,
        env_family="pointmaze",
        reward_type=reward_type,
        seed_start=int(args.seed_map_start),
        seed_end=int(args.seed_map_end),
        trajectories_per_seed=int(args.seed_map_trajectories_per_seed),
        seed_map_spec=seed_map_spec,
        collection_config=collection_config,
        overwrite=bool(args.overwrite),
    )
    if bool(manifest.get("complete", False)):
        print(
            f"[local-pointmaze-gen][seed-map] corpus already complete: {dataset_root}"
        )
        return dataset_root

    capacity = int(args.seed_map_trajectories_per_seed)
    tasks = []
    for map_seed in range(int(args.seed_map_start), int(args.seed_map_end)):
        existing = existing_seed_map_trajectory_indices(dataset_root, map_seed)
        expected_existing = list(range(len(existing)))
        if existing != expected_existing:
            raise ValueError(
                f"Seed-map corpus map_seed={map_seed} has non-contiguous "
                f"trajectory indices {existing}; expected {expected_existing}"
            )
        if len(existing) >= capacity:
            continue
        tasks.append(
            {
                "map_seed": map_seed,
                "seed_map_spec": seed_map_spec.to_dict(),
                "reward_type": reward_type,
                "target_episodes": capacity,
                "seed": int(
                    stable_seed_map_int(
                        "pointmaze",
                        "seed_map_collection",
                        int(args.seed),
                        map_seed,
                        bits=31,
                    )
                ),
                "worker_index": map_seed,
                "max_episode_steps": int(args.max_episode_steps),
                "post_success_hold_steps": int(args.post_success_hold_steps),
                "post_success_hold_noise_std": float(
                    args.post_success_hold_noise_std
                ),
                "hard_sample": bool(args.hard_sample),
                "hard_retry": int(args.hard_retry),
                "hard_sample_alpha": float(args.hard_sample_alpha),
                "hard_sample_top_n": int(args.hard_sample_top_n),
                "temporary_dataset_root": str(temporary_dataset_root),
                "existing_count": len(existing),
            }
        )

    if not tasks:
        finalize_seed_map_corpus(dataset_root)
        print(
            f"[local-pointmaze-gen][seed-map] corpus complete: {dataset_root}"
        )
        return dataset_root

    worker_count = min(max(int(args.num_workers), 1), len(tasks))
    print(
        "[local-pointmaze-gen][seed-map] "
        f"dataset={dataset_root}, range=[{args.seed_map_start}, {args.seed_map_end}), "
        f"trajectories_per_seed={capacity}, pending_maps={len(tasks)}, "
        f"workers={worker_count}, size_spec={seed_map_spec.to_dict()}, "
        f"hard_sample={args.hard_sample}, "
        f"temporary_dataset_root={temporary_dataset_root}"
    )
    worker_tasks = []
    for task in tasks:
        worker_task = dict(task)
        worker_task.pop("existing_count")
        worker_tasks.append(worker_task)

    completed_results = []
    try:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for task, result in zip(
                tasks,
                executor.map(_collect_seed_map_shard_from_kwargs, worker_tasks),
            ):
                completed_results.append(result)
                existing_count = int(task["existing_count"])
                missing_count = capacity - existing_count
                map_seed = int(result["map_seed"])
                with temporary_dataset_merge_path(
                    temporary_dataset_root,
                    label=f"pointmaze-seed-map-{map_seed}",
                ) as merged_root:
                    _merge_shards_into_temporary_dataset(
                        [result],
                        merged_root,
                        label=f"seed-map-{map_seed}",
                    )
                    print(
                        "[local-pointmaze-gen][seed-map] "
                        f"map_seed={map_seed}: publishing merged map dataset "
                        f"from {merged_root} to {dataset_root}"
                    )
                    append_minari_shard_to_seed_map_corpus(
                        dataset_root,
                        merged_root,
                        map_seed=map_seed,
                        maze_map=result["maze_map"],
                        trajectory_start_index=existing_count,
                        trajectory_count=missing_count,
                        source_episode_start_index=existing_count,
                    )
                print(
                    "[local-pointmaze-gen][seed-map] "
                    f"map_seed={map_seed}: appended={missing_count}, "
                    f"dataset={dataset_root}"
                )
                _cleanup_dataset_id(
                    str(result["dataset_id"]),
                    str(result["path"]),
                )
    finally:
        for result in completed_results:
            _cleanup_dataset_id(
                str(result["dataset_id"]),
                str(result["path"]),
            )

    final_manifest = finalize_seed_map_corpus(dataset_root)
    print(
        "[local-pointmaze-gen][seed-map] complete: "
        f"maps={final_manifest['generated_map_count']}, "
        f"episodes={final_manifest['total_episodes']}, "
        f"steps={final_manifest['total_steps']}, dataset={dataset_root}"
    )
    return dataset_root


def main():
    args = parse_args()
    dataset_root_override = getattr(args, "dataset_root", None)
    temporary_dataset_root = resolve_temporary_dataset_root(
        getattr(args, "temporary_dataset_root", None)
    )
    args.temporary_dataset_root = temporary_dataset_root
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.post_success_hold_steps < 0:
        raise ValueError("--post-success-hold-steps must be >= 0")
    if args.post_success_hold_noise_std < 0:
        raise ValueError("--post-success-hold-noise-std must be >= 0")
    if args.hard_retry < 0:
        raise ValueError("--hard-retry must be >= 0")
    if args.hard_sample_alpha < 0:
        raise ValueError("--hard-sample-alpha must be >= 0")
    if args.hard_sample_top_n < 0:
        raise ValueError("--hard-sample-top-n must be >= 0")
    if getattr(args, "use_seed_map", False):
        if args.variants:
            print(
                "[local-pointmaze-gen][seed-map] --variants is ignored because "
                "--use-seed-map is enabled."
            )
        if args.target_episodes is not None:
            print(
                "[local-pointmaze-gen][seed-map] --target-episodes is ignored; "
                "use --seed-map-trajectories-per-seed."
            )
        generate_seed_map_corpus(args)
        return

    if not args.variants:
        raise ValueError("--variants is required unless --use-seed-map is enabled")
    if args.target_episodes is None or args.target_episodes < 1:
        raise ValueError(
            "--target-episodes must be >= 1 unless --use-seed-map is enabled"
        )
    for variant in args.variants:
        generate_variant(
            variant,
            num_workers=args.num_workers,
            target_episodes=args.target_episodes,
            overwrite=args.overwrite,
            seed=args.seed,
            reward_type=args.reward_type,
            max_episode_steps=args.max_episode_steps,
            post_success_hold_steps=args.post_success_hold_steps,
            post_success_hold_noise_std=args.post_success_hold_noise_std,
            hard_sample=args.hard_sample,
            hard_retry=args.hard_retry,
            hard_sample_alpha=args.hard_sample_alpha,
            hard_sample_top_n=args.hard_sample_top_n,
            dataset_root_override=dataset_root_override,
            temporary_dataset_root=temporary_dataset_root,
        )


if __name__ == "__main__":
    main()
