from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import d3rlpy
import numpy as np
import yaml

import crossmaze
from baselines.data.observation import BaselineObservationWrapper


ALGORITHMS = ("mlp_bc", "iql", "td3_bc")
BASE_SEED = 20260716
NUM_EPISODES = 100
DEFAULT_RUN_TEMPLATE = (
    "umazeonly1-antmaze-{algorithm}-e300-500k-r100-s20260716"
)
TRAIN_DIRECTION_PAIR = {
    "reset_cell": np.asarray([3, 1], dtype=np.int64),
    "goal_cell": np.asarray([1, 1], dtype=np.int64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate UMaze-only checkpoints on the dataset-direction pair."
    )
    parser.add_argument("--runs-root", default="baseline_runs")
    parser.add_argument("--run-template", default=DEFAULT_RUN_TEMPLATE)
    parser.add_argument("--training-scope", default="umaze-only")
    parser.add_argument(
        "--output",
        default="reports/antmaze_umaze_only_v1_same_direction.json",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def run_dir(runs_root: Path, algorithm: str, run_template: str) -> Path:
    return runs_root / run_template.format(algorithm=algorithm)


def evaluate_algorithm(path: Path, *, device: str) -> dict:
    config = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
    algo = d3rlpy.load_learnable(str(path / "model.d3"), device=device)
    env = BaselineObservationWrapper(
        crossmaze.make(
            "antmaze",
            "umaze",
            mode="eval",
            config={
                "reward_type": "sparse",
                "eval_start_goal_mode": "fix-start-goal",
                "wall_sensing_version": config["observation"][
                    "wall_sensing_version"
                ],
                "map_sensing_boundary_risk_threshold": config["observation"][
                    "map_sensing_boundary_risk_threshold"
                ],
            },
        ),
        env_family="antmaze",
        observation_config=config["observation"],
    )

    episodes = []
    try:
        for episode_index in range(NUM_EPISODES):
            seed = BASE_SEED + episode_index
            observation, _ = env.reset(seed=seed, options=TRAIN_DIRECTION_PAIR)
            reset_state = env.last_crossmaze_state
            start_cell = [int(value) for value in reset_state["position_cell"]]
            goal_cell = [int(value) for value in reset_state["goal_cell"]]
            start_xy = [float(value) for value in reset_state["position_xy"]]
            goal_xy = np.asarray(reset_state["goal_xy"], dtype=np.float64)

            episode_return = 0.0
            episode_length = 0
            first_success_step = None
            minimum_goal_distance = float("inf")
            reached_goal_cell = False
            terminated = False
            truncated = False
            while not (terminated or truncated):
                state = env.last_crossmaze_state
                position_xy = np.asarray(state["position_xy"], dtype=np.float64)
                minimum_goal_distance = min(
                    minimum_goal_distance,
                    float(np.linalg.norm(position_xy - goal_xy)),
                )
                reached_goal_cell = reached_goal_cell or [
                    int(value) for value in state["position_cell"]
                ] == goal_cell

                action = np.asarray(algo.predict(observation[None])[0])
                action = np.clip(
                    action, env.action_space.low, env.action_space.high
                ).astype(env.action_space.dtype, copy=False)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                episode_length += 1
                if bool(info.get("success", False)) and first_success_step is None:
                    first_success_step = episode_length

            final_state = env.last_crossmaze_state
            final_xy_array = np.asarray(final_state["position_xy"], dtype=np.float64)
            minimum_goal_distance = min(
                minimum_goal_distance,
                float(np.linalg.norm(final_xy_array - goal_xy)),
            )
            final_cell = [int(value) for value in final_state["position_cell"]]
            reached_goal_cell = reached_goal_cell or final_cell == goal_cell
            episodes.append(
                {
                    "episode_index": episode_index,
                    "seed": seed,
                    "start_goal": {
                        "sampling_mode": "explicit-dataset-direction",
                        "start_cell": start_cell,
                        "goal_cell": goal_cell,
                        "start_xy": start_xy,
                        "goal_xy": [float(value) for value in goal_xy],
                    },
                    "success": first_success_step is not None,
                    "first_success_step": first_success_step,
                    "return": episode_return,
                    "length": episode_length,
                    "minimum_goal_distance": minimum_goal_distance,
                    "reached_goal_cell": reached_goal_cell,
                    "final_cell": final_cell,
                    "final_xy": [float(value) for value in final_xy_array],
                    "final_goal_distance": float(
                        np.linalg.norm(final_xy_array - goal_xy)
                    ),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
    finally:
        env.close()

    successes = [episode for episode in episodes if episode["success"]]
    success_steps = [episode["first_success_step"] for episode in successes]
    return {
        "experiment_id": config["experiment_id"],
        "algorithm": config["algorithm"],
        "checkpoint_step": config["n_steps"],
        "pair": {"start_cell": [3, 1], "goal_cell": [1, 1]},
        "num_episodes": len(episodes),
        "successful_episode_count": len(successes),
        "success_rate": len(successes) / len(episodes),
        "first_success_step_mean": (
            float(np.mean(success_steps)) if success_steps else None
        ),
        "first_success_step_std": (
            float(np.std(success_steps)) if success_steps else None
        ),
        "return_mean": float(np.mean([episode["return"] for episode in episodes])),
        "return_std": float(np.std([episode["return"] for episode in episodes])),
        "length_mean": float(np.mean([episode["length"] for episode in episodes])),
        "minimum_goal_distance_mean": float(
            np.mean([episode["minimum_goal_distance"] for episode in episodes])
        ),
        "minimum_goal_distance_median": float(
            np.median([episode["minimum_goal_distance"] for episode in episodes])
        ),
        "episodes_reaching_goal_cell": sum(
            episode["reached_goal_cell"] for episode in episodes
        ),
        "episodes": episodes,
    }


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output = Path(args.output)
    results = {
        "protocol": {
            "env_family": "antmaze",
            "variant": "umaze",
            "training_scope": args.training_scope,
            "train_direction_pair": {
                "start_cell": [3, 1],
                "goal_cell": [1, 1],
            },
            "num_episodes_per_algorithm": NUM_EPISODES,
            "seed": BASE_SEED,
            "success_distance_threshold": 0.45,
        },
        "algorithms": {},
    }
    for algorithm in ALGORITHMS:
        path = run_dir(runs_root, algorithm, args.run_template)
        if not (path / "summary.json").is_file() or not (path / "model.d3").is_file():
            raise FileNotFoundError(f"Incomplete UMaze-only run: {path}")
        result = evaluate_algorithm(path, device=args.device)
        results["algorithms"][algorithm] = result
        print(
            f"[umaze same direction] {algorithm}: "
            f"success={result['successful_episode_count']}/{result['num_episodes']}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[umaze same direction] wrote {output}")


if __name__ == "__main__":
    main()
