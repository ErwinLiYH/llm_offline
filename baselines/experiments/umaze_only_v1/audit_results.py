from __future__ import annotations

import json
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = ROOT / "baseline_runs"
SAME_DIRECTION_PATH = ROOT / "reports/antmaze_umaze_only_v1_same_direction.json"
MULTI_LAYOUT_PATH = ROOT / "reports/antmaze_umaze_16layout_same_direction.json"
ALGORITHMS = ("mlp_bc", "iql", "td3_bc")
BASE_SEED = 20260716
EXPECTED_SINGLE_SUCCESS = {"mlp_bc": 80, "iql": 74, "td3_bc": 73}
EXPECTED_MULTI_SUCCESS = {"mlp_bc": 2, "iql": 0, "td3_bc": 3}
EXPECTED_HIDDEN_UNITS = {
    "mlp_bc": [1024, 1024, 1024],
    "iql": [512, 512, 512],
    "td3_bc": [1024, 1024, 1024],
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_finite_tree(value, *, context: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_tree(item, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_tree(item, context=f"{context}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"Non-finite value at {context}: {value}")


def _audit_episode_result(
    result: dict,
    *,
    expected_successes: int,
    start_cell: list[int],
    goal_cell: list[int],
) -> None:
    episodes = result["episodes"]
    assert len(episodes) == result["num_episodes"] == 100
    assert result["successful_episode_count"] == expected_successes
    assert math.isclose(result["success_rate"], expected_successes / 100)
    for index, episode in enumerate(episodes):
        assert episode["episode_index"] == index
        assert episode["seed"] == BASE_SEED + index
        assert episode["start_goal"]["start_cell"] == start_cell
        assert episode["start_goal"]["goal_cell"] == goal_cell
        assert episode["length"] == 700
        assert episode["truncated"] is True
        assert episode["terminated"] is False
        assert episode["success"] == (episode["first_success_step"] is not None)
        if episode["success"]:
            assert 1 <= episode["first_success_step"] <= 700
            assert episode["minimum_goal_distance"] <= 0.45 + 1e-9
        else:
            assert episode["minimum_goal_distance"] > 0.45 - 1e-9
    successes = [episode for episode in episodes if episode["success"]]
    assert len(successes) == expected_successes
    assert math.isclose(
        result["return_mean"],
        sum(episode["return"] for episode in episodes) / 100,
    )
    assert result["episodes_reaching_goal_cell"] == sum(
        episode["reached_goal_cell"] for episode in episodes
    )


def main() -> None:
    same_direction = _load_json(SAME_DIRECTION_PATH)
    multi_layout = _load_json(MULTI_LAYOUT_PATH)
    assert same_direction["protocol"]["training_scope"] == "umaze-only"
    assert multi_layout["protocol"]["training_scope"] == "16-layout"
    assert set(same_direction["algorithms"]) == set(ALGORITHMS)
    assert set(multi_layout["algorithms"]) == set(ALGORITHMS)

    selected_indices = None
    for algorithm in ALGORITHMS:
        experiment_id = (
            f"umazeonly1-antmaze-{algorithm}-e300-500k-r100-s{BASE_SEED}"
        )
        run_dir = RUNS_ROOT / experiment_id
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        manifest = _load_json(run_dir / "dataset_manifest.json")
        summary = _load_json(run_dir / "summary.json")

        assert config["experiment_id"] == experiment_id
        assert config["algorithm"] == algorithm
        assert config["resolved_train_variants"] == ["umaze"]
        assert config["resolved_eval_variants"] == ["umaze"]
        assert config["episode_keep_num"] == 300
        assert config["sampling_seed"] == BASE_SEED
        assert config["seed"] == 0
        assert config["n_steps"] == 500000
        assert config["evaluation"]["num_episodes"] == 100
        assert config["evaluation"]["seed"] == BASE_SEED
        assert config["network"]["hidden_units"] == EXPECTED_HIDDEN_UNITS[algorithm]
        assert config["observation"] == {
            "include_map": True,
            "include_location_sensing": True,
            "include_wall_sensing": True,
            "wall_sensing_version": "v3",
            "map_sensing_boundary_risk_threshold": 0.1,
        }

        assert set(manifest["variants"]) == {"umaze"}
        variant_manifest = manifest["variants"]["umaze"]
        assert variant_manifest["sampled_episode_count"] == 300
        assert variant_manifest["train_episode_count"] == 270
        assert variant_manifest["validation_episode_count"] == 30
        assert variant_manifest["train_transition_count"] == 189000
        assert variant_manifest["validation_transition_count"] == 21000
        assert manifest["observation_schema"]["dimension"] == 231
        indices = (
            variant_manifest["train_episode_indices"],
            variant_manifest["validation_episode_indices"],
        )
        if selected_indices is None:
            selected_indices = indices
        else:
            assert indices == selected_indices

        assert len(summary["training_history"]) == 50
        assert summary["training_history"][-1]["epoch"] == 50
        _assert_finite_tree(summary["training_history"], context=experiment_id)
        final_evaluation = summary["final_evaluation"]
        assert final_evaluation["step"] == 500000
        reverse = final_evaluation["rollout"]["variants"]["umaze"]
        assert reverse["num_episodes"] == 100
        assert reverse["successful_episode_count"] == 0
        assert reverse["success_rate"] == 0.0
        for index, episode in enumerate(reverse["episodes"]):
            assert episode["episode_index"] == index
            assert episode["seed"] == BASE_SEED + index
            assert episode["start_goal"]["start_cell"] == [1, 1]
            assert episode["start_goal"]["goal_cell"] == [3, 1]
            assert episode["success"] is False
            assert episode["first_success_step"] is None
        assert (run_dir / "model.d3").is_file()
        assert {
            path.name for path in (run_dir / "checkpoints").glob("step_*.d3")
        } == {
            "step_100000.d3",
            "step_200000.d3",
            "step_300000.d3",
            "step_400000.d3",
            "step_500000.d3",
        }

        _audit_episode_result(
            same_direction["algorithms"][algorithm],
            expected_successes=EXPECTED_SINGLE_SUCCESS[algorithm],
            start_cell=[3, 1],
            goal_cell=[1, 1],
        )
        _audit_episode_result(
            multi_layout["algorithms"][algorithm],
            expected_successes=EXPECTED_MULTI_SUCCESS[algorithm],
            start_cell=[3, 1],
            goal_cell=[1, 1],
        )

    print("Audited 3/3 UMaze-only trainings and 600 same-direction episodes.")
    print("Verified 300 reverse-direction episodes from training-time evaluation.")


if __name__ == "__main__":
    main()
