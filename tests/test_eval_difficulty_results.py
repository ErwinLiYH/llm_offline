import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.rollout.evaluate_runner import (
    _mean_difficulty_component,
    _write_eval_position_pool,
    run_evaluate_variant,
)
from utils.rollout.protocol import EpisodeResult, SupervisorResult
from utils.rollout.score_runner import _score_episode_dict


def _episode(index: int, components: dict | None) -> EpisodeResult:
    return EpisodeResult(
        variant="umaze",
        episode_index=index,
        seed=17 + index,
        episode_return=1.0,
        success=True,
        steps=2,
        parse_failures=0,
        fallbacks=0,
        action_time_seconds=0.1,
        action_count=2,
        worker_id=0,
        worker_pid=123,
        start_goal_difficulty=0.5 if components is not None else None,
        start_goal_difficulty_components=components,
    )


class EvalDifficultyResultTest(unittest.TestCase):
    def test_component_means_ignore_missing_episode_difficulty(self):
        episodes = [
            _episode(0, {"length_score": 0.25}),
            _episode(1, {"length_score": 0.75}),
            _episode(2, None),
        ]

        self.assertEqual(
            _mean_difficulty_component(episodes, "length_score"),
            0.5,
        )
        self.assertIsNone(_mean_difficulty_component(episodes, "branch_score"))

    def test_eval_position_pool_is_written_next_to_variant_result(self):
        payload = {
            "difficulty_version": "v2",
            "map_difficulty": 0.6,
            "start_goal_list": [{"difficulty": 0.7}],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_eval_position_pool(
                variant_results_dir=temp_dir,
                payload=payload,
            )

            self.assertEqual(
                Path(path).name,
                "eval_position_pool.json",
            )
            self.assertEqual(
                json.loads(Path(path).read_text(encoding="utf-8")),
                payload,
            )

    def test_official_score_schema_removes_eval_difficulty(self):
        episode = _episode(0, {"version": "v2", "length_score": 0.5})
        payload = _score_episode_dict(episode)

        self.assertNotIn("start_goal_difficulty", payload)
        self.assertNotIn("start_goal_difficulty_components", payload)

    def test_evaluate_variant_saves_map_episode_and_pool_difficulty(self):
        components = {
            "version": "v2",
            "length_score": 0.4,
            "branch_score": 0.2,
            "detour_score": 0.6,
        }
        episodes = [_episode(0, components), _episode(1, components)]
        supervisor_result = SupervisorResult(
            variant="umaze",
            episode_results=episodes,
        )
        config = {
            "env_family": "pointmaze",
            "num_episodes": 2,
            "seed": 17,
            "action_token_mode": "text",
            "eval_start_goal_mode": "hard-sample",
            "eval_hard_sample_top_n": 5,
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "utils.rollout.evaluate_runner.get_formatter",
            return_value=object(),
        ), patch(
            "utils.rollout.evaluate_runner.RolloutPolicy",
            return_value=object(),
        ), patch(
            "utils.rollout.evaluate_runner.run_episode_supervisor",
            return_value=supervisor_result,
        ):
            result = run_evaluate_variant(
                config=config,
                variant="umaze",
                model=None,
                tokenizer=None,
                device="cpu",
                template="test",
                variant_results_dir=temp_dir,
            )

            self.assertEqual(result["difficulty_version"], "v2")
            self.assertIsNotNone(result["map_difficulty"])
            self.assertEqual(result["mean_start_goal_length_score"], 0.4)
            self.assertEqual(result["mean_start_goal_branch_score"], 0.2)
            self.assertEqual(result["mean_start_goal_detour_score"], 0.6)
            self.assertEqual(
                result["episode_results"][0][
                    "start_goal_difficulty_components"
                ],
                components,
            )
            self.assertTrue(Path(result["eval_position_pool_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
