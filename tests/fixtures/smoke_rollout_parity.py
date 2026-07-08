"""CPU-only end-to-end rollout smoke: real env workers + deterministic stub policy.

Usage: python smoke_rollout.py <output_dir>
Runs pointmaze eval, antmaze eval, and pointmaze score rollouts with 2 episodes
each and record_step_logs enabled. steps.txt files (prompts embedded) land under
<output_dir>/<mode>_<family>_<variant>/ for pre/post refactor diffing.
"""
import os
import sys

from utils.prompt_loader import load_named_templates
from utils.rollout.protocol import ActionResponse
from utils.rollout.supervisor import run_episode_supervisor


class StubPolicy:
    """Deterministic small actions as a function of (episode, step)."""

    def respond(self, requests):
        responses = []
        for request in requests:
            dim = int(request.action_dim)
            k = request.episode_index * 1000 + request.step_index
            action = [
                round(0.05 * (((k + i * 7) % 11) - 5) / 5.0, 6) for i in range(dim)
            ]
            responses.append(
                ActionResponse(
                    request_id=request.request_id,
                    action=action,
                    executed_action_text=",".join(str(v) for v in action),
                    generated_attempts=["stub"],
                    generated_probability_logs=[],
                    attempt_count=1,
                    parse_status="ok",
                    parse_failures=0,
                    fallback_count=0,
                    action_time_seconds=0.0,
                    generation_count=1,
                )
            )
        return responses


def base_config(env_family):
    return {
        "env_family": env_family,
        "num_episodes": 2,
        "seed": 64,
        "history_num": 0,
        "history_stride": 1,
        "parse_retry_limit": 3,
        "record_step_logs": True,
        "record_video": False,
        "rollout_worker_num": 1,
        "rollout_worker_lifetime": "slot",
        "rollout_worker_retries": 0,
        "rollout_worker_start_timeout_seconds": 120,
        "rollout_action_timeout_seconds": 120,
        "policy_batch_timeout_ms": 10,
        "max_steps": 40,
    }


def main():
    out_root = sys.argv[1]
    runs = [
        ("eval", "pointmaze", "umaze", {"env_kwargs": {"continuing_task": False}}),
        ("eval", "pointmaze", "medium", {"wall_sensing_version": "v5"}),
        ("eval", "antmaze", "umaze", {}),
        ("score", "pointmaze", "medium", {}),
    ]
    for mode, family, variant, extra in runs:
        config = base_config(family)
        config.update(extra)
        template = load_named_templates(family, ["parallel_full_sensing"])[0]
        results_dir = os.path.join(out_root, f"{mode}_{family}_{variant}")
        os.makedirs(results_dir, exist_ok=True)
        result = run_episode_supervisor(
            config=config,
            variant=variant,
            mode=mode,
            template=template,
            policy=StubPolicy(),
            variant_results_dir=results_dir,
        )
        assert not result.worker_failures, result.worker_failures
        summary = [
            (er.episode_index, er.steps, round(er.episode_return, 4), er.success)
            for er in sorted(result.episode_results, key=lambda e: e.episode_index)
        ]
        print(f"{mode}/{family}/{variant}: {summary}")
    print("smoke done")


if __name__ == "__main__":
    main()
