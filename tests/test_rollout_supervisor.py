import os
import time
import unittest

from utils.rollout.protocol import ActionRequest, ActionResponse, EpisodeResult
from utils.rollout.supervisor import run_episode_supervisor


def fake_worker_entry(worker_id, control_queue, event_queue, worker_config):
    event_queue.put({"type": "ready", "worker_id": worker_id, "pid": os.getpid()})
    while True:
        message = control_queue.get()
        if message.get("type") == "shutdown":
            break
        episode_index = int(message["episode_index"])
        attempt = int(message.get("attempt", 1))
        crash_episodes = set(worker_config["config"].get("fake_crash_episodes", []))
        if episode_index in crash_episodes and attempt == 1:
            os._exit(139)

        request = ActionRequest(
            request_id=f"{worker_id}-{episode_index}-{attempt}",
            worker_id=worker_id,
            episode_index=episode_index,
            step_index=0,
            prompt=f"episode {episode_index}",
            action_shape=(2,),
            action_dim=2,
            action_low=[-1.0, -1.0],
            action_high=[1.0, 1.0],
        )
        event_queue.put(
            {
                "type": "action_request",
                "worker_id": worker_id,
                "pid": os.getpid(),
                "request": request,
            }
        )
        response_message = control_queue.get(timeout=5)
        assert response_message["type"] == "action_response"
        result = EpisodeResult(
            variant=worker_config["variant"],
            episode_index=episode_index,
            seed=message.get("seed"),
            episode_return=1.0 + episode_index,
            success=episode_index % 2 == 0,
            steps=1,
            parse_failures=0,
            fallbacks=0,
            action_time_seconds=0.01,
            action_count=1,
            worker_id=worker_id,
            worker_pid=os.getpid(),
        )
        event_queue.put(
            {
                "type": "episode_result",
                "worker_id": worker_id,
                "pid": os.getpid(),
                "episode_index": episode_index,
                "attempt": attempt,
                "result": result,
            }
        )
        if worker_config.get("rollout_worker_lifetime") == "episode":
            break


class FakePolicy:
    def __init__(self):
        self.batch_sizes = []

    def respond(self, requests):
        self.batch_sizes.append(len(requests))
        return [
            ActionResponse(
                request_id=request.request_id,
                action=[0.0, 0.0],
                executed_action_text="0,0",
                generated_attempts=["0,0"],
                generated_probability_logs=[],
                attempt_count=1,
                parse_status="success",
                parse_failures=0,
                fallback_count=0,
                action_time_seconds=0.01,
                generation_count=1,
            )
            for request in requests
        ]


class RolloutSupervisorTest(unittest.TestCase):
    def test_slot_workers_use_multiple_processes(self):
        policy = FakePolicy()
        result = run_episode_supervisor(
            config={
                "num_episodes": 4,
                "seed": 10,
                "rollout_worker_num": 2,
                "rollout_worker_lifetime": "slot",
                "rollout_worker_retries": 0,
                "rollout_worker_start_timeout_seconds": 5,
                "rollout_action_timeout_seconds": 5,
                "policy_batch_timeout_ms": 1,
            },
            variant="dummy",
            mode="eval",
            template="template",
            policy=policy,
            variant_results_dir=None,
            worker_target=fake_worker_entry,
            multiprocessing_context="fork",
        )

        self.assertEqual(len(result.episode_results), 4)
        self.assertGreaterEqual(len(set(result.workers_used)), 2)
        self.assertEqual([episode.seed for episode in result.episode_results], [10, 11, 12, 13])
        self.assertFalse(result.worker_failures)

    def test_worker_crash_is_retried_without_killing_parent(self):
        policy = FakePolicy()
        result = run_episode_supervisor(
            config={
                "num_episodes": 2,
                "seed": 1,
                "rollout_worker_num": 1,
                "rollout_worker_lifetime": "slot",
                "rollout_worker_retries": 1,
                "rollout_worker_start_timeout_seconds": 5,
                "rollout_action_timeout_seconds": 5,
                "policy_batch_timeout_ms": 1,
                "fake_crash_episodes": [0],
            },
            variant="dummy",
            mode="eval",
            template="template",
            policy=policy,
            variant_results_dir=None,
            worker_target=fake_worker_entry,
            multiprocessing_context="fork",
        )

        self.assertEqual(len(result.episode_results), 2)
        self.assertEqual(result.worker_failures[0].exitcode, 139)
        self.assertFalse(result.episode_results[0].worker_failed)

    def test_worker_crash_returns_failed_episode_after_retries(self):
        policy = FakePolicy()
        result = run_episode_supervisor(
            config={
                "num_episodes": 1,
                "seed": 1,
                "rollout_worker_num": 1,
                "rollout_worker_lifetime": "slot",
                "rollout_worker_retries": 0,
                "rollout_worker_start_timeout_seconds": 5,
                "rollout_action_timeout_seconds": 5,
                "policy_batch_timeout_ms": 1,
                "fake_crash_episodes": [0],
            },
            variant="dummy",
            mode="eval",
            template="template",
            policy=policy,
            variant_results_dir=None,
            worker_target=fake_worker_entry,
            multiprocessing_context="fork",
        )

        self.assertEqual(len(result.episode_results), 1)
        self.assertTrue(result.episode_results[0].worker_failed)
        self.assertEqual(result.worker_failures[0].exitcode, 139)


if __name__ == "__main__":
    unittest.main()

