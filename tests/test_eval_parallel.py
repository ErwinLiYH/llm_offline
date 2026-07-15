import unittest
import threading
import tempfile
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

unsloth_stub = types.ModuleType("unsloth")
unsloth_stub.FastLanguageModel = object()
sys.modules.setdefault("unsloth", unsloth_stub)

import evaluate
from utils.distributed import DistributedContext
from utils.eval_parallel import (
    apply_rollout_config_defaults,
    assigned_eval_variants,
    eval_variant_assignments,
    resolve_rollout_worker_num,
)
from utils.eval_rollout import (
    ActionRolloutContext,
    generate_valid_continuous_actions_batch,
)
from utils.rollout.artifacts import write_step_log
from utils.rollout.worker_main import _prepare_eval_prompt_vars
from utils.video_writer import VideoSaveManager


class DummyEncoding(dict):
    __getattr__ = dict.__getitem__


class DummyTokenizer:
    def __init__(self):
        self.name_or_path = "dummy"
        self.chat_template = "dummy"
        self.eos_token_id = 1
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.padding_side = "right"

    def add_special_tokens(self, payload):
        return 0

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    ):
        suffix = "\nASSISTANT:" if add_generation_prompt else ""
        return messages[0]["content"] + suffix

    def __call__(
        self,
        *,
        text,
        return_tensors,
        add_special_tokens,
        padding,
        **kwargs,
    ):
        texts = text if isinstance(text, list) else [text]
        lengths = [max(len(value), 1) for value in texts]
        width = max(lengths)
        input_ids = []
        attention_mask = []
        for length in lengths:
            pad = width - length
            input_ids.append([0] * pad + [2] * length)
            attention_mask.append([0] * pad + [1] * length)
        return DummyEncoding(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        )


class DummyFormatter:
    def format_action(self, action):
        return ",".join(f"{float(value):.2f}" for value in action)

    def validate_action(self, action):
        values = np.asarray(action, dtype=np.float32)
        return values.shape == (2,) and bool(np.all(np.abs(values) <= 1.0))


class DummyContinuousModel:
    def __init__(self, outputs=None):
        self.batch_sizes = []
        self.outputs = outputs

    def __call__(self, *, input_ids, attention_mask, continuous_action):
        self.batch_sizes.append(int(input_ids.shape[0]))
        if self.outputs is not None:
            return self.outputs
        return torch.zeros((input_ids.shape[0], 2), dtype=torch.float32)


class DummyActionSpace:
    shape = (2,)
    low = np.asarray([-1.0, -1.0], dtype=np.float32)
    high = np.asarray([1.0, 1.0], dtype=np.float32)

    def seed(self, seed):
        self.last_seed = seed


class DummyEnv:
    def __init__(self):
        self.action_space = DummyActionSpace()
        self.seed = None
        self.steps = 0
        self.horizon = 1

    def reset(self, *, seed):
        self.seed = seed
        self.steps = 0
        self.horizon = 1 + seed % 2
        return {"observation": np.asarray([seed, 0], dtype=np.float32)}, {}

    def step(self, action):
        self.steps += 1
        terminated = self.steps >= self.horizon
        obs = {
            "observation": np.asarray(
                [self.seed, self.steps],
                dtype=np.float32,
            )
        }
        return obs, 1.0, terminated, False, {}

    def close(self):
        pass


class EvalParallelTest(unittest.TestCase):
    def test_eval_cli_seed_overrides_merged_config(self):
        args = evaluate.parse_args(
            ["--config", "base.yaml", "override.yaml", "--seed", "42"]
        )

        config = evaluate.apply_eval_cli_overrides({"seed": 64}, args)

        self.assertEqual(args.config, ["base.yaml", "override.yaml"])
        self.assertEqual(config["seed"], 42)

    def test_eval_cli_without_seed_preserves_merged_config(self):
        args = evaluate.parse_args(["--config", "eval.yaml"])

        config = evaluate.apply_eval_cli_overrides({"seed": 64}, args)

        self.assertEqual(config["seed"], 64)

    def test_checkpoint_sensing_config_is_inherited_for_eval(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            (Path(checkpoint_dir) / "config.yaml").write_text(
                "wall_sensing_version: v5\n"
                "map_sensing_boundary_risk_threshold: 0.25\n",
                encoding="utf-8",
            )

            config = evaluate.apply_checkpoint_sensing_config(
                {"model_path": checkpoint_dir}
            )

        self.assertEqual(config["wall_sensing_version"], "v5")
        self.assertEqual(config["map_sensing_boundary_risk_threshold"], 0.25)

    def test_eval_yaml_sensing_config_is_used_when_checkpoint_is_missing_fields(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            (Path(checkpoint_dir) / "config.yaml").write_text("{}\n", encoding="utf-8")

            config = evaluate.apply_checkpoint_sensing_config(
                {
                    "model_path": checkpoint_dir,
                    "wall_sensing_version": "v5",
                    "map_sensing_boundary_risk_threshold": 0.25,
                }
            )

        self.assertEqual(config["wall_sensing_version"], "v5")
        self.assertEqual(config["map_sensing_boundary_risk_threshold"], 0.25)

    def test_eval_yaml_sensing_config_conflict_with_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            (Path(checkpoint_dir) / "config.yaml").write_text(
                "wall_sensing_version: v5\n"
                "map_sensing_boundary_risk_threshold: 0.10\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "wall_sensing_version"):
                evaluate.apply_checkpoint_sensing_config(
                    {
                        "model_path": checkpoint_dir,
                        "wall_sensing_version": "v3",
                    }
                )

    def test_prepare_eval_prompt_vars_uses_formatter_hook(self):
        formatter = SimpleNamespace(
            prepare_eval_prompt_vars=mock.Mock(return_value={"maze_map": "eval"})
        )
        env = object()

        resolved = _prepare_eval_prompt_vars(
            formatter,
            {"maze_map": "train"},
            env,
        )

        self.assertEqual(resolved, {"maze_map": "eval"})
        formatter.prepare_eval_prompt_vars.assert_called_once_with(
            {"maze_map": "train"},
            env,
        )

    def test_video_save_manager_submits_without_waiting(self):
        started = threading.Event()
        release = threading.Event()

        def blocking_save(frames, output_path, fps):
            started.set()
            release.wait(timeout=5)

        manager = VideoSaveManager(
            {
                "video_save_workers": 1,
                "video_save_max_pending": 2,
            }
        )
        with mock.patch("utils.video_writer.save_video", side_effect=blocking_save):
            manager.submit([], "/tmp/video.gif", 20)
            self.assertTrue(started.wait(timeout=1))
            self.assertTrue(manager.asynchronous)
            release.set()
            manager.close()

    def test_video_save_manager_propagates_background_error(self):
        manager = VideoSaveManager(
            {
                "video_save_workers": 1,
                "video_save_max_pending": 2,
            }
        )
        with mock.patch(
            "utils.video_writer.save_video",
            side_effect=RuntimeError("encode failed"),
        ):
            manager.submit([], "/tmp/video.gif", 20)
            with self.assertRaisesRegex(RuntimeError, "encode failed"):
                manager.close()

    def test_video_save_manager_blocks_only_at_pending_limit(self):
        started = threading.Event()
        release = threading.Event()
        third_submit_returned = threading.Event()

        def blocking_save(frames, output_path, fps):
            started.set()
            release.wait(timeout=5)

        manager = VideoSaveManager(
            {
                "video_save_workers": 1,
                "video_save_max_pending": 2,
            }
        )
        with mock.patch("utils.video_writer.save_video", side_effect=blocking_save):
            manager.submit([], "/tmp/video-1.gif", 20)
            self.assertTrue(started.wait(timeout=1))
            manager.submit([], "/tmp/video-2.gif", 20)

            submit_thread = threading.Thread(
                target=lambda: (
                    manager.submit([], "/tmp/video-3.gif", 20),
                    third_submit_returned.set(),
                )
            )
            submit_thread.start()
            self.assertFalse(third_submit_returned.wait(timeout=0.1))
            release.set()
            self.assertTrue(third_submit_returned.wait(timeout=1))
            submit_thread.join(timeout=1)
            manager.close()

    def test_variant_assignment_round_robin(self):
        context = DistributedContext(
            backend="ddp",
            rank=1,
            world_size=3,
            local_rank=1,
            is_distributed=True,
        )
        variants = ["a", "b", "c", "d", "e"]
        self.assertEqual(
            assigned_eval_variants(
                variants,
                context,
                distribute_variants=True,
            ),
            ["b", "e"],
        )
        self.assertEqual(
            eval_variant_assignments(
                variants,
                context,
                distribute_variants=True,
            ),
            {0: ["a", "d"], 1: ["b", "e"], 2: ["c"]},
        )
        self.assertEqual(resolve_rollout_worker_num({}), 1)
        self.assertEqual(resolve_rollout_worker_num({"rollout_worker_num": 3}), 3)
        with self.assertRaises(ValueError):
            resolve_rollout_worker_num({"rollout_worker_num": 0})
        with self.assertRaisesRegex(ValueError, "rename it to rollout_worker_num"):
            apply_rollout_config_defaults({"eval_parallel_episodes": 2})

    def test_standalone_eval_output_mode_keeps_standalone_dir(self):
        self.assertEqual(evaluate.resolve_eval_output_mode({}), "standalone")
        self.assertEqual(
            evaluate.get_standalone_results_dir("/tmp/results", "abc123"),
            "/tmp/results/standalone_abc123",
        )

    def test_training_eval_results_dir_uses_training_context(self):
        epoch_context = evaluate.resolve_training_eval_context(
            {
                "training_eval_context": {
                    "eval_type": "epoch",
                    "epoch": 3,
                    "batch_step": 99,
                    "epoch_step": None,
                    "optimizer_step": 7,
                    "scheduled_step": None,
                    "scheduled_epoch_step": None,
                    "train_loss": 1.2,
                    "val_loss": 2.3,
                    "val_metrics": {"mae": 0.4},
                    "checkpoint_path": "/ckpt/ep3(step99)",
                    "experiment_id": "exp",
                }
            }
        )
        step_context = dict(epoch_context)
        step_context.update(
            {
                "eval_type": "step",
                "batch_step": 42,
                "epoch_step": 5,
                "scheduled_step": 40,
                "scheduled_epoch_step": 4,
            }
        )

        self.assertEqual(
            evaluate.get_training_results_dir("/tmp/results", epoch_context),
            "/tmp/results/ep3(step99)",
        )
        self.assertEqual(
            evaluate.get_training_results_dir("/tmp/results", step_context),
            "/tmp/results/step42",
        )

    def test_training_eval_context_fields_are_added_to_result(self):
        context = evaluate.resolve_training_eval_context(
            {
                "training_eval_context": {
                    "eval_type": "step",
                    "epoch": 2,
                    "batch_step": 12,
                    "epoch_step": 4,
                    "optimizer_step": 6,
                    "scheduled_step": 10,
                    "scheduled_epoch_step": 3,
                    "train_loss": 1.0,
                    "val_loss": 0.5,
                    "val_metrics": {"mae": 0.25},
                    "checkpoint_path": "/ckpt/step12",
                    "experiment_id": "exp",
                }
            }
        )

        result = evaluate.apply_training_eval_context_to_result(
            {"variant": "umaze"},
            context,
        )

        self.assertEqual(result["eval_type"], "step")
        self.assertEqual(result["eval_tag"], "step12")
        self.assertEqual(result["epoch"], 2)
        self.assertEqual(result["batch_step"], 12)
        self.assertEqual(result["checkpoint_path"], "/ckpt/step12")
        self.assertEqual(result["experiment_id"], "exp")
        self.assertEqual(result["val_mae"], 0.25)

    def test_continuous_batch_clips_and_restores_padding_side(self):
        tokenizer = DummyTokenizer()
        tokenizer.pad_token = None
        model = DummyContinuousModel(
            outputs=torch.tensor(
                [[0.25, -0.5], [2.0, -2.0]],
                dtype=torch.float32,
            )
        )
        context = ActionRolloutContext(
            action_token_mode="parallel_l1",
            action_generation_config={
                "action_sampling": False,
                "action_temperature": 1.0,
                "action_top_p": 1.0,
                "action_top_k": 0,
            },
            action_codec=None,
            collect_bin_probabilities=False,
            generation_max_new_tokens=20,
            allowed_token_ids=None,
        )
        results = generate_valid_continuous_actions_batch(
            model=model,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            formatter=DummyFormatter(),
            prompts=["a", "longer"],
            config={"action_token_mode": "parallel_l1"},
            action_context=context,
            action_shape=(2,),
            action_dim=2,
        )
        self.assertTrue(np.allclose(results[0].action, [0.25, -0.5]))
        self.assertTrue(np.allclose(results[1].action, [1.0, -1.0]))
        self.assertEqual(tokenizer.padding_side, "right")
        self.assertIsNone(tokenizer.pad_token)

    def test_step_logs_are_combined_per_episode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "episode_0001"
            for step_index in range(2):
                write_step_log(
                    str(episode_dir),
                    step_index,
                    prompt=f"prompt {step_index + 1}",
                    action_text=f"action {step_index + 1}",
                    executed_action=f"executed {step_index + 1}",
                    parse_status="ok",
                    attempt_count=1,
                )

            step_log_path = episode_dir / "steps.txt"
            content = step_log_path.read_text(encoding="utf-8")

            self.assertTrue(step_log_path.is_file())
            self.assertFalse((episode_dir / "steps").exists())
            self.assertEqual(content.count("=" * 80), 4)
            self.assertEqual(content.count("Step 0001"), 1)
            self.assertEqual(content.count("Step 0002"), 1)
            self.assertLess(content.index("Step 0001"), content.index("Step 0002"))
            self.assertIn("prompt 1", content)
            self.assertIn("prompt 2", content)


if __name__ == "__main__":
    unittest.main()
