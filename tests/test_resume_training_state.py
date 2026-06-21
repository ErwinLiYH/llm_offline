import unittest
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import torch
import yaml

unsloth_stub = types.ModuleType("unsloth")
unsloth_stub.FastLanguageModel = object()
sys.modules.setdefault("unsloth", unsloth_stub)

import train
from train import (
    _completed_optimizer_steps_in_epoch,
    _resume_epoch_plan,
    _validate_resume_compatibility,
)
from utils.distributed import DistributedContext
from utils.lr_scheduler import lr_scale_for_step


class ResumeEpochPlanTest(unittest.TestCase):
    def test_epoch_checkpoint_runs_additional_full_epochs(self):
        start, end, resume_epoch, completed = _resume_epoch_plan(
            additional_epochs=2,
            train_batches_per_epoch=100,
            resume_loop_state={
                "current_epoch": 3,
                "completed_epoch_batch_step": 100,
            },
        )

        self.assertEqual((start, end), (4, 5))
        self.assertEqual(resume_epoch, 3)
        self.assertEqual(completed, 100)

    def test_step_checkpoint_finishes_epoch_then_additional_epochs(self):
        start, end, resume_epoch, completed = _resume_epoch_plan(
            additional_epochs=2,
            train_batches_per_epoch=100,
            resume_loop_state={
                "current_epoch": 3,
                "completed_epoch_batch_step": 40,
            },
        )

        self.assertEqual((start, end), (3, 5))
        self.assertEqual(resume_epoch, 3)
        self.assertEqual(completed, 40)

    def test_step_checkpoint_with_zero_epochs_only_finishes_current_epoch(self):
        start, end, _, _ = _resume_epoch_plan(
            additional_epochs=0,
            train_batches_per_epoch=100,
            resume_loop_state={
                "current_epoch": 3,
                "completed_epoch_batch_step": 40,
            },
        )

        self.assertEqual((start, end), (3, 3))

    def test_completed_optimizer_steps_requires_optimizer_boundary(self):
        self.assertEqual(_completed_optimizer_steps_in_epoch(40, 8, 100), 5)
        self.assertEqual(_completed_optimizer_steps_in_epoch(100, 8, 100), 13)
        with self.assertRaisesRegex(ValueError, "mid gradient-accumulation"):
            _completed_optimizer_steps_in_epoch(42, 8, 100)


class ResumeCompatibilityTest(unittest.TestCase):
    def test_compatibility_reports_changed_train_critical_fields(self):
        saved = {
            "train_variants": ["umaze"],
            "world_size": 1,
            "batch_size": 4,
            "gradient_accumulation_steps": 2,
            "action_token_mode": "bin",
            "action_dim": 2,
            "dataset_load_partitions": 1,
            "train_batches_per_epoch": 10,
            "optimizer_param_groups": [{"param_count": 3, "weight_decay": 0.0}],
            "partition_stats": [],
        }
        current = dict(saved)
        current["batch_size"] = 8
        current["world_size"] = 2

        with self.assertRaisesRegex(ValueError, "batch_size"):
            _validate_resume_compatibility(saved, current)


class ResumeLrContinuityTest(unittest.TestCase):
    def test_saved_original_lr_horizon_is_used_after_resume(self):
        saved_scheduler = {
            "scheduler_type": "linear",
            "total_training_steps": 10,
            "warmup_steps": 0,
            "lr_decay_steps": 10,
            "min_lr_ratio": 0.2,
            "optimizer_step": 10,
        }

        next_scale = lr_scale_for_step(
            step_index=saved_scheduler["optimizer_step"] + 1,
            total_training_steps=saved_scheduler["total_training_steps"],
            warmup_steps=saved_scheduler["warmup_steps"],
            decay_steps=saved_scheduler["lr_decay_steps"],
            scheduler_type=saved_scheduler["scheduler_type"],
            min_lr_ratio=saved_scheduler["min_lr_ratio"],
        )

        self.assertAlmostEqual(next_scale, 0.2)

    def test_optimizer_state_round_trip_preserves_lr(self):
        param = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([param], lr=0.123)
        state = optimizer.state_dict()

        restored = torch.optim.AdamW([torch.nn.Parameter(torch.tensor([2.0]))], lr=1.0)
        restored.load_state_dict(state)

        self.assertAlmostEqual(restored.param_groups[0]["lr"], 0.123)


class DummyWandbLogger:
    enabled = True

    def __init__(self):
        self.logs = []

    def log(self, payload):
        self.logs.append(payload)


def _isolated_eval_base_config(tmpdir: str, *, retry_times: int = 0) -> dict:
    return {
        "env_family": "pointmaze",
        "model_name": "org/dummy-model",
        "experiment_id": "exp1",
        "result_root": tmpdir,
        "eval_num_episodes": 1,
        "eval_seed": 3,
        "prompt_templete_index": ["0"],
        "eval_distribute_variants": True,
        "training_eval_rollout_isolated": True,
        "isolated_eval_rollout_retry_times": retry_times,
    }


def _write_fake_isolated_eval_results(parent_config, selection_tag, child_config):
    context = child_config["training_eval_context"]
    for variant in child_config["variants"]:
        result_dir = train.get_eval_variant_results_dir(
            parent_config,
            selection_tag,
            variant,
            epoch=context["epoch"] if context["eval_type"] == "epoch" else None,
            step=context["batch_step"] if context["eval_type"] == "step" else None,
        )
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        result = {
            "variant": variant,
            "success_rate": 0.75,
            "mean_episode_steps": 11.0,
            "mean_return": 2.0,
            "eval_rank": context.get("eval_rank"),
            "eval_world_size": context.get("eval_world_size"),
        }
        Path(result_dir, "result.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )


def _isolated_eval_config_from_command(command):
    config_path = command[command.index("--config") + 1]
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


class IsolatedTrainingEvalTest(unittest.TestCase):
    def test_training_eval_config_keeps_mode_specific_action_keys(self):
        base = {
            "env_family": "pointmaze",
            "eval_num_episodes": 1,
            "action_token_mode": "parallel_l1",
            "gaussian_log_std_min": -2.5,
            "gaussian_log_std_max": 0.5,
            "gaussian_log_std_init": -1.0,
            "student_t_df": 3.0,
            "continuous_mean_l1_weight": 0.2,
        }

        l1_config = train._build_training_eval_config(base)

        self.assertNotIn("gaussian_log_std_min", l1_config)
        self.assertNotIn("gaussian_log_std_max", l1_config)
        self.assertNotIn("gaussian_log_std_init", l1_config)
        self.assertNotIn("student_t_df", l1_config)
        self.assertNotIn("continuous_mean_l1_weight", l1_config)

        gaussian_config = train._build_training_eval_config(
            {**base, "action_token_mode": "parallel_gaussian"}
        )

        self.assertEqual(gaussian_config["gaussian_log_std_min"], -2.5)
        self.assertEqual(gaussian_config["gaussian_log_std_max"], 0.5)
        self.assertEqual(gaussian_config["gaussian_log_std_init"], -1.0)
        self.assertNotIn("student_t_df", gaussian_config)

    def test_success_reads_results_and_logs_wandb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _isolated_eval_base_config(tmpdir)
            selection_tag = "umaze"

            def fake_run(command, **kwargs):
                child_config = _isolated_eval_config_from_command(command)
                _write_fake_isolated_eval_results(config, selection_tag, child_config)
                return subprocess.CompletedProcess(command, 0)

            wandb = DummyWandbLogger()
            with mock.patch("train.subprocess.run", side_effect=fake_run) as run_mock:
                train._run_eval_isolated(
                    config,
                    selection_tag,
                    ["umaze"],
                    "epoch",
                    train_loss=1.0,
                    val_loss=0.5,
                    val_metrics={"mae": 0.25},
                    checkpoint_dir="/tmp/checkpoint/ep1",
                    epoch=1,
                    optimizer_step=2,
                    wandb_logger=wandb,
                    train_env_steps=100.0,
                    dist_context=DistributedContext(backend="single"),
                )

            self.assertEqual(run_mock.call_count, 1)
            self.assertEqual(len(wandb.logs), 1)
            self.assertEqual(wandb.logs[0]["eval/umaze/success_rate"], 0.75)
            self.assertEqual(wandb.logs[0]["eval/umaze/mean_episode_steps"], 11.0)

            attempt_config = Path(
                tmpdir,
                "dummy-model",
                "train=pointmaze-umaze",
                "exp=exp1",
                "epoch_1",
                "isolated_eval",
                "rank_0",
                "attempt_1.yaml",
            )
            child_config = yaml.safe_load(attempt_config.read_text(encoding="utf-8"))
            self.assertEqual(child_config["eval_output_mode"], "training")
            self.assertEqual(child_config["parallel_backend"], "single")
            self.assertEqual(child_config["variants"], ["umaze"])
            self.assertEqual(child_config["training_eval_context"]["train_loss"], 1.0)
            self.assertEqual(
                child_config["training_eval_context"]["val_metrics"],
                {"mae": 0.25},
            )

    def test_retries_until_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _isolated_eval_base_config(tmpdir, retry_times=1)
            selection_tag = "umaze"
            calls = {"count": 0}

            def fake_run(command, **kwargs):
                calls["count"] += 1
                if calls["count"] == 1:
                    return subprocess.CompletedProcess(command, 1)
                child_config = _isolated_eval_config_from_command(command)
                _write_fake_isolated_eval_results(config, selection_tag, child_config)
                return subprocess.CompletedProcess(command, 0)

            wandb = DummyWandbLogger()
            with mock.patch("train.subprocess.run", side_effect=fake_run):
                train._run_eval_isolated(
                    config,
                    selection_tag,
                    ["umaze"],
                    "epoch",
                    train_loss=1.0,
                    val_loss=0.5,
                    checkpoint_dir="/tmp/checkpoint/ep1",
                    epoch=1,
                    optimizer_step=2,
                    wandb_logger=wandb,
                    train_env_steps=100.0,
                    dist_context=DistributedContext(backend="single"),
                )

            self.assertEqual(calls["count"], 2)
            self.assertEqual(wandb.logs[0]["eval/umaze/success_rate"], 0.75)

    def test_all_failures_warn_and_log_failure_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _isolated_eval_base_config(tmpdir, retry_times=1)
            wandb = DummyWandbLogger()

            with mock.patch(
                "train.subprocess.run",
                return_value=subprocess.CompletedProcess(["evaluate.py"], 1),
            ):
                train._run_eval_isolated(
                    config,
                    "umaze",
                    ["umaze"],
                    "epoch",
                    train_loss=1.0,
                    val_loss=0.5,
                    checkpoint_dir="/tmp/checkpoint/ep1",
                    epoch=1,
                    optimizer_step=2,
                    wandb_logger=wandb,
                    train_env_steps=100.0,
                    dist_context=DistributedContext(backend="single"),
                )

            self.assertEqual(len(wandb.logs), 1)
            self.assertEqual(wandb.logs[0]["eval/umaze/rollout_failed"], 1.0)
            self.assertEqual(wandb.logs[0]["eval/umaze/isolated_attempts"], 2.0)
            self.assertNotIn("eval/umaze/success_rate", wandb.logs[0])

    def test_ddp_rank_passes_only_assigned_variants_to_child(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _isolated_eval_base_config(tmpdir)
            selection_tag = "all"
            seen_child_configs = []

            def fake_run(command, **kwargs):
                child_config = _isolated_eval_config_from_command(command)
                seen_child_configs.append(child_config)
                _write_fake_isolated_eval_results(config, selection_tag, child_config)
                return subprocess.CompletedProcess(command, 0)

            dist_context = DistributedContext(
                backend="ddp",
                rank=1,
                world_size=3,
                local_rank=1,
                is_distributed=True,
            )
            with mock.patch("train.subprocess.run", side_effect=fake_run), mock.patch(
                "train.all_gather_objects",
                side_effect=lambda value, context: [value],
            ):
                train._run_eval_isolated(
                    config,
                    selection_tag,
                    ["a", "b", "c", "d", "e"],
                    "step",
                    train_loss=1.0,
                    val_loss=0.5,
                    checkpoint_dir="/tmp/checkpoint/step8",
                    epoch=2,
                    batch_step=8,
                    epoch_step=4,
                    optimizer_step=3,
                    wandb_logger=DummyWandbLogger(),
                    train_env_steps=100.0,
                    dist_context=dist_context,
                )

            self.assertEqual(seen_child_configs[0]["variants"], ["b", "e"])
            self.assertEqual(
                seen_child_configs[0]["training_eval_context"]["eval_rank"],
                1,
            )
            self.assertEqual(
                seen_child_configs[0]["training_eval_context"]["eval_world_size"],
                3,
            )


if __name__ == "__main__":
    unittest.main()
