import unittest
import sys
import types

import torch

unsloth_stub = types.ModuleType("unsloth")
unsloth_stub.FastLanguageModel = object()
sys.modules.setdefault("unsloth", unsloth_stub)

from train import (
    _completed_optimizer_steps_in_epoch,
    _resume_epoch_plan,
    _validate_resume_compatibility,
)
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


if __name__ == "__main__":
    unittest.main()
