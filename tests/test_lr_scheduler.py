import unittest

from utils.lr_scheduler import (
    resolve_lr_decay_ref_epochs,
    resolve_lr_decay_steps,
    resolve_warmup_ratio_ref_epoch,
    resolve_warmup_steps,
)


class LearningRateSchedulerTest(unittest.TestCase):
    def test_warmup_ratio_ref_epoch_uses_epoch_reference_steps(self):
        warmup_steps = resolve_warmup_steps(
            {
                "warmup_ratio": 0.1,
                "warmup_ratio_ref_epoch": 2.0,
            },
            total_training_steps=1000,
            steps_per_epoch=100,
        )

        self.assertEqual(warmup_steps, 20)

    def test_lr_decay_ref_epochs_uses_epoch_reference_steps(self):
        decay_steps = resolve_lr_decay_steps(
            {"lr_decay_ref_epochs": 1.5},
            total_training_steps=1000,
            warmup_steps=20,
            steps_per_epoch=100,
        )

        self.assertEqual(decay_steps, 150)

    def test_explicit_step_overrides_still_win(self):
        self.assertEqual(
            resolve_warmup_steps(
                {
                    "warmup_steps": 7,
                    "warmup_ratio": 0.1,
                    "warmup_ratio_ref_epoch": 2.0,
                },
                total_training_steps=1000,
                steps_per_epoch=100,
            ),
            7,
        )
        self.assertEqual(
            resolve_lr_decay_steps(
                {
                    "lr_decay_steps": 300,
                    "lr_decay_ref_epochs": 1.5,
                },
                total_training_steps=1000,
                warmup_steps=20,
                steps_per_epoch=100,
            ),
            300,
        )

    def test_legacy_scheduler_fields_still_load(self):
        self.assertEqual(
            resolve_warmup_steps(
                {
                    "warmup_ratio": 0.1,
                    "warmup_ratio_basis": "epoch",
                },
                total_training_steps=1000,
                steps_per_epoch=100,
            ),
            10,
        )
        self.assertEqual(resolve_lr_decay_ref_epochs({"lr_decay_epochs": 2.0}), 2.0)

    def test_reference_epochs_validate_non_negative(self):
        with self.assertRaises(ValueError):
            resolve_warmup_ratio_ref_epoch({"warmup_ratio_ref_epoch": -1})
        with self.assertRaises(ValueError):
            resolve_lr_decay_ref_epochs({"lr_decay_ref_epochs": -1})


if __name__ == "__main__":
    unittest.main()
