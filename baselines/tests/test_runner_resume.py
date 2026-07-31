from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from baselines.config import normalize_baseline_config
from baselines.runner import _load_resume_context
from baselines.trainer_state import save_trainer_state


def _config() -> dict:
    return normalize_baseline_config(
        {
            "algorithm": "mlp_bc",
            "env_family": "pointmaze",
            "train_mode": "single",
            "train_variants": ["umaze"],
            "eval_mode": "single",
            "eval_variants": ["umaze"],
            "n_steps": 2,
            "n_steps_per_epoch": 1,
            "save_interval_epochs": 1,
            "network": {
                "architecture": "residual_mlp",
                "width": 8,
                "body_depth": 4,
                "block_size": 4,
                "activation": "swish",
                "use_batch_norm": False,
                "use_layer_norm": True,
                "dropout_rate": None,
            },
            "training_diagnostics": {
                "enabled": True,
                "gradient_sample_interval_updates": 1,
                "action_probe_size": 8,
            },
        }
    )


class RunnerResumeTest(unittest.TestCase):
    def _source_run(self, directory: str, source_config: dict) -> tuple[Path, Path]:
        run_dir = Path(directory) / "source"
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        checkpoint = checkpoint_dir / "step_1.d3"
        checkpoint.write_bytes(b"checkpoint")
        trainer_state = checkpoint_dir / "trainer_state_step_1.pt"
        save_trainer_state(trainer_state, epoch=1, step=1)
        stored_config = dict(source_config)
        stored_config["n_steps"] = 1
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(stored_config), encoding="utf-8"
        )
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "n_steps": 1,
                    "training_history": [],
                    "training_diagnostics": [],
                    "evaluation_history": [],
                }
            ),
            encoding="utf-8",
        )
        return checkpoint, trainer_state

    def test_valid_resume_context_resolves_segment(self) -> None:
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, trainer_state = self._source_run(directory, config)
            config["resume"] = {
                "checkpoint": str(checkpoint),
                "trainer_state": str(trainer_state),
            }
            context = _load_resume_context(config)

        self.assertIsNotNone(context)
        self.assertEqual(context["step"], 1)
        self.assertEqual(context["epoch"], 1)
        self.assertEqual(context["segment_steps"], 1)

    def test_changed_training_recipe_is_rejected(self) -> None:
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, trainer_state = self._source_run(directory, config)
            config["network"]["width"] = 16
            config["resume"] = {
                "checkpoint": str(checkpoint),
                "trainer_state": str(trainer_state),
            }
            with self.assertRaisesRegex(ValueError, "network"):
                _load_resume_context(config)


if __name__ == "__main__":
    unittest.main()
