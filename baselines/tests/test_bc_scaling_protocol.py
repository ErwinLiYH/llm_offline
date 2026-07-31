from __future__ import annotations

import unittest
from pathlib import Path

from baselines.config import normalize_baseline_config
from utils.config_loader import load_merged_config


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "baselines/experiments/bc_scaling_20260723"


def _resolved(
    family: str,
    architecture: str,
    *,
    budget: str | None = None,
    rescue: str | None = None,
) -> dict:
    paths = [
        ROOT / f"baselines/configs/base.{family}.yaml",
        ROOT / "baselines/configs/mlp_bc.yaml",
        EXPERIMENT / f"protocol.{family}.yaml",
        EXPERIMENT / "optimizer.base.yaml",
        EXPERIMENT / "architecture" / f"{architecture}.yaml",
    ]
    if rescue is not None:
        paths.append(EXPERIMENT / "rescue" / f"{rescue}.yaml")
    if budget is not None:
        paths.append(EXPERIMENT / "budget" / f"{budget}.yaml")
    paths.append(EXPERIMENT / "seed/s0.yaml")
    return normalize_baseline_config(
        load_merged_config([str(path) for path in paths])
    )


class ScalingProtocolConfigTest(unittest.TestCase):
    def _assert_shared_protocol(
        self,
        config: dict,
        *,
        train_variants: list[str],
        test_variants: list[str],
        updates: int,
        eval_start_goal_mode: str,
    ) -> None:
        self.assertEqual(config["train_variants"], train_variants)
        self.assertEqual(
            config["eval_variants"], train_variants + test_variants
        )
        self.assertEqual(config["n_steps"], updates)
        self.assertEqual(config["n_steps_per_epoch"], 25_000)
        self.assertEqual(config["save_interval_epochs"], 4)
        self.assertEqual(config["evaluation"]["every_epochs"], 4)
        self.assertEqual(config["evaluation"]["num_episodes"], 30)
        self.assertEqual(config["evaluation"]["rollout_batch_size"], 30)
        self.assertEqual(config["evaluation"]["seed"], 0)
        self.assertEqual(
            config["evaluation"]["env_config"]["eval_start_goal_mode"],
            eval_start_goal_mode,
        )
        self.assertEqual(config["algorithm_config"]["batch_size"], 256)
        self.assertEqual(config["reward_type"], "sparse")
        self.assertFalse(config["allow_mixed_reward_types"])
        self.assertEqual(
            config["observation"],
            {
                "include_map": True,
                "include_location_sensing": True,
                "include_wall_sensing": True,
                "wall_sensing_version": "v3",
                "map_sensing_boundary_risk_threshold": 0.1,
            },
        )
        self.assertEqual(config["sampling_seed"], 0)
        self.assertEqual(config["train_data_ratio"], 0.9)
        self.assertTrue(config["balance_variant_episode_count"])
        self.assertEqual(config["episode_keep_num"], 300)

    def test_point_core_grid_preserves_16_train_plus_6_test_protocol(self) -> None:
        train_variants = [
            "open",
            "umaze",
            "medium",
            "large",
            *[f"local-layoutV2-{index:02d}" for index in range(1, 13)],
        ]
        test_variants = [
            f"test-layoutV2-{index:02d}" for index in range(1, 7)
        ]
        recipes = {
            "plain_d4_w256": (3e-4, None),
            "plain_d4_w512": (3e-4, None),
            "plain_d4_w1024": (3e-4, None),
            "plain_d4_w2048": (3e-4, None),
            "plain_d4_w3072": (3e-4, None),
            "plain_d4_w4096": (3e-4, None),
            "residual_d4_w256": (3e-4, None),
            "residual_d16_w256": (3e-4, None),
            "residual_d64_w256": (1e-4, "d64_lr1e4"),
            "residual_d128_w256": (3e-4, None),
            "residual_d256_w256": (1e-5, "d256_lr1e5"),
        }
        for architecture, (learning_rate, rescue) in recipes.items():
            with self.subTest(architecture=architecture):
                config = _resolved(
                    "pointmaze", architecture, rescue=rescue
                )
                self._assert_shared_protocol(
                    config,
                    train_variants=train_variants,
                    test_variants=test_variants,
                    updates=500_000,
                    eval_start_goal_mode="random-start-goal",
                )
                self.assertEqual(
                    config["algorithm_config"]["learning_rate"],
                    learning_rate,
                )

    def test_ant_core_grid_preserves_16_train_plus_4_test_1m_protocol(self) -> None:
        train_variants = [
            "umaze",
            "medium-diverse",
            "large-diverse",
            "ultra",
            *[f"local-layout-{index:02d}" for index in range(1, 13)],
        ]
        test_variants = [
            f"test-layout-{index:02d}" for index in range(1, 5)
        ]
        architectures = [
            "plain_d4_w256",
            "plain_d4_w512",
            "plain_d4_w1024",
            "plain_d4_w2048",
            "plain_d4_w3072",
            "plain_d4_w4096",
            "residual_d4_w256",
            "residual_d16_w256",
            "residual_d64_w256",
            "residual_d128_w256",
            "residual_d256_w256",
        ]
        for architecture in architectures:
            with self.subTest(architecture=architecture):
                config = _resolved(
                    "antmaze", architecture, budget="u1m"
                )
                self._assert_shared_protocol(
                    config,
                    train_variants=train_variants,
                    test_variants=test_variants,
                    updates=1_000_000,
                    eval_start_goal_mode="fix-start-goal",
                )
                self.assertEqual(
                    config["algorithm_config"]["learning_rate"], 3e-4
                )

    def test_point_w7168_preserves_protocol_with_inherited_range_lr(self) -> None:
        config = _resolved(
            "pointmaze",
            "plain_d4_w7168",
            rescue="wide_lr3e5",
        )
        self._assert_shared_protocol(
            config,
            train_variants=[
                "open",
                "umaze",
                "medium",
                "large",
                *[
                    f"local-layoutV2-{index:02d}"
                    for index in range(1, 13)
                ],
            ],
            test_variants=[
                f"test-layoutV2-{index:02d}" for index in range(1, 7)
            ],
            updates=500_000,
            eval_start_goal_mode="random-start-goal",
        )
        self.assertEqual(config["network"]["width"], 7168)
        self.assertEqual(config["network"]["body_depth"], 4)
        self.assertEqual(config["algorithm_config"]["learning_rate"], 3e-5)

    def test_point_w8192_preserves_protocol_with_inherited_range_lr(self) -> None:
        config = _resolved(
            "pointmaze",
            "plain_d4_w8192",
            rescue="wide_lr3e5",
        )
        self._assert_shared_protocol(
            config,
            train_variants=[
                "open",
                "umaze",
                "medium",
                "large",
                *[
                    f"local-layoutV2-{index:02d}"
                    for index in range(1, 13)
                ],
            ],
            test_variants=[
                f"test-layoutV2-{index:02d}" for index in range(1, 7)
            ],
            updates=500_000,
            eval_start_goal_mode="random-start-goal",
        )
        self.assertEqual(config["network"]["width"], 8192)
        self.assertEqual(config["network"]["body_depth"], 4)
        self.assertEqual(config["algorithm_config"]["learning_rate"], 3e-5)


if __name__ == "__main__":
    unittest.main()
