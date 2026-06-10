import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import evaluate
from utils.distributed import DistributedContext
from utils.eval_parallel import (
    assigned_eval_variants,
    eval_variant_assignments,
    resolve_eval_parallel_episodes,
)
from utils.eval_rollout import (
    ActionRolloutContext,
    generate_valid_continuous_actions_batch,
)


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
        self.assertEqual(resolve_eval_parallel_episodes({}), 1)
        with self.assertRaises(ValueError):
            resolve_eval_parallel_episodes({"eval_parallel_episodes": 0})

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

    def test_batched_episode_slots_are_reused(self):
        model = DummyContinuousModel()
        config = {
            "env_family": "pointmaze",
            "num_episodes": 5,
            "parse_retry_limit": 0,
            "seed": 10,
            "history_num": 0,
            "history_stride": 1,
            "record_video": False,
            "record_step_logs": False,
            "action_token_mode": "parallel_l1",
            "action_dim": 2,
            "action_sampling": False,
            "eval_parallel_episodes": 3,
        }
        with (
            mock.patch.object(
                evaluate,
                "_resolve_variant_env_spec",
                return_value=({"prompt_vars": {}}, "Dummy-v0", {}),
            ),
            mock.patch.object(evaluate, "get_formatter", return_value=DummyFormatter()),
            mock.patch.object(evaluate, "render_policy_prompt", return_value="prompt"),
            mock.patch.object(evaluate.gym, "make", side_effect=lambda *args, **kwargs: DummyEnv()),
        ):
            result = evaluate.evaluate_variant(
                config,
                "dummy",
                model,
                DummyTokenizer(),
                torch.device("cpu"),
                "template",
            )

        self.assertEqual(result["episode_seeds"], [10, 11, 12, 13, 14])
        self.assertEqual(result["eval_parallel_episodes_used"], 3)
        self.assertEqual(result["mean_episode_steps"], 1.4)
        self.assertEqual(model.batch_sizes[0], 3)
        self.assertTrue(any(batch_size < 3 for batch_size in model.batch_sizes))


if __name__ == "__main__":
    unittest.main()
