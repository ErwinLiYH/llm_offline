"""Evaluation entry point: rollout the fine-tuned policy in gymnasium environments.

Usage:
    python evaluate.py --config eval.yaml
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  registers PointMaze envs

from data.registry import get_formatter
from data.pointmaze.variants import POINTMAZE_VARIANTS
from model.policy import load_from_checkpoint
from utils.prompt_loader import load_templates


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="eval.yaml")
    return parser.parse_args()


def get_results_dir(model_path: str) -> str:
    """Mirror checkpoints/ → results/ directory structure."""
    return model_path.replace("checkpoints/", "results/", 1)


def generate_action(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 20,
) -> str:
    """Run inference and return the generated text (action portion)."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def evaluate_variant(
    config: dict,
    variant: str,
    model,
    tokenizer,
    device: torch.device,
    template: str,
) -> dict:
    formatter = get_formatter(config["env_family"])
    meta = POINTMAZE_VARIANTS[variant]
    env_id = meta["env_id"]
    num_episodes = config["num_episodes"]
    parse_retry_limit = config["parse_retry_limit"]

    env = gym.make(env_id)

    episode_returns = []
    episode_successes = []
    total_parse_failures = 0
    total_fallbacks = 0

    for ep_idx in range(num_episodes):
        obs_dict, info = env.reset()
        obs = obs_dict["observation"].astype(np.float32)
        goal = obs_dict["desired_goal"].astype(np.float32)

        ep_return = 0.0
        ep_success = False
        terminated = False
        truncated = False

        while not (terminated or truncated):
            obs_text = formatter.format_obs(obs, goal)
            prompt = template.format(obs_text=obs_text)

            # Try to get a valid action, retrying on parse/validate failure
            action = None
            for attempt in range(parse_retry_limit + 1):
                generated = generate_action(model, tokenizer, prompt, device)
                parsed_action, success = formatter.parse_action(generated)
                if success and formatter.validate_action(parsed_action):
                    action = np.clip(parsed_action, -1.0, 1.0)
                    break
                total_parse_failures += 1

            if action is None:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
                total_fallbacks += 1

            obs_dict, reward, terminated, truncated, info = env.step(action)
            obs = obs_dict["observation"].astype(np.float32)
            goal = obs_dict["desired_goal"].astype(np.float32)
            ep_return += float(reward)

            if terminated:
                ep_success = True

        episode_returns.append(ep_return)
        episode_successes.append(ep_success)

        if (ep_idx + 1) % 5 == 0:
            print(
                f"  [{variant}] episode {ep_idx+1}/{num_episodes} | "
                f"return={ep_return:.2f} | success={ep_success}"
            )

    env.close()

    return {
        "variant": variant,
        "num_episodes": num_episodes,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "success_rate": float(np.mean(episode_successes)),
        "total_parse_failures": total_parse_failures,
        "total_fallbacks": total_fallbacks,
    }


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Using device: {device}")
    print(f"[eval] Loading model from: {config['model_path']}")

    model, tokenizer = load_from_checkpoint(config["model_path"])
    model.to(device)
    model.eval()

    env_family = config["env_family"]
    variant_arg = config["variant"]

    if variant_arg == "all":
        variants_to_eval = list(POINTMAZE_VARIANTS.keys())
    else:
        variants_to_eval = [variant_arg]

    all_results = []
    for variant in variants_to_eval:
        print(f"\n[eval] Evaluating variant: {variant}")
        # Template 0 is always used for evaluation (first English template)
        templates = load_templates(env_family, variant)
        template = templates[0]

        result = evaluate_variant(config, variant, model, tokenizer, device, template)
        all_results.append(result)
        print(
            f"[eval] {variant}: mean_return={result['mean_return']:.4f}, "
            f"success_rate={result['success_rate']:.2%}, "
            f"parse_failures={result['total_parse_failures']}, "
            f"fallbacks={result['total_fallbacks']}"
        )

    results_dir = get_results_dir(config["model_path"])
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[eval] Results saved to: {results_path}")


if __name__ == "__main__":
    main()
