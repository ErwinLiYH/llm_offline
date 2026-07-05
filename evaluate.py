"""Evaluation entry point: rollout the fine-tuned policy in gymnasium environments.

Usage:
    python evaluate.py --config eval.yaml
    python evaluate.py --config eval.yaml --model_path checkpoints/.../final
    python evaluate.py --config eval.yaml --model_path 'checkpoints/.../ep7*'
"""

import argparse
import json
import os
import uuid
import warnings

import torch
import yaml

from model.continuous_action import (
    resolve_action_head_dropout,
    resolve_action_head_num_blocks,
    resolve_action_query_len,
    resolve_gaussian_log_std_bounds,
    resolve_gaussian_log_std_init,
    resolve_student_t_df,
)
from model.mtp_bin import resolve_mtp_k
from model.policy import load_from_checkpoint
from utils.action_bins import (
    get_action_bin_range,
    get_action_num_bins,
    get_action_token_mode,
)
from utils.config_loader import load_merged_config
from utils.distributed import (
    all_gather_objects,
    barrier,
    broadcast_object,
    cleanup_distributed,
    init_distributed_context,
    resolve_parallel_backend,
)
from utils.eval_parallel import (
    apply_rollout_config_defaults,
    assigned_eval_variants,
    eval_variant_assignments,
    resolve_eval_distribute_variants,
    resolve_rollout_worker_lifetime,
    resolve_rollout_worker_num,
)
from utils.model_path_glob import resolve_trailing_wildcard_path
from utils.prompt_loader import load_named_templates, load_template_names
from utils.rollout.evaluate_runner import run_evaluate_variant
from utils.sensing_config import apply_checkpoint_sensing_config as _apply_checkpoint_sensing_config
from utils.training_tags import format_training_eval_tag
from utils.variant_selection import get_available_variants, resolve_selection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", default=["eval.yaml"])
    parser.add_argument(
        "--model_path",
        metavar="PATH",
        default=None,
        help="Override model_path from the merged eval config. A single trailing '*' wildcard is supported.",
    )
    parser.add_argument("--parallel_backend", type=str, choices=["single", "ddp"], default=None)
    parser.add_argument("-y", "--yes", action="store_true", help="Automatically confirm strong warnings.")
    return parser.parse_args()


def resolve_eval_model_path(config: dict) -> dict:
    if "model_path" not in config:
        return config
    resolved_model_path = resolve_trailing_wildcard_path(
        config["model_path"],
        field_name="model_path",
    )
    if resolved_model_path == config["model_path"]:
        return config
    resolved = dict(config)
    resolved["model_path_pattern"] = config["model_path"]
    resolved["model_path"] = resolved_model_path
    return resolved



def resolve_standalone_eval_selection(config: dict):
    available_variants = get_available_variants(config["env_family"])

    eval_mode = config.get("eval_mode")
    variants = config.get("variants")
    legacy_variant = config.get("variant")

    if eval_mode is None and variants is None and legacy_variant is not None:
        if legacy_variant == "all":
            eval_mode = "all"
            variants = []
        else:
            eval_mode = "single"
            variants = [legacy_variant]

    resolved_eval_mode = eval_mode or "single"
    selection = resolve_selection(
        mode=resolved_eval_mode,
        variants=variants,
        available_variants=available_variants,
        field_name="variants",
    )
    return selection


ACTION_CONFIG_KEYS = (
    "action_token_mode",
    "action_num_bins",
    "action_bin_min",
    "action_bin_max",
    "new_token",
    "mtp_k",
    "mtp_lcm_weight",
    "action_soft_label_sigma",
    "action_loss_weight",
    "action_stop_loss_weight",
    "action_dim",
    "action_query_len",
    "action_head_num_blocks",
    "action_head_dropout",
    "gaussian_log_std_min",
    "gaussian_log_std_max",
    "gaussian_log_std_init",
    "student_t_df",
    "max_length",
)


def _load_checkpoint_config(model_path: str) -> dict:
    config_path = os.path.join(model_path, "config.yaml")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _load_checkpoint_action_config(model_path: str) -> dict:
    saved_config = _load_checkpoint_config(model_path)

    action_config = {
        "action_token_mode": saved_config.get("action_token_mode", "text"),
        "action_num_bins": saved_config.get("action_num_bins", 10),
        "action_bin_min": saved_config.get("action_bin_min", -1.0),
        "action_bin_max": saved_config.get("action_bin_max", 1.0),
        "new_token": saved_config.get("new_token", False),
        "mtp_k": saved_config.get("mtp_k"),
        "mtp_lcm_weight": saved_config.get("mtp_lcm_weight", 1.0),
    }
    for key in ("action_soft_label_sigma", "action_loss_weight", "action_stop_loss_weight"):
        if key in saved_config:
            action_config[key] = saved_config[key]
    if "action_dim" in saved_config:
        action_config["action_dim"] = saved_config["action_dim"]
    if "max_length" in saved_config:
        action_config["max_length"] = saved_config["max_length"]
    if action_config["action_token_mode"] in {
        "mtp_bin",
        "simple_mtp_bin",
        "parallel_l1",
        "parallel_gaussian",
        "parallel_t",
    }:
        if "action_dim" not in action_config:
            raise ValueError(
                "Checkpoint config.yaml uses a parallel action mode but does not contain action_dim."
            )
    if action_config["action_token_mode"] == "mtp_bin":
        action_config["mtp_k"] = resolve_mtp_k(
            int(action_config["action_dim"]),
            saved_config.get("mtp_k"),
        )
    if action_config["action_token_mode"] == "simple_mtp_bin":
        action_config.pop("mtp_k", None)
    if action_config["action_token_mode"] in {"parallel_l1", "parallel_gaussian", "parallel_t"}:
        action_config["action_query_len"] = resolve_action_query_len(
            int(action_config["action_dim"]),
            saved_config.get("action_query_len"),
        )
        action_config["action_head_num_blocks"] = resolve_action_head_num_blocks(
            saved_config.get("action_head_num_blocks")
        )
        action_config["action_head_dropout"] = resolve_action_head_dropout(
            saved_config.get("action_head_dropout")
        )
        if action_config["action_token_mode"] in {"parallel_gaussian", "parallel_t"}:
            gaussian_log_std_min, gaussian_log_std_max = resolve_gaussian_log_std_bounds(
                saved_config
            )
            action_config["gaussian_log_std_min"] = gaussian_log_std_min
            action_config["gaussian_log_std_max"] = gaussian_log_std_max
        if action_config["action_token_mode"] == "parallel_gaussian":
            gaussian_log_std_init = resolve_gaussian_log_std_init(saved_config)
            action_config["gaussian_log_std_init"] = max(
                action_config["gaussian_log_std_min"],
                min(gaussian_log_std_init, action_config["gaussian_log_std_max"]),
            )
        if action_config["action_token_mode"] == "parallel_t":
            action_config["student_t_df"] = resolve_student_t_df(saved_config)
    return action_config


def apply_checkpoint_action_config(config: dict) -> dict:
    action_config = _load_checkpoint_action_config(config["model_path"])
    saved_config = _load_checkpoint_config(config["model_path"])
    for key in ACTION_CONFIG_KEYS:
        if key in config and config[key] != action_config.get(key):
            raise ValueError(
                f"Standalone eval action config must come from checkpoint config.yaml; "
                f"remove {key}={config[key]!r} from eval.yaml or match checkpoint value {action_config.get(key)!r}."
            )
    merged = dict(config)
    merged.update(action_config)
    if "mtp_quadratic_decoding" not in merged:
        merged["mtp_quadratic_decoding"] = saved_config.get("mtp_quadratic_decoding", True)
    get_action_token_mode(merged)
    get_action_num_bins(merged)
    get_action_bin_range(merged)
    return merged


def apply_checkpoint_sensing_config(config: dict) -> dict:
    saved_config = _load_checkpoint_config(config["model_path"])
    return _apply_checkpoint_sensing_config(config, saved_config)


def _normalize_prompt_name_list(value, *, field_name: str, allow_single_string: bool) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if not allow_single_string:
            raise ValueError(f"{field_name} must be a list of prompt names, got str")
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of prompt names, got {type(value).__name__}")

    names = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} must contain non-empty strings, got {item!r}")
        names.append(item.strip())
    if not names:
        raise ValueError(f"{field_name} must not be empty")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{field_name} contains duplicate prompt names: {duplicates}")
    return names


def _resolve_prompt_config_values(config: dict, *, allow_single_string: bool) -> list[str] | None:
    primary_key = "prompt_templete_index"
    legacy_key = "prompt_template_index"
    primary_present = primary_key in config
    legacy_present = legacy_key in config
    primary_value = config.get(primary_key)
    legacy_value = config.get(legacy_key)
    if primary_present and legacy_present and primary_value != legacy_value:
        raise ValueError(
            f"{primary_key} and {legacy_key} both exist but differ; keep only {primary_key}."
        )
    if not primary_present and not legacy_present:
        return None
    field_name = primary_key if primary_present else legacy_key
    raw_value = primary_value if primary_present else legacy_value
    return _normalize_prompt_name_list(
        raw_value,
        field_name=field_name,
        allow_single_string=allow_single_string,
    )


def _confirm_strong_warning(message: str, *, assume_yes: bool):
    print(f"\n[eval] STRONG WARNING: {message}")
    if assume_yes:
        print("[eval] -y/--yes supplied; continuing despite the warning.")
        return
    try:
        response = input("[eval] Type uppercase Y to continue: ")
    except EOFError as exc:
        raise SystemExit("[eval] Aborted: confirmation required but stdin is unavailable.") from exc
    if response != "Y":
        raise SystemExit("[eval] Aborted by user.")


def apply_checkpoint_prompt_config(config: dict, *, assume_yes: bool) -> dict:
    saved_config = _load_checkpoint_config(config["model_path"])
    train_prompt_names = _resolve_prompt_config_values(saved_config, allow_single_string=False)
    eval_prompt_names = _resolve_prompt_config_values(config, allow_single_string=True)

    if eval_prompt_names is not None:
        if len(eval_prompt_names) != 1:
            raise ValueError(
                "Standalone eval accepts exactly one prompt in prompt_templete_index; "
                f"got {eval_prompt_names}."
            )
        eval_prompt_name = eval_prompt_names[0]
        if train_prompt_names and eval_prompt_name not in train_prompt_names:
            _confirm_strong_warning(
                "eval prompt is outside the checkpoint training prompt list. "
                f"train_prompts={train_prompt_names}, eval_prompt={eval_prompt_name!r}. "
                "This changes the prompt distribution used for rollout.",
                assume_yes=assume_yes,
            )
    elif train_prompt_names:
        eval_prompt_name = train_prompt_names[0]
    else:
        warnings.warn(
            "Checkpoint config.yaml does not contain prompt_templete_index; "
            "standalone eval will use the first prompt template in filename order.",
            RuntimeWarning,
            stacklevel=2,
        )
        eval_prompt_name = None

    merged = dict(config)
    if train_prompt_names:
        merged["checkpoint_prompt_templete_index"] = train_prompt_names
    if eval_prompt_name is not None:
        merged["prompt_templete_index"] = [eval_prompt_name]
    merged.pop("prompt_template_index", None)
    merged["resolved_eval_prompt_name"] = eval_prompt_name
    return merged



def get_results_base_dir(config: dict) -> str:
    """Build the base results directory from model/training context only."""
    from model.policy import get_model_slug

    model_path = config["model_path"]
    result_root = config.get("result_root", "results")
    norm_path = model_path.replace("\\", "/").rstrip("/")
    parts = [part for part in norm_path.split("/") if part]

    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        ckpt_parts = parts[idx + 1 :]
        if len(ckpt_parts) >= 5:
            env_family, model_slug, train_selection_tag, experiment_id, _checkpoint_tag = ckpt_parts[-5:]
            train_tag = f"train={env_family}-{train_selection_tag}"
            exp_tag = f"exp={experiment_id}"
            return os.path.join(result_root, model_slug, train_tag, exp_tag)

    model_slug = get_model_slug(model_path)
    train_tag = "train=pretrained"
    return os.path.join(result_root, model_slug, train_tag)


def get_standalone_results_dir(base_results_dir: str, standalone_eval_id: str) -> str:
    return os.path.join(base_results_dir, f"standalone_{standalone_eval_id}")


def resolve_eval_output_mode(config: dict) -> str:
    mode = str(config.get("eval_output_mode", "standalone")).strip().lower()
    if mode not in {"standalone", "training"}:
        raise ValueError(
            "eval_output_mode must be 'standalone' or 'training', "
            f"got {mode!r}"
        )
    return mode


TRAINING_EVAL_CONTEXT_KEYS = (
    "eval_type",
    "epoch",
    "batch_step",
    "epoch_step",
    "optimizer_step",
    "scheduled_step",
    "scheduled_epoch_step",
    "train_loss",
    "val_loss",
    "val_metrics",
    "checkpoint_path",
    "experiment_id",
)


def resolve_training_eval_context(config: dict) -> dict:
    context = config.get("training_eval_context")
    if not isinstance(context, dict):
        raise ValueError(
            "training_eval_context must be provided as a mapping when "
            "eval_output_mode='training'"
        )

    missing = [key for key in TRAINING_EVAL_CONTEXT_KEYS if key not in context]
    if missing:
        raise ValueError(f"training_eval_context is missing required keys: {missing}")

    eval_type = context.get("eval_type")
    if eval_type not in {"epoch", "step"}:
        raise ValueError(
            "training_eval_context.eval_type must be 'epoch' or 'step', "
            f"got {eval_type!r}"
        )
    if eval_type == "epoch" and context.get("epoch") is None:
        raise ValueError("training_eval_context.epoch is required for epoch eval")
    if eval_type == "step" and context.get("batch_step") is None:
        raise ValueError("training_eval_context.batch_step is required for step eval")

    resolved = dict(context)
    if not isinstance(resolved.get("val_metrics"), dict):
        raise ValueError("training_eval_context.val_metrics must be a mapping")
    return resolved


def get_training_eval_tag(training_eval_context: dict) -> str:
    return format_training_eval_tag(
        training_eval_context["eval_type"],
        epoch=training_eval_context.get("epoch"),
        batch_step=training_eval_context.get("batch_step"),
    )


def get_training_results_dir(base_results_dir: str, training_eval_context: dict) -> str:
    return os.path.join(base_results_dir, get_training_eval_tag(training_eval_context))


def apply_training_eval_context_to_result(
    result: dict,
    training_eval_context: dict,
) -> dict:
    val_metrics = training_eval_context.get("val_metrics") or {}
    result["train_loss"] = training_eval_context.get("train_loss")
    result["val_loss"] = training_eval_context.get("val_loss")
    result["val_metrics"] = val_metrics
    if "mae" in val_metrics:
        result["val_mae"] = val_metrics["mae"]
    result["experiment_id"] = training_eval_context.get("experiment_id")
    result["eval_type"] = training_eval_context.get("eval_type")
    result["eval_tag"] = get_training_eval_tag(training_eval_context)
    result["epoch"] = training_eval_context.get("epoch")
    result["batch_step"] = training_eval_context.get("batch_step")
    result["epoch_step"] = training_eval_context.get("epoch_step")
    result["optimizer_step"] = training_eval_context.get("optimizer_step")
    result["scheduled_step"] = training_eval_context.get("scheduled_step")
    result["scheduled_epoch_step"] = training_eval_context.get("scheduled_epoch_step")
    result["checkpoint_path"] = training_eval_context.get("checkpoint_path")
    return result


def get_variant_results_dir(parent_results_dir: str, env_family: str, variant: str) -> str:
    return os.path.join(parent_results_dir, f"eval={env_family}-{variant}")


def configure_mujoco_gl(config: dict):
    mujoco_gl = config.get("mujoco_gl")
    if mujoco_gl:
        os.environ["MUJOCO_GL"] = str(mujoco_gl)
        return

    if config.get("record_video", False):
        os.environ.setdefault("MUJOCO_GL", "egl")


def evaluate_variant(
    config: dict,
    variant: str,
    model,
    tokenizer,
    device: torch.device,
    template: str,
    variant_results_dir: str | None = None,
) -> dict:
    num_episodes = int(config["num_episodes"])
    if num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")
    return run_evaluate_variant(
        config=config,
        variant=variant,
        model=model,
        tokenizer=tokenizer,
        device=device,
        template=template,
        variant_results_dir=variant_results_dir,
    )



def main():
    args = parse_args()
    config = load_merged_config(args.config)
    if args.model_path is not None:
        config["model_path"] = args.model_path
    config = resolve_eval_model_path(config)
    parallel_backend = resolve_parallel_backend(config, args.parallel_backend)
    dist_context = init_distributed_context(config, parallel_backend)
    try:
        if dist_context.is_main_process:
            config = apply_checkpoint_action_config(config)
            config = apply_checkpoint_sensing_config(config)
            config = apply_checkpoint_prompt_config(config, assume_yes=args.yes)
            config.setdefault("seed", 1)
            config = apply_rollout_config_defaults(config)
        else:
            config = None
        config = broadcast_object(config, dist_context)

        eval_output_mode = resolve_eval_output_mode(config)
        training_eval_context = (
            resolve_training_eval_context(config)
            if eval_output_mode == "training"
            else None
        )
        eval_selection = resolve_standalone_eval_selection(config)
        distribute_variants = resolve_eval_distribute_variants(config)
        rollout_worker_num = resolve_rollout_worker_num(config)
        rollout_worker_lifetime = resolve_rollout_worker_lifetime(config)
        assignments = eval_variant_assignments(
            eval_selection.selected_variants,
            dist_context,
            distribute_variants=distribute_variants,
        )
        local_variants = assigned_eval_variants(
            eval_selection.selected_variants,
            dist_context,
            distribute_variants=distribute_variants,
        )
        configure_mujoco_gl(config)

        device = dist_context.device
        if dist_context.is_main_process:
            print(f"[eval] Using backend: {dist_context.backend}")
            print(f"[eval] Output mode: {eval_output_mode}")
            if "model_path_pattern" in config:
                print(
                    "[eval] Resolved model_path wildcard: "
                    f"{config['model_path_pattern']} -> {config['model_path']}"
                )
            print(f"[eval] Loading model from: {config['model_path']}")
            print(
                "[eval] Wall sensing: "
                f"version={config['wall_sensing_version']}, "
                f"boundary_risk_threshold={config['map_sensing_boundary_risk_threshold']}"
            )
            print(f"[eval] Resolved eval variants: {eval_selection.selected_variants}")
            print(f"[eval] Variant assignments: {assignments}")
            print(
                "[eval] Rollout workers per rank: "
                f"{rollout_worker_num} (lifetime={rollout_worker_lifetime})"
            )

        model, tokenizer = load_from_checkpoint(
            config["model_path"],
            load_in_4bit=config.get("load_in_4bit"),
            runtime_config=config,
        )
        model.to(device)
        model.eval()

        env_family = config["env_family"]
        base_results_dir = get_results_base_dir(config)
        standalone_eval_id = None
        if eval_output_mode == "standalone":
            standalone_eval_id = (
                uuid.uuid4().hex[:8]
                if dist_context.is_main_process
                else None
            )
            standalone_eval_id = broadcast_object(
                standalone_eval_id,
                dist_context,
            )
            run_results_dir = get_standalone_results_dir(
                base_results_dir,
                standalone_eval_id,
            )
            if dist_context.is_main_process:
                print(f"[eval] Eval ID: {standalone_eval_id}")
        else:
            run_results_dir = get_training_results_dir(
                base_results_dir,
                training_eval_context,
            )
            if dist_context.is_main_process:
                print(
                    f"[eval] Training eval tag: "
                    f"{get_training_eval_tag(training_eval_context)}"
                )
        prompt_name = config.get("resolved_eval_prompt_name")
        if prompt_name is None:
            prompt_name = load_template_names(env_family)[0]
            config["resolved_eval_prompt_name"] = prompt_name
        template = load_named_templates(env_family, [prompt_name])[0]
        config["eval_config_source"] = (
            args.config[0] if len(args.config) == 1 else list(args.config)
        )
        config["config_sources"] = list(args.config)
        config["eval_output_mode"] = eval_output_mode
        if eval_output_mode == "standalone":
            config["standalone_eval_id"] = standalone_eval_id
            config["standalone_results_dir"] = run_results_dir
        else:
            config["training_eval_context"] = training_eval_context
            config["training_eval_tag"] = get_training_eval_tag(training_eval_context)
            config["training_results_dir"] = run_results_dir
        config["resolved_eval_variants"] = eval_selection.selected_variants
        config["resolved_eval_selection_tag"] = eval_selection.selection_tag
        config["resolved_eval_selection_tag_full"] = eval_selection.full_selection_tag
        config["resolved_eval_variant_assignments"] = assignments
        config["eval_world_size"] = (
            training_eval_context.get("eval_world_size", dist_context.world_size)
            if training_eval_context is not None
            else dist_context.world_size
        )
        config["rollout_worker_num"] = rollout_worker_num
        config["rollout_worker_lifetime"] = rollout_worker_lifetime
        config["eval_distribute_variants"] = distribute_variants
        eval_config_path = os.path.join(run_results_dir, "eval_config.yaml")
        if dist_context.is_main_process:
            os.makedirs(run_results_dir, exist_ok=True)
            with open(eval_config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
            print(f"[eval] Eval config saved to: {eval_config_path}")
        barrier(dist_context)

        local_results = []
        for variant in local_variants:
            print(
                f"\n[eval][rank {dist_context.rank}] "
                f"Evaluating variant: {variant}"
            )
            results_dir = get_variant_results_dir(
                run_results_dir,
                env_family,
                variant,
            )
            os.makedirs(results_dir, exist_ok=True)
            result_path = os.path.join(results_dir, "result.json")

            result = evaluate_variant(
                config,
                variant,
                model,
                tokenizer,
                device,
                template,
                variant_results_dir=results_dir,
            )
            result["result_path"] = result_path
            result["prompt_template_name"] = prompt_name
            if training_eval_context is not None:
                apply_training_eval_context_to_result(result, training_eval_context)
            result["eval_rank"] = (
                training_eval_context.get("eval_rank", dist_context.rank)
                if training_eval_context is not None
                else dist_context.rank
            )
            result["eval_world_size"] = (
                training_eval_context.get("eval_world_size", dist_context.world_size)
                if training_eval_context is not None
                else dist_context.world_size
            )
            result["eval_distribute_variants"] = (
                training_eval_context.get("eval_distribute_variants", distribute_variants)
                if training_eval_context is not None
                else distribute_variants
            )
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print(
                f"[eval][rank {dist_context.rank}] {variant}: "
                f"mean_return={result['mean_return']:.4f}, "
                f"success_rate={result['success_rate']:.2%}, "
                f"parse_failures={result['total_parse_failures']}, "
                f"fallbacks={result['total_fallbacks']}"
            )
            print(
                f"[eval][rank {dist_context.rank}] "
                f"Results saved to: {result_path}"
            )
            local_results.append(result)

        gathered_results = all_gather_objects(local_results, dist_context)
        if dist_context.is_main_process:
            results_by_variant = {
                result["variant"]: result
                for rank_results in gathered_results
                for result in rank_results
            }
            print("\n[eval] Completed variants:")
            for variant in eval_selection.selected_variants:
                result = results_by_variant[variant]
                print(
                    f"  {variant}: success_rate={result['success_rate']:.2%}, "
                    f"mean_return={result['mean_return']:.4f}, "
                    f"rank={result['eval_rank']}"
                )
    finally:
        cleanup_distributed(dist_context)


if __name__ == "__main__":
    main()
