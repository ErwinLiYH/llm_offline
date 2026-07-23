from __future__ import annotations

import copy
import math
import os
from collections.abc import Mapping

from crossmaze.reward import normalize_reward_type
from crossmaze.sensing_config import resolve_sensing_config

from baselines import SUPPORTED_ALGORITHMS


_TOP_LEVEL_KEYS = {
    "algorithm",
    "env_family",
    "train_mode",
    "train_variants",
    "eval_mode",
    "eval_variants",
    "reward_type",
    "allow_mixed_reward_types",
    "local_dataset_root",
    "episode_keep_num",
    "episode_keep_per_variant",
    "balance_variant_episode_count",
    "sampling_seed",
    "train_data_ratio",
    "seed",
    "device",
    "n_steps",
    "n_steps_per_epoch",
    "save_interval_epochs",
    "show_progress",
    "output_root",
    "experiment_id",
    "observation",
    "network",
    "algorithm_config",
    "evaluation",
    "logging",
}

_COMMON_ALGORITHM_FIELDS = {"batch_size", "compile_graph"}
_Q_ALGORITHM_FIELDS = _COMMON_ALGORITHM_FIELDS | {"gamma"}
_ALGORITHM_FIELDS = {
    "mlp_bc": _COMMON_ALGORITHM_FIELDS
    | {"learning_rate", "policy_type"},
    "td3_bc": _Q_ALGORITHM_FIELDS
    | {
        "actor_learning_rate",
        "critic_learning_rate",
        "tau",
        "n_critics",
        "target_smoothing_sigma",
        "target_smoothing_clip",
        "alpha",
        "update_actor_interval",
        "reward_scaler",
    },
    "iql": _Q_ALGORITHM_FIELDS
    | {
        "actor_learning_rate",
        "critic_learning_rate",
        "tau",
        "n_critics",
        "expectile",
        "weight_temp",
        "max_weight",
        "reward_scaler",
    },
    "crl": {
        "batch_size",
        "learning_rate",
        "discount",
        "latent_dim",
        "alpha",
        "const_std",
        "normalize_observations",
        "value_p_curgoal",
        "value_p_trajgoal",
        "value_p_randomgoal",
        "value_geom_sample",
        "actor_p_curgoal",
        "actor_p_trajgoal",
        "actor_p_randomgoal",
        "actor_geom_sample",
    },
    "hiql": {
        "batch_size",
        "learning_rate",
        "discount",
        "tau",
        "expectile",
        "low_alpha",
        "high_alpha",
        "subgoal_steps",
        "rep_dim",
        "low_actor_rep_grad",
        "const_std",
        "normalize_observations",
        "value_p_curgoal",
        "value_p_trajgoal",
        "value_p_randomgoal",
        "value_geom_sample",
        "actor_p_curgoal",
        "actor_p_trajgoal",
        "actor_p_randomgoal",
        "actor_geom_sample",
    },
}

_GCRL_DEFAULTS = {
    "crl": {
        "batch_size": 1024,
        "learning_rate": 3e-4,
        "discount": 0.99,
        "latent_dim": 512,
        "alpha": 0.1,
        "const_std": True,
        "normalize_observations": True,
        "value_p_curgoal": 0.0,
        "value_p_trajgoal": 1.0,
        "value_p_randomgoal": 0.0,
        "value_geom_sample": True,
        "actor_p_curgoal": 0.0,
        "actor_p_trajgoal": 1.0,
        "actor_p_randomgoal": 0.0,
        "actor_geom_sample": False,
    },
    "hiql": {
        "batch_size": 1024,
        "learning_rate": 3e-4,
        "discount": 0.99,
        "tau": 0.005,
        "expectile": 0.7,
        "low_alpha": 3.0,
        "high_alpha": 3.0,
        "subgoal_steps": 25,
        "rep_dim": 10,
        "low_actor_rep_grad": False,
        "const_std": True,
        "normalize_observations": True,
        "value_p_curgoal": 0.2,
        "value_p_trajgoal": 0.5,
        "value_p_randomgoal": 0.3,
        "value_geom_sample": True,
        "actor_p_curgoal": 0.0,
        "actor_p_trajgoal": 1.0,
        "actor_p_randomgoal": 0.0,
        "actor_geom_sample": False,
    },
}

_NETWORK_DEFAULTS = {
    # legacy_mlp preserves the existing d3rlpy VectorEncoderFactory path.
    # The two scaling bodies use the paper-style Dense -> LayerNorm -> Swish
    # encoder defined in baselines.algorithms.scaling_mlp.
    "architecture": "legacy_mlp",
    "hidden_units": [256, 256],
    "width": None,
    "body_depth": None,
    "block_size": 4,
    "activation": "relu",
    "use_batch_norm": False,
    "use_layer_norm": False,
    "dropout_rate": None,
}

_OBSERVATION_DEFAULTS = {
    "include_map": False,
    "include_location_sensing": False,
    "include_wall_sensing": False,
    "wall_sensing_version": None,
    "map_sensing_boundary_risk_threshold": None,
}

_EVALUATION_DEFAULTS = {
    "enabled": True,
    "every_epochs": 10,
    "num_episodes": 10,
    "seed": 0,
    "env_config": {},
}

_LOGGING_DEFAULTS = {
    "wandb": {
        "enabled": False,
        "project": "llm-offline-baselines",
    }
}


def _mapping(value, field_name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return copy.deepcopy(dict(value))


def _positive_int(value, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be an integer >= 1, got {value!r}")
    return int(value)


def _optional_positive_int(value, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name)


def _float(value, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number, got {value!r}")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite number, got {value!r}")
    return normalized


def _bool(value, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool, got {value!r}")
    return value


def _string_list(value, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings")
    normalized = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} must contain non-empty strings")
        normalized.append(item.strip())
    return normalized


def _normalize_network(value) -> dict:
    network = copy.deepcopy(_NETWORK_DEFAULTS)
    network.update(_mapping(value, "network"))
    unknown = sorted(set(network) - set(_NETWORK_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown network keys: {unknown}")
    architecture = network["architecture"]
    if not isinstance(architecture, str) or not architecture.strip():
        raise ValueError("network.architecture must be a non-empty string")
    architecture = architecture.strip()
    if architecture not in {"legacy_mlp", "plain_mlp", "residual_mlp"}:
        raise ValueError(
            "network.architecture must be one of legacy_mlp, plain_mlp, or residual_mlp"
        )
    network["architecture"] = architecture

    hidden_units = network["hidden_units"]
    if architecture == "legacy_mlp":
        if (
            not isinstance(hidden_units, list)
            or not hidden_units
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in hidden_units
            )
        ):
            raise ValueError(
                "network.hidden_units must be a non-empty list of positive integers"
            )
        if network["width"] is not None or network["body_depth"] is not None:
            raise ValueError(
                "network.width and network.body_depth require a scaling architecture"
            )
    else:
        network["width"] = _positive_int(network["width"], "network.width")
        network["body_depth"] = _positive_int(
            network["body_depth"], "network.body_depth"
        )
        # A layered scaling config commonly inherits hidden_units from a base
        # baseline YAML. It has no effect on this architecture, so clear it in
        # the resolved config rather than presenting a misleading second body.
        network["hidden_units"] = None
    network["block_size"] = _positive_int(network["block_size"], "network.block_size")
    activation = network["activation"]
    if not isinstance(activation, str) or not activation.strip():
        raise ValueError("network.activation must be a non-empty string")
    network["activation"] = activation.strip()
    if network["activation"] not in {"relu", "gelu", "tanh", "swish", "none", "geglu"}:
        raise ValueError(
            "network.activation must be one of relu, gelu, tanh, swish, none, or geglu"
        )
    for field in ("use_batch_norm", "use_layer_norm"):
        network[field] = _bool(network[field], f"network.{field}")
    dropout_rate = network["dropout_rate"]
    if dropout_rate is not None:
        if isinstance(dropout_rate, bool) or not isinstance(dropout_rate, (int, float)):
            raise ValueError("network.dropout_rate must be a number in [0, 1) or null")
        dropout_rate = float(dropout_rate)
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("network.dropout_rate must be in [0, 1)")
        network["dropout_rate"] = dropout_rate
    if architecture != "legacy_mlp":
        if network["block_size"] != 4:
            raise ValueError("paper scaling MLP requires network.block_size=4")
        if (
            network["activation"] != "swish"
            or not network["use_layer_norm"]
            or network["use_batch_norm"]
            or network["dropout_rate"] is not None
        ):
            raise ValueError(
                "plain_mlp/residual_mlp require activation=swish, "
                "use_layer_norm=true, use_batch_norm=false, and dropout_rate=null"
            )
        if (
            architecture == "residual_mlp"
            and network["body_depth"] % network["block_size"] != 0
        ):
            raise ValueError(
                "residual_mlp network.body_depth must be divisible by network.block_size"
            )
    return network


def _normalize_observation(value) -> dict:
    raw = _mapping(value, "observation")
    unknown = sorted(set(raw) - set(_OBSERVATION_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown observation keys: {unknown}")
    config = copy.deepcopy(_OBSERVATION_DEFAULTS)
    config.update(raw)
    for field in (
        "include_map",
        "include_location_sensing",
        "include_wall_sensing",
    ):
        config[field] = _bool(config[field], f"observation.{field}")
    config.update(resolve_sensing_config(config))
    return config


def _normalize_reward_scaler(value) -> dict | None:
    if value is None or value == "none":
        return None
    scaler = _mapping(value, "algorithm_config.reward_scaler")
    scaler_type = scaler.pop("type", None)
    if scaler_type == "multiply":
        allowed = {"multiplier"}
        required = {"multiplier"}
    elif scaler_type == "constant_shift":
        allowed = {"shift", "multiplier", "multiply_first"}
        required = {"shift"}
    elif scaler_type == "clip":
        allowed = {"low", "high", "multiplier"}
        required = set()
    else:
        raise ValueError(
            "algorithm_config.reward_scaler.type must be one of "
            "multiply, constant_shift, or clip"
        )
    unknown = sorted(set(scaler) - allowed)
    missing = sorted(required - set(scaler))
    if unknown:
        raise ValueError(f"Unknown reward_scaler keys for {scaler_type}: {unknown}")
    if missing:
        raise ValueError(f"Missing reward_scaler keys for {scaler_type}: {missing}")
    for field in ("multiplier", "shift", "low", "high"):
        if field in scaler and scaler[field] is not None:
            scaler[field] = _float(
                scaler[field], f"algorithm_config.reward_scaler.{field}"
            )
    if scaler.get("multiplier") == 0.0:
        raise ValueError("algorithm_config.reward_scaler.multiplier must be non-zero")
    if "multiply_first" in scaler:
        scaler["multiply_first"] = _bool(
            scaler["multiply_first"],
            "algorithm_config.reward_scaler.multiply_first",
        )
    if (
        scaler_type == "clip"
        and scaler.get("low") is not None
        and scaler.get("high") is not None
        and scaler["low"] >= scaler["high"]
    ):
        raise ValueError("reward_scaler.low must be smaller than reward_scaler.high")
    return {"type": scaler_type, **scaler}


def _normalize_algorithm_config(algorithm: str, value) -> dict:
    config = _mapping(value, "algorithm_config")
    unknown = sorted(set(config) - _ALGORITHM_FIELDS[algorithm])
    if unknown:
        raise ValueError(f"Unknown algorithm_config keys for {algorithm}: {unknown}")
    if algorithm in _GCRL_DEFAULTS:
        config = {**copy.deepcopy(_GCRL_DEFAULTS[algorithm]), **config}
    if "batch_size" in config:
        config["batch_size"] = _positive_int(config["batch_size"], "algorithm_config.batch_size")
    if "n_critics" in config:
        config["n_critics"] = _positive_int(config["n_critics"], "algorithm_config.n_critics")
    if "update_actor_interval" in config:
        config["update_actor_interval"] = _positive_int(
            config["update_actor_interval"], "algorithm_config.update_actor_interval"
        )
    for field in ("latent_dim", "subgoal_steps", "rep_dim"):
        if field in config:
            config[field] = _positive_int(
                config[field], f"algorithm_config.{field}"
            )
    if "compile_graph" in config:
        config["compile_graph"] = _bool(config["compile_graph"], "algorithm_config.compile_graph")
    positive_fields = {
        "learning_rate",
        "actor_learning_rate",
        "critic_learning_rate",
        "alpha",
        "weight_temp",
        "max_weight",
        "low_alpha",
        "high_alpha",
    }
    nonnegative_fields = {"target_smoothing_sigma", "target_smoothing_clip"}
    unit_interval_fields = {"gamma", "tau"}
    for field in positive_fields & set(config):
        config[field] = _float(config[field], f"algorithm_config.{field}")
        if config[field] <= 0.0:
            raise ValueError(f"algorithm_config.{field} must be > 0")
    for field in nonnegative_fields & set(config):
        config[field] = _float(config[field], f"algorithm_config.{field}")
        if config[field] < 0.0:
            raise ValueError(f"algorithm_config.{field} must be >= 0")
    for field in unit_interval_fields & set(config):
        config[field] = _float(config[field], f"algorithm_config.{field}")
        if not 0.0 < config[field] <= 1.0:
            raise ValueError(f"algorithm_config.{field} must satisfy 0 < value <= 1")
    if "expectile" in config:
        config["expectile"] = _float(
            config["expectile"], "algorithm_config.expectile"
        )
        if not 0.0 < config["expectile"] < 1.0:
            raise ValueError("algorithm_config.expectile must satisfy 0 < value < 1")
    if "discount" in config:
        config["discount"] = _float(
            config["discount"], "algorithm_config.discount"
        )
        if not 0.0 < config["discount"] < 1.0:
            raise ValueError("algorithm_config.discount must satisfy 0 < value < 1")
    for field in (
        "const_std",
        "normalize_observations",
        "value_geom_sample",
        "actor_geom_sample",
        "low_actor_rep_grad",
    ):
        if field in config:
            config[field] = _bool(config[field], f"algorithm_config.{field}")
    if algorithm in _GCRL_DEFAULTS:
        for prefix in ("value", "actor"):
            probability_fields = [
                f"{prefix}_p_curgoal",
                f"{prefix}_p_trajgoal",
                f"{prefix}_p_randomgoal",
            ]
            probabilities = []
            for field in probability_fields:
                probability = _float(config[field], f"algorithm_config.{field}")
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(
                        f"algorithm_config.{field} must satisfy 0 <= value <= 1"
                    )
                config[field] = probability
                probabilities.append(probability)
            if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-8):
                raise ValueError(
                    f"algorithm_config {prefix} goal probabilities must sum to 1"
                )
        if algorithm == "hiql" and config["actor_p_curgoal"] != 0.0:
            raise ValueError("hiql requires algorithm_config.actor_p_curgoal=0")
        if algorithm == "crl" and config["batch_size"] < 2:
            raise ValueError("crl requires algorithm_config.batch_size >= 2")
    if algorithm == "mlp_bc" and config.get("policy_type", "deterministic") != "deterministic":
        raise ValueError("mlp_bc only supports policy_type='deterministic' in this baseline")
    if "reward_scaler" in config:
        config["reward_scaler"] = _normalize_reward_scaler(config["reward_scaler"])
    return config


def _normalize_evaluation(value) -> dict:
    config = copy.deepcopy(_EVALUATION_DEFAULTS)
    config.update(_mapping(value, "evaluation"))
    unknown = sorted(set(config) - set(_EVALUATION_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown evaluation keys: {unknown}")
    config["enabled"] = _bool(config["enabled"], "evaluation.enabled")
    config["every_epochs"] = _positive_int(config["every_epochs"], "evaluation.every_epochs")
    config["num_episodes"] = _positive_int(config["num_episodes"], "evaluation.num_episodes")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int):
        raise ValueError("evaluation.seed must be an int")
    env_config = _mapping(config["env_config"], "evaluation.env_config")
    if "reward_type" in env_config:
        raise ValueError("Configure reward_type at the baseline top level, not evaluation.env_config")
    env_kwargs = env_config.get("env_kwargs") or {}
    if not isinstance(env_kwargs, Mapping):
        raise ValueError("evaluation.env_config.env_kwargs must be a mapping")
    if "reward_type" in env_kwargs:
        raise ValueError(
            "Configure reward_type at the baseline top level, not "
            "evaluation.env_config.env_kwargs"
        )
    sensing_keys = {
        "wall_sensing_version",
        "map_sensing_boundary_risk_threshold",
    }
    misplaced_sensing = sorted(sensing_keys & set(env_config))
    if misplaced_sensing:
        raise ValueError(
            "Configure baseline sensing under observation, not "
            f"evaluation.env_config: {misplaced_sensing}"
        )
    config["env_config"] = env_config
    return config


def _normalize_logging(value) -> dict:
    config = copy.deepcopy(_LOGGING_DEFAULTS)
    raw = _mapping(value, "logging")
    unknown = sorted(set(raw) - {"wandb"})
    if unknown:
        raise ValueError(f"Unknown logging keys: {unknown}")
    config["wandb"].update(_mapping(raw.get("wandb"), "logging.wandb"))
    unknown_wandb = sorted(set(config["wandb"]) - {"enabled", "project"})
    if unknown_wandb:
        raise ValueError(f"Unknown logging.wandb keys: {unknown_wandb}")
    config["wandb"]["enabled"] = _bool(
        config["wandb"]["enabled"], "logging.wandb.enabled"
    )
    project = config["wandb"]["project"]
    if not isinstance(project, str) or not project.strip():
        raise ValueError("logging.wandb.project must be a non-empty string")
    config["wandb"]["project"] = project.strip()
    return config


def normalize_baseline_config(raw_config: dict) -> dict:
    if not isinstance(raw_config, Mapping):
        raise ValueError("Baseline config must be a mapping")
    raw = copy.deepcopy(dict(raw_config))
    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"Unknown baseline config keys: {unknown}")

    algorithm = raw.get("algorithm")
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"algorithm must be one of {list(SUPPORTED_ALGORITHMS)}, got {algorithm!r}"
        )
    env_family = raw.get("env_family", "pointmaze")
    if env_family not in {"pointmaze", "antmaze"}:
        raise ValueError("env_family must be 'pointmaze' or 'antmaze'")

    train_mode = raw.get("train_mode", "single")
    if train_mode not in {"single", "all", "except"}:
        raise ValueError("train_mode must be single, all, or except")
    eval_mode = raw.get("eval_mode")
    if eval_mode is not None and eval_mode not in {"single", "all", "except"}:
        raise ValueError("eval_mode must be single, all, except, or null")
    eval_variants = _string_list(raw.get("eval_variants"), "eval_variants")
    if eval_mode is None and eval_variants:
        raise ValueError("eval_variants requires an explicit eval_mode")

    reward_type = raw.get("reward_type")
    if reward_type is not None:
        reward_type = normalize_reward_type(reward_type)

    episode_keep_per_variant = _mapping(
        raw.get("episode_keep_per_variant"), "episode_keep_per_variant"
    )
    for variant, keep in episode_keep_per_variant.items():
        if not isinstance(variant, str) or not variant.strip():
            raise ValueError("episode_keep_per_variant keys must be non-empty strings")
        episode_keep_per_variant[variant] = _positive_int(
            keep, f"episode_keep_per_variant.{variant}"
        )

    train_data_ratio = _float(raw.get("train_data_ratio", 0.9), "train_data_ratio")
    if not 0.0 < train_data_ratio < 1.0:
        raise ValueError("train_data_ratio must satisfy 0 < value < 1")

    n_steps = _positive_int(raw.get("n_steps", 1_000_000), "n_steps")
    n_steps_per_epoch = _positive_int(
        raw.get("n_steps_per_epoch", 10_000), "n_steps_per_epoch"
    )
    if n_steps % n_steps_per_epoch != 0:
        raise ValueError("n_steps must be divisible by n_steps_per_epoch")

    local_dataset_root = raw.get("local_dataset_root")
    if local_dataset_root is not None:
        if not isinstance(local_dataset_root, (str, os.PathLike)) or not os.fspath(local_dataset_root):
            raise ValueError("local_dataset_root must be a non-empty path or null")
        local_dataset_root = os.fspath(local_dataset_root)

    output_root = raw.get("output_root", "baseline_runs")
    if not isinstance(output_root, (str, os.PathLike)) or not os.fspath(output_root):
        raise ValueError("output_root must be a non-empty path")

    experiment_id = raw.get("experiment_id")
    if experiment_id is not None and (
        not isinstance(experiment_id, str) or not experiment_id.strip()
    ):
        raise ValueError("experiment_id must be a non-empty string or null")

    sampling_seed = raw.get("sampling_seed", 0)
    seed = raw.get("seed", 0)
    for value, field in ((sampling_seed, "sampling_seed"), (seed, "seed")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be an int")

    device = raw.get("device", "cuda:0")
    if not isinstance(device, (str, bool, int)):
        raise ValueError("device must be a device string, bool, or integer GPU index")
    if isinstance(device, str) and not device.strip():
        raise ValueError("device must not be an empty string")

    normalized_network = _normalize_network(raw.get("network"))
    if (
        normalized_network["architecture"] != "legacy_mlp"
        and algorithm != "mlp_bc"
    ):
        raise ValueError(
            "plain_mlp/residual_mlp are currently supported only by algorithm=mlp_bc"
        )
    if algorithm in _GCRL_DEFAULTS:
        if normalized_network["use_batch_norm"]:
            raise ValueError("CRL/HIQL do not support network.use_batch_norm")
        if normalized_network["dropout_rate"] is not None:
            raise ValueError("CRL/HIQL do not support network.dropout_rate")
        if normalized_network["activation"] not in {
            "relu",
            "gelu",
            "tanh",
            "swish",
        }:
            raise ValueError(
                "CRL/HIQL network.activation must be relu, gelu, tanh, or swish"
            )

    return {
        "algorithm": algorithm,
        "env_family": env_family,
        "train_mode": train_mode,
        "train_variants": _string_list(raw.get("train_variants"), "train_variants"),
        "eval_mode": eval_mode,
        "eval_variants": eval_variants,
        "reward_type": reward_type,
        "allow_mixed_reward_types": _bool(
            raw.get("allow_mixed_reward_types", False), "allow_mixed_reward_types"
        ),
        "local_dataset_root": local_dataset_root,
        "episode_keep_num": _optional_positive_int(
            raw.get("episode_keep_num"), "episode_keep_num"
        ),
        "episode_keep_per_variant": episode_keep_per_variant,
        "balance_variant_episode_count": _bool(
            raw.get("balance_variant_episode_count", False),
            "balance_variant_episode_count",
        ),
        "sampling_seed": int(sampling_seed),
        "train_data_ratio": train_data_ratio,
        "seed": int(seed),
        "device": device,
        "n_steps": n_steps,
        "n_steps_per_epoch": n_steps_per_epoch,
        "save_interval_epochs": _positive_int(
            raw.get("save_interval_epochs", 10), "save_interval_epochs"
        ),
        "show_progress": _bool(raw.get("show_progress", True), "show_progress"),
        "output_root": os.fspath(output_root),
        "experiment_id": experiment_id.strip() if experiment_id else None,
        "observation": _normalize_observation(raw.get("observation")),
        "network": normalized_network,
        "algorithm_config": _normalize_algorithm_config(
            algorithm, raw.get("algorithm_config")
        ),
        "evaluation": _normalize_evaluation(raw.get("evaluation")),
        "logging": _normalize_logging(raw.get("logging")),
    }
