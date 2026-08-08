from __future__ import annotations

from dataclasses import dataclass

from crossmaze import get_env_facts, list_variants
from crossmaze.reward import normalize_reward_type
from crossmaze.seed_map import normalize_seed_map_spec
from data.seed_map_corpus import (
    SeedMapSelection,
    load_seed_map_manifest,
    resolve_seed_map_selection,
)

from utils.seed_map_config import seed_map_section_enabled
from utils.seed_map_eval import (
    SeedMapEvalSelection,
    resolve_seed_map_eval_selection,
    seed_map_eval_variant,
)
from utils.variant_selection import VariantSelection, resolve_selection


@dataclass(frozen=True)
class BaselineSelections:
    train: VariantSelection
    eval: VariantSelection
    train_reward_types: dict[str, str]
    eval_reward_types: dict[str, str]
    seed_map_train: SeedMapSelection | None = None
    seed_map_eval: SeedMapEvalSelection | None = None


def _resolve_reward_types(
    env_family: str,
    variants: list[str],
    configured_reward_type: str | None,
) -> dict[str, str]:
    resolved = {}
    for variant in variants:
        facts = get_env_facts(env_family, variant)
        default_reward_type = normalize_reward_type(facts["reward_type"])
        if facts["kind"] == "remote":
            if (
                configured_reward_type is not None
                and configured_reward_type != default_reward_type
            ):
                raise ValueError(
                    f"Remote variant {variant!r} has a fixed {default_reward_type!r} "
                    f"dataset; reward_type={configured_reward_type!r} cannot override it. "
                    "Select the matching registered variant instead."
                )
            resolved[variant] = default_reward_type
        else:
            resolved[variant] = normalize_reward_type(
                configured_reward_type, default=default_reward_type
            )
    return resolved


def _seed_map_variant_selection(
    selected_seeds: tuple[int, ...],
    *,
    selection_tag: str,
) -> VariantSelection:
    return VariantSelection(
        mode="seed_map",
        configured_variants=[],
        selected_variants=[seed_map_eval_variant(seed) for seed in selected_seeds],
        selection_tag=selection_tag,
        full_selection_tag=selection_tag,
    )


def _legacy_train_selection(config: dict, available: list[str]) -> VariantSelection:
    return resolve_selection(
        mode=config["train_mode"],
        variants=config["train_variants"],
        available_variants=available,
        field_name="train_variants",
    )


def _seed_map_spec_shape(spec) -> tuple[int, int]:
    if spec.size_mode == "fixed":
        return int(spec.fixed_rows), int(spec.fixed_cols)
    return int(spec.max_size), int(spec.max_size)


def _registered_map_shape(env_family: str, available: list[str]) -> tuple[int, int]:
    shapes = []
    for variant in available:
        maze_map = get_env_facts(env_family, variant)["maze_map"]
        shapes.append((len(maze_map), len(maze_map[0])))
    return (
        max(rows for rows, _cols in shapes),
        max(cols for _rows, cols in shapes),
    )


def _resolve_observation_map_shape(
    config: dict,
    *,
    available: list[str],
    train_spec=None,
    eval_spec=None,
) -> None:
    specs = [spec for spec in (train_spec, eval_spec) if spec is not None]
    if not specs:
        return
    rows, cols = _registered_map_shape(config["env_family"], available)
    for spec in specs:
        spec_rows, spec_cols = _seed_map_spec_shape(spec)
        rows = max(rows, spec_rows)
        cols = max(cols, spec_cols)
    configured = config["observation"].get("map_shape")
    if configured is not None:
        if configured[0] < rows or configured[1] < cols:
            raise ValueError(
                "observation.map_shape is too small for the selected registered/"
                f"seed maps: configured={configured}, required>={[rows, cols]}"
            )
        rows, cols = int(configured[0]), int(configured[1])
    config["observation"]["map_shape"] = [rows, cols]


def resolve_baseline_selections(config: dict) -> BaselineSelections:
    env_family = config["env_family"]
    available = list_variants(env_family)
    train_seed_map = None
    train_seed_map_spec = None
    train_manifest = None
    if seed_map_section_enabled(config, "seed_map_train"):
        train_seed_map = resolve_seed_map_selection(
            config["seed_map_train"],
            env_family=env_family,
        )
        train_manifest = load_seed_map_manifest(
            train_seed_map.dataset_path,
            require_complete=True,
        )
        train_seed_map_spec = normalize_seed_map_spec(
            train_manifest["seed_map_spec"]
        )
        config["resolved_seed_map_train"] = train_seed_map.to_dict()
        train = _seed_map_variant_selection(
            train_seed_map.selected_seeds,
            selection_tag=train_seed_map.source_tag,
        )
    else:
        train = _legacy_train_selection(config, available)

    eval_seed_map = None
    if seed_map_section_enabled(config, "seed_map_eval"):
        eval_seed_map = resolve_seed_map_eval_selection(
            config["seed_map_eval"],
            env_family=env_family,
            default_reward_type=config.get("reward_type"),
        )
        config["resolved_seed_map_eval"] = eval_seed_map.to_dict()
        eval_selection = _seed_map_variant_selection(
            eval_seed_map.selected_seeds,
            selection_tag=eval_seed_map.selection_tag,
        )
    elif config["eval_mode"] is None and train_seed_map is None:
        eval_selection = VariantSelection(
            mode=train.mode,
            configured_variants=list(train.configured_variants),
            selected_variants=list(train.selected_variants),
            selection_tag=train.selection_tag,
            full_selection_tag=train.full_selection_tag,
        )
    elif config["eval_mode"] is None:
        # Match the LLM training-time fallback: seed-map training does not
        # implicitly turn its offline corpus into an evaluation suite.
        eval_selection = _legacy_train_selection(config, available)
    else:
        eval_selection = resolve_selection(
            mode=config["eval_mode"],
            variants=config["eval_variants"],
            available_variants=available,
            field_name="eval_variants",
        )

    if train_seed_map is None:
        train_reward_types = _resolve_reward_types(
            env_family, train.selected_variants, config["reward_type"]
        )
    else:
        corpus_reward_type = normalize_reward_type(train_manifest["reward_type"])
        configured_reward_type = config.get("reward_type")
        if (
            configured_reward_type is not None
            and normalize_reward_type(configured_reward_type) != corpus_reward_type
        ):
            raise ValueError(
                "seed_map_train corpus reward type conflicts with reward_type: "
                f"corpus={corpus_reward_type!r}, configured={configured_reward_type!r}"
            )
        train_reward_types = {
            variant: corpus_reward_type for variant in train.selected_variants
        }
    distinct_train_rewards = sorted(set(train_reward_types.values()))
    if len(distinct_train_rewards) > 1 and not config["allow_mixed_reward_types"]:
        raise ValueError(
            "Selected training variants mix reward types "
            f"{distinct_train_rewards}. Set allow_mixed_reward_types=true only "
            "when this mismatch is intentional."
        )
    if eval_seed_map is None:
        eval_reward_types = _resolve_reward_types(
            env_family, eval_selection.selected_variants, config["reward_type"]
        )
    else:
        eval_reward_types = {
            variant: eval_seed_map.reward_type
            for variant in eval_selection.selected_variants
        }
    _resolve_observation_map_shape(
        config,
        available=available,
        train_spec=train_seed_map_spec,
        eval_spec=(eval_seed_map.seed_map_spec if eval_seed_map is not None else None),
    )
    return BaselineSelections(
        train=train,
        eval=eval_selection,
        train_reward_types=train_reward_types,
        eval_reward_types=eval_reward_types,
        seed_map_train=train_seed_map,
        seed_map_eval=eval_seed_map,
    )
