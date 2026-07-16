from __future__ import annotations

from dataclasses import dataclass

from crossmaze import get_env_facts, list_variants
from crossmaze.reward import normalize_reward_type

from utils.variant_selection import VariantSelection, resolve_selection


@dataclass(frozen=True)
class BaselineSelections:
    train: VariantSelection
    eval: VariantSelection
    train_reward_types: dict[str, str]
    eval_reward_types: dict[str, str]


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


def resolve_baseline_selections(config: dict) -> BaselineSelections:
    env_family = config["env_family"]
    available = list_variants(env_family)
    train = resolve_selection(
        mode=config["train_mode"],
        variants=config["train_variants"],
        available_variants=available,
        field_name="train_variants",
    )
    if config["eval_mode"] is None:
        eval_selection = VariantSelection(
            mode=train.mode,
            configured_variants=list(train.configured_variants),
            selected_variants=list(train.selected_variants),
            selection_tag=train.selection_tag,
            full_selection_tag=train.full_selection_tag,
        )
    else:
        eval_selection = resolve_selection(
            mode=config["eval_mode"],
            variants=config["eval_variants"],
            available_variants=available,
            field_name="eval_variants",
        )

    train_reward_types = _resolve_reward_types(
        env_family, train.selected_variants, config["reward_type"]
    )
    distinct_train_rewards = sorted(set(train_reward_types.values()))
    if len(distinct_train_rewards) > 1 and not config["allow_mixed_reward_types"]:
        raise ValueError(
            "Selected training variants mix reward types "
            f"{distinct_train_rewards}. Set allow_mixed_reward_types=true only "
            "when this mismatch is intentional."
        )
    eval_reward_types = _resolve_reward_types(
        env_family, eval_selection.selected_variants, config["reward_type"]
    )
    return BaselineSelections(
        train=train,
        eval=eval_selection,
        train_reward_types=train_reward_types,
        eval_reward_types=eval_reward_types,
    )
