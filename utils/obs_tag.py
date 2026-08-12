"""Configuration and controlled corruption helpers for observation tags."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


RANDOM_OBS_TAG_KEY = "random_obs_tag"
RANDOM_OBS_TAG_SCHEMA = "fixed_derangement_v1"


def resolve_random_obs_tag(value=None) -> bool:
    """Resolve the observation-tag ablation switch with strict bool semantics."""
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(
            f"{RANDOM_OBS_TAG_KEY} must be a bool, got {value!r}"
        )
    return value


def normalize_random_obs_tag_config(config: dict) -> bool:
    """Normalize and persist ``random_obs_tag`` in a runtime config."""
    resolved = resolve_random_obs_tag(config.get(RANDOM_OBS_TAG_KEY))
    config[RANDOM_OBS_TAG_KEY] = resolved
    return resolved


def apply_random_obs_tag_to_prompt_vars(
    prompt_vars: Mapping,
    config: Mapping | None,
) -> dict:
    """Copy prompt vars and inject the normalized observation-tag switch."""
    resolved = dict(prompt_vars)
    resolved[RANDOM_OBS_TAG_KEY] = resolve_random_obs_tag(
        None if config is None else config.get(RANDOM_OBS_TAG_KEY)
    )
    return resolved


def apply_checkpoint_random_obs_tag_config(
    config: Mapping,
    checkpoint_config: Mapping | None,
) -> dict:
    """Inherit the training switch when eval/score does not explicitly set it.

    An explicit eval/score value is retained so the same checkpoint can also be
    used for test-time observation-tag ablations.
    """
    merged = dict(config)
    if RANDOM_OBS_TAG_KEY not in merged and checkpoint_config:
        if RANDOM_OBS_TAG_KEY in checkpoint_config:
            merged[RANDOM_OBS_TAG_KEY] = checkpoint_config[RANDOM_OBS_TAG_KEY]
    normalize_random_obs_tag_config(merged)
    return merged


def fixed_obs_tags(
    tags: Sequence[str],
    shuffled_tags: Sequence[str],
    *,
    enabled: bool,
) -> tuple[str, ...]:
    """Return one explicit, validated tag derangement when enabled."""
    resolved_tags = tuple(tags)
    if not enabled:
        return resolved_tags
    if len(resolved_tags) < 2:
        raise ValueError("random_obs_tag requires at least two observation tags")
    if len(set(resolved_tags)) != len(resolved_tags):
        raise ValueError("random_obs_tag requires unique observation tags")

    resolved_shuffled_tags = tuple(shuffled_tags)
    if len(resolved_shuffled_tags) != len(resolved_tags):
        raise ValueError(
            "random_obs_tag fixed order must have the same length as the original tags: "
            f"tags={len(resolved_tags)}, fixed_order={len(resolved_shuffled_tags)}"
        )
    if set(resolved_shuffled_tags) != set(resolved_tags):
        raise ValueError(
            "random_obs_tag fixed order must be a permutation of the original tags"
        )
    if any(
        original == shuffled
        for original, shuffled in zip(resolved_tags, resolved_shuffled_tags)
    ):
        raise ValueError(
            "random_obs_tag fixed order must be a derangement with no unchanged slots"
        )
    return resolved_shuffled_tags
