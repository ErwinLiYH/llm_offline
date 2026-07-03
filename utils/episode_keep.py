from __future__ import annotations

from collections.abc import Mapping


EPISODE_KEEP_PER_VARIANT_KEY = "episode_keep_per_varient"
RESOLVED_EPISODE_KEEP_PER_VARIANT_KEY = "resolved_episode_keep_per_varient"


def normalize_episode_keep_num(value, *, field_name: str = "episode_keep_num") -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an integer episode count >= 1 or null to use all episodes, "
            f"got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1 when provided, got {value}")
    return int(value)


def has_episode_keep_per_variant(config: Mapping) -> bool:
    return EPISODE_KEEP_PER_VARIANT_KEY in config


def resolve_episode_keep_per_variant(
    config: Mapping,
    selected_variants: list[str],
    *,
    available_variants: list[str] | None = None,
) -> dict[str, int | None]:
    base_keep = normalize_episode_keep_num(config.get("episode_keep_num"))
    if EPISODE_KEEP_PER_VARIANT_KEY not in config:
        return {variant: base_keep for variant in selected_variants}

    raw = config.get(EPISODE_KEEP_PER_VARIANT_KEY)
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"{EPISODE_KEEP_PER_VARIANT_KEY} must be a dict mapping selected variant names "
            "to integer episode counts >= 1 or null."
        )

    overrides: dict[str, int | None] = {}
    duplicate_keys = []
    for raw_key, raw_value in raw.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ValueError(
                f"{EPISODE_KEEP_PER_VARIANT_KEY} keys must be non-empty variant names, "
                f"got {raw_key!r}"
            )
        key = raw_key.strip()
        if key in overrides:
            duplicate_keys.append(key)
        overrides[key] = normalize_episode_keep_num(
            raw_value,
            field_name=f"{EPISODE_KEEP_PER_VARIANT_KEY}.{key}",
        )
    if duplicate_keys:
        duplicates = ", ".join(sorted(set(duplicate_keys)))
        raise ValueError(f"{EPISODE_KEEP_PER_VARIANT_KEY} contains duplicate variants: {duplicates}")

    override_variants = set(overrides)
    if available_variants is not None:
        unknown = sorted(override_variants - set(available_variants))
        if unknown:
            raise ValueError(
                f"{EPISODE_KEEP_PER_VARIANT_KEY} contains unknown variants: "
                f"{', '.join(unknown)}"
            )

    unselected = sorted(override_variants - set(selected_variants))
    if unselected:
        raise ValueError(
            f"{EPISODE_KEEP_PER_VARIANT_KEY} contains variants that are not selected: "
            f"{', '.join(unselected)}"
        )

    return {
        variant: overrides.get(variant, base_keep)
        for variant in selected_variants
    }


def effective_episode_keep_num(config: Mapping, variant: str) -> int | None:
    resolved = config.get(RESOLVED_EPISODE_KEEP_PER_VARIANT_KEY)
    if resolved is not None:
        if not isinstance(resolved, Mapping):
            raise ValueError(
                f"{RESOLVED_EPISODE_KEEP_PER_VARIANT_KEY} must be a dict, "
                f"got {type(resolved).__name__}"
            )
        if variant not in resolved:
            raise ValueError(
                f"{RESOLVED_EPISODE_KEEP_PER_VARIANT_KEY} is missing selected variant {variant!r}"
            )
        return normalize_episode_keep_num(
            resolved[variant],
            field_name=f"{RESOLVED_EPISODE_KEEP_PER_VARIANT_KEY}.{variant}",
        )

    base_keep = normalize_episode_keep_num(config.get("episode_keep_num"))
    if EPISODE_KEEP_PER_VARIANT_KEY not in config:
        return base_keep
    raw = config.get(EPISODE_KEEP_PER_VARIANT_KEY)
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"{EPISODE_KEEP_PER_VARIANT_KEY} must be a dict mapping selected variant names "
            "to integer episode counts >= 1 or null."
        )
    if variant not in raw:
        return base_keep
    return normalize_episode_keep_num(
        raw[variant],
        field_name=f"{EPISODE_KEEP_PER_VARIANT_KEY}.{variant}",
    )
