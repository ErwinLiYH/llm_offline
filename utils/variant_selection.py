from __future__ import annotations

from dataclasses import dataclass

from data.pointmaze.variants import POINTMAZE_VARIANTS


@dataclass(frozen=True)
class VariantSelection:
    mode: str
    configured_variants: list[str]
    selected_variants: list[str]
    selection_tag: str



def get_available_variants(env_family: str) -> list[str]:
    if env_family == "pointmaze":
        return list(POINTMAZE_VARIANTS.keys())
    raise ValueError(f"Unsupported env_family for variant resolution: {env_family!r}")



def _normalize_variants(value, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError(f"{field_name} must be a list of variant names, got {type(value).__name__}")

    normalized = []
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} must contain non-empty strings, got {item!r}")
        normalized.append(item.strip())
    return normalized



def _validate_variants(variants: list[str], available_variants: list[str], field_name: str):
    seen = set()
    duplicates = []
    for variant in variants:
        if variant in seen:
            duplicates.append(variant)
        seen.add(variant)
    if duplicates:
        dup_str = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"{field_name} contains duplicate variants: {dup_str}")

    unknown = sorted(set(variants) - set(available_variants))
    if unknown:
        unknown_str = ", ".join(unknown)
        raise ValueError(f"{field_name} contains unknown variants: {unknown_str}")



def _except_tag(excluded_variants: list[str]) -> str:
    joined = "+".join(sorted(excluded_variants))
    return f"except-{joined}"


def _all_subset_tag(selected_variants: list[str]) -> str:
    joined = "+".join(sorted(selected_variants))
    return f"all-{joined}"



def resolve_selection(
    *,
    mode: str,
    variants,
    available_variants: list[str],
    field_name: str,
    default_variants: list[str] | None = None,
) -> VariantSelection:
    configured_variants = _normalize_variants(variants, field_name)
    _validate_variants(configured_variants, available_variants, field_name)

    if mode == "single":
        candidates = configured_variants or list(default_variants or [])
        if len(candidates) != 1:
            raise ValueError(
                f"{field_name} must contain exactly one variant when mode='single', got {candidates}"
            )
        selected_variants = [candidates[0]]
        selection_tag = selected_variants[0]
        configured_for_record = configured_variants or selected_variants
    elif mode == "all":
        selected_variants = configured_variants or list(available_variants)
        selection_tag = "all" if not configured_variants else _all_subset_tag(selected_variants)
        configured_for_record = configured_variants
    elif mode == "except":
        excluded_variants = configured_variants or list(default_variants or [])
        if not excluded_variants:
            raise ValueError(f"{field_name} must contain at least one variant when mode='except'")
        selected_variants = [v for v in available_variants if v not in set(excluded_variants)]
        if not selected_variants:
            raise ValueError("except mode excluded all available variants")
        selection_tag = _except_tag(excluded_variants)
        configured_for_record = excluded_variants
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'single', 'all', or 'except'.")

    return VariantSelection(
        mode=mode,
        configured_variants=configured_for_record,
        selected_variants=selected_variants,
        selection_tag=selection_tag,
    )
