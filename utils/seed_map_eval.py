"""Resolve procedural seed-map evaluation targets.

Evaluation treats each selected map seed as one synthetic variant.  This keeps
the existing variant-parallel rollout machinery intact while the worker
regenerates the map from the versioned seed-map spec.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from crossmaze.reward import normalize_reward_type
from crossmaze.seed_map import (
    SEED_MAP_VERSION,
    SeedMapSpec,
    normalize_seed_map_spec,
    stable_seed_map_int,
)
from data.seed_map_corpus import (
    load_seed_map_manifest,
    normalize_seed_ranges,
)


def seed_map_eval_variant(map_seed: int) -> str:
    return f"seed-map-{int(map_seed)}"


@dataclass(frozen=True)
class SeedMapEvalSelection:
    seed_ranges: tuple[tuple[int, int], ...]
    seed_count: int
    selection_seed: int
    selected_seeds: tuple[int, ...]
    seed_map_spec: SeedMapSpec
    reward_type: str
    max_episode_steps: int | None
    dataset_path: str | None
    selection_hash: str

    @property
    def selected_variants(self) -> list[str]:
        return [seed_map_eval_variant(seed) for seed in self.selected_seeds]

    @property
    def selection_tag(self) -> str:
        return f"seed-map-eval-n{self.seed_count}-{self.selection_hash[:10]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_ranges": [list(item) for item in self.seed_ranges],
            "seed_count": self.seed_count,
            "selection_seed": self.selection_seed,
            "selected_seeds": list(self.selected_seeds),
            "selected_variants": self.selected_variants,
            "seed_map_spec": self.seed_map_spec.to_dict(),
            "reward_type": self.reward_type,
            "max_episode_steps": self.max_episode_steps,
            "dataset_path": self.dataset_path,
            "selection_hash": self.selection_hash,
            "selection_tag": self.selection_tag,
        }


def _require_int(value: Any, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {value}")
    return int(value)


def _spec_from_section(raw: dict[str, Any]) -> SeedMapSpec:
    size_mode = raw.get("seed_map_size_mode", "random")
    spec = {
        "version": raw.get("seed_map_version", SEED_MAP_VERSION),
        "size_mode": size_mode,
    }
    if size_mode == "fixed":
        spec["fixed_rows"] = raw.get("seed_map_fixed_rows")
        spec["fixed_cols"] = raw.get("seed_map_fixed_cols")
    else:
        if "seed_map_min_size" in raw:
            spec["min_size"] = raw["seed_map_min_size"]
        if "seed_map_max_size" in raw:
            spec["max_size"] = raw["seed_map_max_size"]
    return normalize_seed_map_spec(spec)


def resolve_seed_map_eval_selection(
    section: Mapping[str, Any],
    *,
    env_family: str,
    default_reward_type: str | None = None,
) -> SeedMapEvalSelection:
    if not isinstance(section, Mapping):
        raise ValueError(
            f"seed_map_eval must be a mapping, got {type(section).__name__}"
        )
    raw = dict(section)
    allowed = {
        "enabled",
        "dataset_path",
        "seed_ranges",
        "seed_count",
        "selection_seed",
        "reward_type",
        "max_episode_steps",
        "seed_map_version",
        "seed_map_size_mode",
        "seed_map_min_size",
        "seed_map_max_size",
        "seed_map_fixed_rows",
        "seed_map_fixed_cols",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown seed_map_eval fields: {unknown}")
    if raw.get("enabled") is not True:
        raise ValueError(
            "resolve_seed_map_eval_selection requires seed_map_eval.enabled=true"
        )
    if env_family not in {"pointmaze", "antmaze"}:
        raise ValueError(f"seed_map_eval does not support env_family={env_family!r}")

    dataset_path = None
    manifest = None
    if raw.get("dataset_path") is not None:
        value = raw["dataset_path"]
        if not isinstance(value, (str, os.PathLike)) or not os.fspath(value):
            raise ValueError("seed_map_eval.dataset_path must be a non-empty path")
        dataset_path = str(Path(value).expanduser().resolve())
        manifest = load_seed_map_manifest(dataset_path, require_complete=True)
        if manifest["env_family"] != env_family:
            raise ValueError(
                f"seed_map_eval corpus env_family={manifest['env_family']!r} "
                f"does not match {env_family!r}"
            )
        seed_map_spec = normalize_seed_map_spec(manifest["seed_map_spec"])
        spec_field_values = {
            "seed_map_version": seed_map_spec.version,
            "seed_map_size_mode": seed_map_spec.size_mode,
            "seed_map_min_size": seed_map_spec.min_size,
            "seed_map_max_size": seed_map_spec.max_size,
            "seed_map_fixed_rows": seed_map_spec.fixed_rows,
            "seed_map_fixed_cols": seed_map_spec.fixed_cols,
        }
        conflicting_spec_fields = [
            key
            for key, expected in spec_field_values.items()
            if key in raw and raw[key] != expected
        ]
        if conflicting_spec_fields:
            details = {
                key: {
                    "configured": raw[key],
                    "manifest": spec_field_values[key],
                }
                for key in conflicting_spec_fields
            }
            raise ValueError(
                "seed_map_eval seed-map size/version fields conflict with "
                f"the dataset manifest: {details}"
            )
        ranges = normalize_seed_ranges(
            raw.get("seed_ranges"),
            default_start=int(manifest["seed_start"]),
            default_end=int(manifest["seed_end"]),
            field="seed_map_eval.seed_ranges",
        )
        manifest_reward_type = normalize_reward_type(manifest["reward_type"])
        reward_type = normalize_reward_type(
            raw.get("reward_type"),
            default=manifest_reward_type,
        )
    else:
        raw_ranges = raw.get("seed_ranges")
        if not isinstance(raw_ranges, list) or not raw_ranges:
            raise ValueError(
                "seed_map_eval without dataset_path requires seed_ranges"
            )
        range_starts = []
        range_ends = []
        for index, item in enumerate(raw_ranges):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(
                    "seed_map_eval.seed_ranges must contain [start, end] pairs"
                )
            range_starts.append(
                _require_int(
                    item[0],
                    field=f"seed_map_eval.seed_ranges[{index}][0]",
                    minimum=0,
                )
            )
            range_ends.append(
                _require_int(
                    item[1],
                    field=f"seed_map_eval.seed_ranges[{index}][1]",
                    minimum=1,
                )
            )
        ranges = normalize_seed_ranges(
            raw_ranges,
            default_start=min(range_starts),
            default_end=max(range_ends),
            field="seed_map_eval.seed_ranges",
        )
        seed_map_spec = _spec_from_section(raw)
        reward_type = normalize_reward_type(
            raw.get("reward_type"),
            default=normalize_reward_type(default_reward_type, default="sparse"),
        )

    allowed_seeds = [
        seed
        for start, end in ranges
        for seed in range(start, end)
    ]
    requested_seed_count = raw.get("seed_count")
    seed_count = (
        len(allowed_seeds)
        if requested_seed_count is None
        else _require_int(
            requested_seed_count,
            field="seed_map_eval.seed_count",
            minimum=1,
        )
    )
    if seed_count > len(allowed_seeds):
        raise ValueError(
            f"seed_map_eval.seed_count={seed_count} exceeds "
            f"{len(allowed_seeds)} seeds in seed_ranges"
        )
    selection_seed = _require_int(
        raw.get("selection_seed", 0),
        field="seed_map_eval.selection_seed",
        minimum=0,
    )
    max_episode_steps = raw.get("max_episode_steps")
    if max_episode_steps is not None:
        max_episode_steps = _require_int(
            max_episode_steps,
            field="seed_map_eval.max_episode_steps",
            minimum=1,
        )

    ordered_seeds = sorted(
        allowed_seeds,
        key=lambda seed: (
            stable_seed_map_int(
                "seed_map_eval",
                selection_seed,
                "map",
                seed,
            ),
            seed,
        ),
    )
    selected_seeds = tuple(sorted(ordered_seeds[:seed_count]))
    payload = {
        "env_family": env_family,
        "seed_ranges": ranges,
        "seed_count": seed_count,
        "selection_seed": selection_seed,
        "selected_seeds": selected_seeds,
        "seed_map_spec": seed_map_spec.to_dict(),
        "reward_type": reward_type,
        "max_episode_steps": max_episode_steps,
        "dataset_path": dataset_path,
        "manifest_content_hash": (
            manifest.get("content_hash") if manifest is not None else None
        ),
    }
    selection_hash = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return SeedMapEvalSelection(
        seed_ranges=ranges,
        seed_count=seed_count,
        selection_seed=selection_seed,
        selected_seeds=selected_seeds,
        seed_map_spec=seed_map_spec,
        reward_type=reward_type,
        max_episode_steps=max_episode_steps,
        dataset_path=dataset_path,
        selection_hash=selection_hash,
    )


def seed_map_eval_target(config: Mapping[str, Any], variant: str) -> dict[str, Any] | None:
    resolved = config.get("resolved_seed_map_eval")
    if not isinstance(resolved, Mapping):
        return None
    selected_variants = list(resolved.get("selected_variants") or [])
    if variant not in selected_variants:
        return None
    index = selected_variants.index(variant)
    selected_seeds = list(resolved.get("selected_seeds") or [])
    return {
        "variant": variant,
        "map_seed": int(selected_seeds[index]),
        "seed_map_spec": dict(resolved["seed_map_spec"]),
        "reward_type": str(resolved["reward_type"]),
        "max_episode_steps": resolved.get("max_episode_steps"),
    }
