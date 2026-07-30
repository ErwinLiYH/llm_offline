"""Aggregated offline trajectory corpus for procedural seed maps.

A corpus is one logical dataset directory, not one directory per map:

```
<dataset>/
  manifest.json
  index.jsonl
  data/main_data.hdf5
```

Each HDF5 episode group carries its ``map_seed`` and trajectory index.  The
manifest owns the versioned seed-map generator config, so training and eval can
reconstruct every map without registering thousands of synthetic variants.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any
import uuid

import h5py
import numpy as np

from crossmaze.reward import normalize_reward_type
from crossmaze.seed_map import (
    SeedMapSpec,
    build_seed_map_prompt_vars,
    generate_seed_map,
    normalize_seed_map_spec,
    seed_map_generator_hash,
    seed_map_hash,
    stable_seed_map_int,
)


SEED_MAP_CORPUS_SCHEMA_VERSION = 1
SEED_MAP_MANIFEST_NAME = "manifest.json"
SEED_MAP_INDEX_NAME = "index.jsonl"
SEED_MAP_DATA_RELATIVE_PATH = Path("data") / "main_data.hdf5"
SEED_MAP_SOURCE_KIND = "seed_map"


def seed_map_dataset_name(
    seed_start: int,
    seed_end: int,
    trajectories_per_seed: int,
) -> str:
    seed_start = _require_int(seed_start, field="seed_map_start", minimum=0)
    seed_end = _require_int(seed_end, field="seed_map_end", minimum=1)
    trajectories_per_seed = _require_int(
        trajectories_per_seed,
        field="seed_map_trajectories_per_seed",
        minimum=1,
    )
    if seed_start >= seed_end:
        raise ValueError(
            "seed_map_start must be < seed_map_end for the half-open seed range, "
            f"got [{seed_start}, {seed_end})"
        )
    return f"{seed_start}-{seed_end}-{trajectories_per_seed}"


def _require_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {value}")
    return int(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _corpus_paths(dataset_path: str | os.PathLike[str]) -> tuple[Path, Path, Path, Path]:
    root = Path(dataset_path).expanduser().resolve()
    return (
        root,
        root / SEED_MAP_MANIFEST_NAME,
        root / SEED_MAP_INDEX_NAME,
        root / SEED_MAP_DATA_RELATIVE_PATH,
    )


def _manifest_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return immutable generation fields used to validate resume."""

    return {
        "schema_version": manifest["schema_version"],
        "source_kind": manifest["source_kind"],
        "env_family": manifest["env_family"],
        "reward_type": manifest["reward_type"],
        "seed_start": manifest["seed_start"],
        "seed_end": manifest["seed_end"],
        "trajectories_per_seed": manifest["trajectories_per_seed"],
        "dataset_name": manifest["dataset_name"],
        "seed_map_spec": manifest["seed_map_spec"],
        "seed_map_generator_hash": manifest["seed_map_generator_hash"],
        "collection_config": manifest.get("collection_config") or {},
    }


def create_seed_map_corpus(
    dataset_path: str | os.PathLike[str],
    *,
    env_family: str,
    reward_type: str,
    seed_start: int,
    seed_end: int,
    trajectories_per_seed: int,
    seed_map_spec: SeedMapSpec | Mapping[str, Any] | None,
    collection_config: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create or resume an incomplete corpus with an exact immutable contract."""

    if env_family not in {"pointmaze", "antmaze"}:
        raise ValueError(
            f"seed-map corpus env_family must be pointmaze or antmaze, got {env_family!r}"
        )
    reward_type = normalize_reward_type(reward_type)
    name = seed_map_dataset_name(seed_start, seed_end, trajectories_per_seed)
    resolved_spec = normalize_seed_map_spec(seed_map_spec)
    root, manifest_path, index_path, data_path = _corpus_paths(dataset_path)
    expected = {
        "schema_version": SEED_MAP_CORPUS_SCHEMA_VERSION,
        "source_kind": SEED_MAP_SOURCE_KIND,
        "env_family": env_family,
        "reward_type": reward_type,
        "seed_start": int(seed_start),
        "seed_end": int(seed_end),
        "trajectories_per_seed": int(trajectories_per_seed),
        "dataset_name": name,
        "seed_map_spec": resolved_spec.to_dict(),
        "seed_map_generator_hash": seed_map_generator_hash(resolved_spec),
        "collection_config": dict(collection_config or {}),
    }

    if overwrite and root.exists():
        if root.name != name:
            raise ValueError(
                "Refusing to overwrite a seed-map corpus whose leaf directory does "
                f"not match the derived dataset name {name!r}: {root}"
            )
        shutil.rmtree(root)

    if manifest_path.exists():
        manifest = load_seed_map_manifest(root, require_complete=False)
        if _manifest_contract(manifest) != expected:
            raise ValueError(
                "Existing seed-map corpus contract differs from the requested "
                "generation config. Use --overwrite or choose another output root."
            )
        return repair_seed_map_corpus(root)

    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Seed-map corpus path exists without a valid manifest: {root}"
        )

    data_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(data_path, "w") as target:
        target.attrs["seed_map_corpus_schema_version"] = SEED_MAP_CORPUS_SCHEMA_VERSION
        target.attrs["total_episodes"] = 0
        target.attrs["total_steps"] = 0
    index_path.touch()
    manifest = {
        **expected,
        "complete": False,
        "expected_map_count": int(seed_end - seed_start),
        "expected_episode_count": int(
            (seed_end - seed_start) * trajectories_per_seed
        ),
        "generated_map_count": 0,
        "total_episodes": 0,
        "total_steps": 0,
        "content_hash": None,
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def repair_seed_map_corpus(
    dataset_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Repair an interrupted append at the HDF5/index transaction boundary.

    Appends write HDF5 first and index second. A killed process can therefore
    leave only unindexed trailing HDF5 groups; they are incomplete transaction
    artifacts and are removed before deterministic regeneration resumes.
    """

    root, manifest_path, _, data_path = _corpus_paths(dataset_path)
    manifest = load_seed_map_manifest(root, require_complete=False)
    records = read_seed_map_index(root)
    indexed_names = {str(record["episode_group"]) for record in records}
    removed_names = []
    with h5py.File(data_path, "a") as handle:
        actual_names = set(_episode_names(handle))
        missing_names = sorted(indexed_names - actual_names)
        if missing_names:
            raise ValueError(
                "Seed-map corpus index references missing HDF5 groups and cannot "
                f"be repaired automatically: {missing_names}"
            )
        removed_names = sorted(
            actual_names - indexed_names,
            key=lambda name: int(name.split("_", 1)[1]),
        )
        for name in removed_names:
            del handle[name]
        handle.attrs["total_episodes"] = len(records)
        handle.attrs["total_steps"] = sum(
            int(record["step_count"]) for record in records
        )
        handle.flush()

    manifest["generated_map_count"] = len(
        {int(record["map_seed"]) for record in records}
    )
    manifest["total_episodes"] = len(records)
    manifest["total_steps"] = sum(
        int(record["step_count"]) for record in records
    )
    if removed_names:
        manifest["complete"] = False
        manifest["content_hash"] = None
    _write_json_atomic(manifest_path, manifest)
    if removed_names:
        print(
            "[seed-map-corpus] Removed interrupted unindexed HDF5 episode "
            f"group(s) before resume: {removed_names}"
        )
    return manifest


def load_seed_map_manifest(
    dataset_path: str | os.PathLike[str],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    root, manifest_path, _, data_path = _corpus_paths(dataset_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Seed-map manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != SEED_MAP_CORPUS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported seed-map corpus schema at {manifest_path}: "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("source_kind") != SEED_MAP_SOURCE_KIND:
        raise ValueError(f"Not a seed-map corpus manifest: {manifest_path}")
    if require_complete and not bool(manifest.get("complete", False)):
        raise ValueError(
            f"Seed-map corpus is incomplete and cannot be loaded for training: {root}"
        )
    if not data_path.exists():
        raise FileNotFoundError(f"Seed-map corpus HDF5 file not found: {data_path}")
    normalized_spec = normalize_seed_map_spec(manifest.get("seed_map_spec"))
    expected_generator_hash = seed_map_generator_hash(normalized_spec)
    if manifest.get("seed_map_generator_hash") != expected_generator_hash:
        raise ValueError(
            "Seed-map manifest generator hash does not match seed_map_spec: "
            f"{manifest_path}"
        )
    return manifest


def read_seed_map_index(
    dataset_path: str | os.PathLike[str],
) -> list[dict[str, Any]]:
    _, _, index_path, _ = _corpus_paths(dataset_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Seed-map index not found: {index_path}")
    records = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid seed-map index JSON at {index_path}:{line_number}"
                ) from exc
            records.append(record)
    return records


def existing_seed_map_trajectory_indices(
    dataset_path: str | os.PathLike[str],
    map_seed: int,
) -> list[int]:
    map_seed = _require_int(map_seed, field="map_seed", minimum=0)
    return sorted(
        int(record["trajectory_index"])
        for record in read_seed_map_index(dataset_path)
        if int(record["map_seed"]) == map_seed
    )


def _episode_names(handle: h5py.File) -> list[str]:
    return sorted(
        (name for name in handle.keys() if name.startswith("episode_")),
        key=lambda name: int(name.split("_", 1)[1]),
    )


def _append_index_records(index_path: Path, records: list[dict[str, Any]]) -> None:
    with index_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(_canonical_json(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_minari_shard_to_seed_map_corpus(
    dataset_path: str | os.PathLike[str],
    shard_dataset_path: str | os.PathLike[str],
    *,
    map_seed: int,
    maze_map: list[list[int]],
    trajectory_start_index: int,
    trajectory_count: int,
    source_episode_start_index: int = 0,
) -> list[dict[str, Any]]:
    """Copy selected Minari shard episodes into the aggregate HDF5 corpus."""

    map_seed = _require_int(map_seed, field="map_seed", minimum=0)
    trajectory_start_index = _require_int(
        trajectory_start_index,
        field="trajectory_start_index",
        minimum=0,
    )
    trajectory_count = _require_int(
        trajectory_count,
        field="trajectory_count",
        minimum=1,
    )
    source_episode_start_index = _require_int(
        source_episode_start_index,
        field="source_episode_start_index",
        minimum=0,
    )
    root, manifest_path, index_path, data_path = _corpus_paths(dataset_path)
    manifest = load_seed_map_manifest(root, require_complete=False)
    if bool(manifest.get("complete", False)):
        raise ValueError(f"Cannot append to completed seed-map corpus: {root}")
    if map_seed < int(manifest["seed_start"]) or map_seed >= int(manifest["seed_end"]):
        raise ValueError(
            f"map_seed={map_seed} is outside corpus range "
            f"[{manifest['seed_start']}, {manifest['seed_end']})"
        )

    existing_records = read_seed_map_index(root)
    existing_keys = {
        (int(record["map_seed"]), int(record["trajectory_index"]))
        for record in existing_records
    }
    requested_keys = {
        (map_seed, trajectory_start_index + offset)
        for offset in range(trajectory_count)
    }
    duplicates = sorted(requested_keys & existing_keys)
    if duplicates:
        raise ValueError(f"Seed-map corpus already contains trajectory keys: {duplicates}")

    shard_root = Path(shard_dataset_path).expanduser().resolve()
    source_path = (
        shard_root
        if shard_root.is_file()
        else shard_root / "data" / "main_data.hdf5"
    )
    if not source_path.exists():
        raise FileNotFoundError(f"Minari shard HDF5 file not found: {source_path}")

    current_total = len(existing_records)
    map_digest = seed_map_hash(maze_map)
    new_records = []
    with h5py.File(source_path, "r") as source, h5py.File(data_path, "a") as target:
        source_names = _episode_names(source)
        source_end = source_episode_start_index + trajectory_count
        if len(source_names) < source_end:
            raise ValueError(
                f"Minari shard has {len(source_names)} episodes but "
                f"episodes [{source_episode_start_index}, {source_end}) are "
                f"required for map_seed={map_seed}"
            )
        selected_source_names = source_names[source_episode_start_index:source_end]
        for offset, source_name in enumerate(selected_source_names):
            global_episode_index = current_total + offset
            target_name = f"episode_{global_episode_index}"
            if target_name in target:
                raise ValueError(
                    f"Target seed-map HDF5 group already exists: {target_name}"
                )
            source.copy(source[source_name], target, name=target_name)
            target_group = target[target_name]
            trajectory_index = trajectory_start_index + offset
            target_group.attrs["seed_map.map_seed"] = map_seed
            target_group.attrs["seed_map.trajectory_index"] = trajectory_index
            target_group.attrs["seed_map.map_hash"] = map_digest
            target_group.attrs["seed_map.map_rows"] = len(maze_map)
            target_group.attrs["seed_map.map_cols"] = len(maze_map[0])
            step_count = int(target_group["actions"].shape[0])
            new_records.append(
                {
                    "episode_index": global_episode_index,
                    "episode_group": target_name,
                    "map_seed": map_seed,
                    "trajectory_index": trajectory_index,
                    "map_hash": map_digest,
                    "map_rows": len(maze_map),
                    "map_cols": len(maze_map[0]),
                    "step_count": step_count,
                }
            )
        total_episodes = current_total + len(new_records)
        total_steps = sum(
            int(record["step_count"]) for record in existing_records + new_records
        )
        target.attrs["total_episodes"] = total_episodes
        target.attrs["total_steps"] = total_steps
        target.flush()

    _append_index_records(index_path, new_records)
    all_records = existing_records + new_records
    manifest["generated_map_count"] = len(
        {int(record["map_seed"]) for record in all_records}
    )
    manifest["total_episodes"] = len(all_records)
    manifest["total_steps"] = sum(int(record["step_count"]) for record in all_records)
    manifest["content_hash"] = None
    _write_json_atomic(manifest_path, manifest)
    return new_records


def validate_seed_map_corpus(
    dataset_path: str | os.PathLike[str],
    *,
    require_complete: bool = False,
    verify_map_hashes: bool = True,
) -> dict[str, Any]:
    root, _, _, data_path = _corpus_paths(dataset_path)
    manifest = load_seed_map_manifest(root, require_complete=False)
    records = read_seed_map_index(root)
    keys = [
        (int(record["map_seed"]), int(record["trajectory_index"]))
        for record in records
    ]
    duplicate_keys = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicate_keys:
        raise ValueError(f"Seed-map corpus index contains duplicate keys: {duplicate_keys}")

    expected_group_names = {str(record["episode_group"]) for record in records}
    with h5py.File(data_path, "r") as handle:
        actual_group_names = set(_episode_names(handle))
        if actual_group_names != expected_group_names:
            raise ValueError(
                "Seed-map HDF5/index episode group mismatch: "
                f"missing={sorted(expected_group_names - actual_group_names)}, "
                f"unindexed={sorted(actual_group_names - expected_group_names)}"
            )
        for record in records:
            group = handle[str(record["episode_group"])]
            if int(group["actions"].shape[0]) != int(record["step_count"]):
                raise ValueError(
                    "Seed-map episode step count mismatch for "
                    f"{record['episode_group']}"
                )
            if int(group.attrs["seed_map.map_seed"]) != int(record["map_seed"]):
                raise ValueError(
                    f"Seed-map HDF5 map_seed mismatch for {record['episode_group']}"
                )

    if verify_map_hashes:
        spec = normalize_seed_map_spec(manifest["seed_map_spec"])
        hashes_by_seed: dict[int, str] = {}
        for record in records:
            map_seed = int(record["map_seed"])
            expected_hash = hashes_by_seed.get(map_seed)
            if expected_hash is None:
                expected_hash = seed_map_hash(generate_seed_map(map_seed, spec))
                hashes_by_seed[map_seed] = expected_hash
            if record["map_hash"] != expected_hash:
                raise ValueError(
                    f"Seed-map hash mismatch for map_seed={map_seed}: "
                    f"index={record['map_hash']}, generated={expected_hash}"
                )

    counts = Counter(int(record["map_seed"]) for record in records)
    capacity = int(manifest["trajectories_per_seed"])
    over_capacity = sorted(
        (seed, count) for seed, count in counts.items() if count > capacity
    )
    if over_capacity:
        raise ValueError(f"Seed-map corpus exceeds per-seed capacity: {over_capacity}")

    expected_seeds = set(range(int(manifest["seed_start"]), int(manifest["seed_end"])))
    complete_now = (
        set(counts) == expected_seeds
        and all(counts[seed] == capacity for seed in expected_seeds)
    )
    if require_complete and not complete_now:
        missing = sorted(seed for seed in expected_seeds if counts[seed] < capacity)
        raise ValueError(
            "Seed-map corpus is incomplete: "
            f"{len(missing)} seed(s) below trajectories_per_seed={capacity}; "
            f"first_missing={missing[:10]}"
        )

    total_steps = sum(int(record["step_count"]) for record in records)
    if int(manifest.get("total_episodes", -1)) != len(records):
        raise ValueError("Seed-map manifest total_episodes does not match index")
    if int(manifest.get("total_steps", -1)) != total_steps:
        raise ValueError("Seed-map manifest total_steps does not match index")
    return {
        "complete": complete_now,
        "map_count": len(counts),
        "episode_count": len(records),
        "total_steps": total_steps,
    }


def finalize_seed_map_corpus(
    dataset_path: str | os.PathLike[str],
) -> dict[str, Any]:
    root, manifest_path, index_path, _ = _corpus_paths(dataset_path)
    stats = validate_seed_map_corpus(root, require_complete=True)
    manifest = load_seed_map_manifest(root, require_complete=False)
    content_hasher = hashlib.sha256()
    content_hasher.update(_canonical_json(_manifest_contract(manifest)).encode("utf-8"))
    with index_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            content_hasher.update(chunk)
    manifest["complete"] = True
    manifest["generated_map_count"] = stats["map_count"]
    manifest["total_episodes"] = stats["episode_count"]
    manifest["total_steps"] = stats["total_steps"]
    manifest["content_hash"] = content_hasher.hexdigest()
    _write_json_atomic(manifest_path, manifest)
    return manifest


def normalize_seed_ranges(
    value: Any,
    *,
    default_start: int,
    default_end: int,
    field: str = "seed_ranges",
) -> tuple[tuple[int, int], ...]:
    if value is None:
        return ((int(default_start), int(default_end)),)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list of [start, end] ranges")
    ranges = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{field}[{index}] must be [start, end], got {item!r}")
        start = _require_int(item[0], field=f"{field}[{index}][0]", minimum=0)
        end = _require_int(item[1], field=f"{field}[{index}][1]", minimum=1)
        if start >= end:
            raise ValueError(
                f"{field}[{index}] must be a non-empty half-open range, "
                f"got [{start}, {end})"
            )
        if start < default_start or end > default_end:
            raise ValueError(
                f"{field}[{index}] [{start}, {end}) is outside corpus range "
                f"[{default_start}, {default_end})"
            )
        ranges.append((start, end))
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] < previous[1]:
            raise ValueError(
                f"{field} contains overlapping ranges: {previous} and {current}"
            )
    return tuple(ranges)


@dataclass(frozen=True)
class SeedMapSelection:
    dataset_path: str
    env_family: str
    seed_ranges: tuple[tuple[int, int], ...]
    seed_count: int
    trajectories_per_seed: int
    selection_seed: int
    split_unit: str
    selected_seeds: tuple[int, ...]
    selected_keys: tuple[tuple[int, int], ...]
    manifest_hash: str
    selection_hash: str

    @property
    def source_tag(self) -> str:
        return (
            f"seed-map-n{self.seed_count}-m{self.trajectories_per_seed}-"
            f"{self.selection_hash[:10]}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "env_family": self.env_family,
            "seed_ranges": [list(item) for item in self.seed_ranges],
            "seed_count": self.seed_count,
            "trajectories_per_seed": self.trajectories_per_seed,
            "selection_seed": self.selection_seed,
            "split_unit": self.split_unit,
            "selected_seeds": list(self.selected_seeds),
            "selected_keys": [list(item) for item in self.selected_keys],
            "manifest_hash": self.manifest_hash,
            "selection_hash": self.selection_hash,
            "source_tag": self.source_tag,
        }


def resolve_seed_map_selection(
    section: Mapping[str, Any],
    *,
    env_family: str,
) -> SeedMapSelection:
    """Validate ``seed_map_train`` and deterministically select corpus records."""

    if not isinstance(section, Mapping):
        raise ValueError(
            f"seed_map_train must be a mapping, got {type(section).__name__}"
        )
    raw = dict(section)
    allowed = {
        "enabled",
        "dataset_path",
        "seed_ranges",
        "seed_count",
        "trajectories_per_seed",
        "selection_seed",
        "split_unit",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown seed_map_train fields: {unknown}")
    if raw.get("enabled") is not True:
        raise ValueError("resolve_seed_map_selection requires seed_map_train.enabled=true")
    dataset_path_value = raw.get("dataset_path")
    if not isinstance(dataset_path_value, (str, os.PathLike)) or not os.fspath(
        dataset_path_value
    ):
        raise ValueError(
            "seed_map_train.enabled=true requires a non-empty dataset_path"
        )
    dataset_path = str(Path(dataset_path_value).expanduser().resolve())
    manifest = load_seed_map_manifest(dataset_path, require_complete=True)
    if manifest["env_family"] != env_family:
        raise ValueError(
            f"seed_map_train corpus env_family={manifest['env_family']!r} "
            f"does not match training env_family={env_family!r}"
        )
    ranges = normalize_seed_ranges(
        raw.get("seed_ranges"),
        default_start=int(manifest["seed_start"]),
        default_end=int(manifest["seed_end"]),
        field="seed_map_train.seed_ranges",
    )
    allowed_seeds = [
        seed
        for start, end in ranges
        for seed in range(start, end)
    ]
    requested_seed_count = raw.get("seed_count")
    if requested_seed_count is None:
        seed_count = len(allowed_seeds)
    else:
        seed_count = _require_int(
            requested_seed_count,
            field="seed_map_train.seed_count",
            minimum=1,
        )
    if seed_count > len(allowed_seeds):
        raise ValueError(
            f"seed_map_train.seed_count={seed_count} exceeds "
            f"{len(allowed_seeds)} seeds in seed_ranges"
        )
    trajectories_per_seed = _require_int(
        raw.get("trajectories_per_seed"),
        field="seed_map_train.trajectories_per_seed",
        minimum=1,
    )
    capacity = int(manifest["trajectories_per_seed"])
    if trajectories_per_seed > capacity:
        raise ValueError(
            "seed_map_train.trajectories_per_seed exceeds corpus capacity: "
            f"requested={trajectories_per_seed}, capacity={capacity}"
        )
    selection_seed = _require_int(
        raw.get("selection_seed", 0),
        field="seed_map_train.selection_seed",
        minimum=0,
    )
    split_unit = str(raw.get("split_unit", "seed")).strip()
    if split_unit not in {"seed", "trajectory"}:
        raise ValueError(
            "seed_map_train.split_unit must be 'seed' or 'trajectory', "
            f"got {split_unit!r}"
        )

    ordered_seeds = sorted(
        allowed_seeds,
        key=lambda seed: (
            stable_seed_map_int("seed_map_train", selection_seed, "map", seed),
            seed,
        ),
    )
    selected_seeds = tuple(sorted(ordered_seeds[:seed_count]))

    records = read_seed_map_index(dataset_path)
    records_by_seed: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        records_by_seed.setdefault(int(record["map_seed"]), []).append(record)
    selected_keys = []
    for map_seed in selected_seeds:
        seed_records = records_by_seed.get(map_seed, [])
        if len(seed_records) < trajectories_per_seed:
            raise ValueError(
                f"Seed-map corpus map_seed={map_seed} has {len(seed_records)} "
                f"trajectories; {trajectories_per_seed} requested"
            )
        ordered_records = sorted(
            seed_records,
            key=lambda record: (
                stable_seed_map_int(
                    "seed_map_train",
                    selection_seed,
                    "trajectory",
                    map_seed,
                    int(record["trajectory_index"]),
                ),
                int(record["trajectory_index"]),
            ),
        )
        selected_keys.extend(
            (map_seed, int(record["trajectory_index"]))
            for record in ordered_records[:trajectories_per_seed]
        )
    selected_keys_tuple = tuple(sorted(selected_keys))
    manifest_hash = _hash_json(manifest)
    selection_payload = {
        "dataset_path": dataset_path,
        "manifest_hash": manifest_hash,
        "seed_ranges": ranges,
        "seed_count": seed_count,
        "trajectories_per_seed": trajectories_per_seed,
        "selection_seed": selection_seed,
        "split_unit": split_unit,
        "selected_seeds": selected_seeds,
        "selected_keys": selected_keys_tuple,
    }
    return SeedMapSelection(
        dataset_path=dataset_path,
        env_family=env_family,
        seed_ranges=ranges,
        seed_count=seed_count,
        trajectories_per_seed=trajectories_per_seed,
        selection_seed=selection_seed,
        split_unit=split_unit,
        selected_seeds=selected_seeds,
        selected_keys=selected_keys_tuple,
        manifest_hash=manifest_hash,
        selection_hash=_hash_json(selection_payload),
    )


def _read_hdf5_tree(node):
    if isinstance(node, h5py.Dataset):
        return node[()]
    return {key: _read_hdf5_tree(node[key]) for key in node.keys()}


def load_seed_map_selected_episodes(
    selection: SeedMapSelection,
) -> tuple[list[SimpleNamespace], list[int], list[dict[str, Any]]]:
    """Load only the selected HDF5 episodes and attach per-map prompt vars."""

    manifest = load_seed_map_manifest(selection.dataset_path, require_complete=True)
    records = read_seed_map_index(selection.dataset_path)
    records_by_key = {
        (int(record["map_seed"]), int(record["trajectory_index"])): record
        for record in records
    }
    missing = [key for key in selection.selected_keys if key not in records_by_key]
    if missing:
        raise ValueError(
            f"Seed-map selection references missing corpus trajectories: {missing[:10]}"
        )
    spec = normalize_seed_map_spec(manifest["seed_map_spec"])
    _, _, _, data_path = _corpus_paths(selection.dataset_path)
    map_cache: dict[int, tuple[list[list[int]], dict[str, Any], str]] = {}
    episodes = []
    selected_records = []
    with h5py.File(data_path, "r") as handle:
        for map_seed, trajectory_index in selection.selected_keys:
            record = records_by_key[(map_seed, trajectory_index)]
            if map_seed not in map_cache:
                maze_map = generate_seed_map(map_seed, spec)
                map_digest = seed_map_hash(maze_map)
                if map_digest != record["map_hash"]:
                    raise ValueError(
                        f"Seed-map record hash mismatch for map_seed={map_seed}"
                    )
                prompt_vars = build_seed_map_prompt_vars(
                    selection.env_family,
                    map_seed=map_seed,
                    maze_map=maze_map,
                    reward_type=manifest["reward_type"],
                )
                map_cache[map_seed] = (maze_map, prompt_vars, map_digest)
            _, prompt_vars, map_digest = map_cache[map_seed]
            group = handle[str(record["episode_group"])]
            kwargs = {
                "observations": _read_hdf5_tree(group["observations"]),
                "actions": np.asarray(group["actions"][()]),
                "seed_map_prompt_vars": prompt_vars,
                "seed_map_metadata": {
                    "map_seed": map_seed,
                    "trajectory_index": trajectory_index,
                    "map_hash": map_digest,
                },
            }
            for field in ("rewards", "terminations", "truncations", "infos"):
                if field in group:
                    kwargs[field] = _read_hdf5_tree(group[field])
            episodes.append(SimpleNamespace(**kwargs))
            selected_records.append(record)
    step_counts = [len(episode.actions) for episode in episodes]
    return episodes, step_counts, selected_records


def build_seed_map_episode_selection(
    selection: SeedMapSelection,
    *,
    train_data_ratio: float,
    episode_transform=None,
) -> dict[str, Any]:
    """Return a PointMazeDataset-compatible episode selection dictionary."""

    if not (0.0 < train_data_ratio < 1.0):
        raise ValueError(
            f"train_data_ratio must satisfy 0 < ratio < 1, got {train_data_ratio}"
        )
    episodes, step_counts, records = load_seed_map_selected_episodes(selection)
    if episode_transform is not None:
        transformed_episodes = []
        transformed_records = []
        for episode, record in zip(episodes, records, strict=True):
            transformed = episode_transform(episode)
            if transformed is None:
                continue
            transformed_episodes.append(transformed)
            transformed_records.append(record)
        episodes = transformed_episodes
        records = transformed_records
        step_counts = [len(episode.actions) for episode in episodes]
        if not episodes:
            raise ValueError(
                "Seed-map family data preprocessing removed every selected episode"
            )
        retained_seeds = {int(record["map_seed"]) for record in records}
        missing_seeds = sorted(set(selection.selected_seeds) - retained_seeds)
        if missing_seeds:
            raise ValueError(
                "Seed-map family data preprocessing removed every trajectory "
                f"from selected map seed(s): {missing_seeds}"
            )
    indices = list(range(len(episodes)))
    if selection.split_unit == "seed":
        split_order = sorted(
            selection.selected_seeds,
            key=lambda seed: (
                stable_seed_map_int(
                    "seed_map_train",
                    selection.selection_seed,
                    "split",
                    seed,
                ),
                seed,
            ),
        )
        train_seed_count = math.floor(len(split_order) * train_data_ratio)
        if train_seed_count < 1:
            raise ValueError(
                "seed_map_train and train_data_ratio select zero train maps: "
                f"seed_count={len(split_order)}, train_data_ratio={train_data_ratio}"
            )
        train_seeds = set(split_order[:train_seed_count])
        train_indices = [
            index
            for index, record in enumerate(records)
            if int(record["map_seed"]) in train_seeds
        ]
        val_indices = [
            index
            for index, record in enumerate(records)
            if int(record["map_seed"]) not in train_seeds
        ]
    else:
        split_order = sorted(
            indices,
            key=lambda index: (
                stable_seed_map_int(
                    "seed_map_train",
                    selection.selection_seed,
                    "trajectory_split",
                    int(records[index]["map_seed"]),
                    int(records[index]["trajectory_index"]),
                ),
                index,
            ),
        )
        train_count = math.floor(len(split_order) * train_data_ratio)
        if train_count < 1:
            raise ValueError(
                "seed_map_train and train_data_ratio select zero train trajectories"
            )
        train_indices = sorted(split_order[:train_count])
        val_indices = sorted(split_order[train_count:])

    train_steps = sum(step_counts[index] for index in train_indices)
    val_steps = sum(step_counts[index] for index in val_indices)
    return {
        "variant": selection.source_tag,
        "episodes": episodes,
        "records": records,
        "total_episodes": len(episodes),
        "total_steps": sum(step_counts),
        "initial_train_target": len(episodes),
        "initial_sampled_target": len(episodes),
        "sampled_episode_count": len(episodes),
        "balanced_train_target": None,
        "train_indices": sorted(train_indices),
        "val_indices": sorted(val_indices),
        "train_episode_count": len(train_indices),
        "val_episode_count": len(val_indices),
        "train_steps": train_steps,
        "val_steps": val_steps,
        "val_target": len(val_indices),
        "val_shortfall_reason": None,
        "seed_map_selection": selection.to_dict(),
    }
