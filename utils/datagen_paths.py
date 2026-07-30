"""Shared final and temporary path handling for local data generators."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
import re
import shutil
import tempfile
from pathlib import Path
import uuid


def add_datagen_path_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        "--seed-map-dataset-root",
        dest="dataset_root",
        type=Path,
        default=None,
        help=(
            "Parent directory for final generated datasets, or an exact dataset "
            "directory whose leaf matches the derived dataset name. Seed-map "
            "parent roots retain the family/version/size/reward namespace before "
            "the dataset leaf. Defaults to the current registered/seed-map output "
            "location. "
            "--seed-map-dataset-root is a compatibility alias."
        ),
    )
    parser.add_argument(
        "--temporary-dataset-root",
        "--temp-dataset-root",
        dest="temporary_dataset_root",
        type=Path,
        default=None,
        help=(
            "Parent directory for intermediate Minari shard datasets. Defaults "
            "to the TMPDIR environment variable, or the platform temporary "
            "directory when TMPDIR is unset."
        ),
    )


def resolve_final_dataset_path(
    default_path: str | Path,
    configured_root: str | Path | None,
) -> Path:
    """Resolve a final dataset path without changing its derived leaf name."""

    default = Path(default_path).expanduser()
    if configured_root is None:
        return default.resolve()
    configured = Path(configured_root).expanduser()
    resolved = configured if configured.name == default.name else configured / default.name
    return resolved.resolve()


def resolve_temporary_dataset_root(
    configured_root: str | Path | None,
) -> Path:
    """Resolve the root under which Minari creates temporary shard datasets."""

    if configured_root is not None:
        return Path(configured_root).expanduser().resolve()
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        return Path(tmpdir).expanduser().resolve()
    return Path(tempfile.gettempdir()).resolve()


def configure_minari_temporary_root(root: str | Path) -> Path:
    """Create and activate a worker-local Minari dataset root."""

    resolved = Path(root).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ["MINARI_DATASETS_PATH"] = str(resolved)
    return resolved


@contextmanager
def temporary_dataset_merge_path(
    root: str | Path,
    *,
    label: str,
):
    """Yield a fresh dataset path for shard merging under the temporary root."""

    resolved_root = configure_minari_temporary_root(root)
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(label)).strip(".-")
    if not safe_label:
        safe_label = "dataset"
    workspace = Path(
        tempfile.mkdtemp(
            prefix=f".{safe_label}-merge-",
            dir=resolved_root,
        )
    ).resolve()
    try:
        yield workspace / "dataset"
    finally:
        if workspace.exists():
            shutil.rmtree(workspace)


def publish_new_dataset(
    staged_dataset_path: str | Path,
    final_dataset_path: str | Path,
) -> None:
    """Transfer a completed staged dataset and atomically expose its final name."""

    staged = Path(staged_dataset_path).expanduser().resolve()
    final = Path(final_dataset_path).expanduser().resolve()
    if not staged.is_dir():
        raise FileNotFoundError(f"Staged dataset directory not found: {staged}")
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset path: {final}")

    final.parent.mkdir(parents=True, exist_ok=True)
    transfer = final.with_name(f".{final.name}.publishing-{uuid.uuid4().hex}")
    try:
        shutil.copytree(staged, transfer)
        os.replace(transfer, final)
    finally:
        if transfer.exists():
            shutil.rmtree(transfer)


def remove_temporary_dataset(dataset_id: str, dataset_path: str | Path) -> None:
    """Remove one generated shard after validating its exact dataset leaf."""

    path = Path(dataset_path).expanduser().resolve()
    if path.name != dataset_id:
        raise ValueError(
            "Refusing to remove temporary dataset whose path leaf differs from "
            f"its dataset id: dataset_id={dataset_id!r}, path={path}"
        )
    if path.exists():
        shutil.rmtree(path)
