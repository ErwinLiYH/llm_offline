"""Helpers for saving per-experiment runtime configuration snapshots."""

from __future__ import annotations

import hashlib
import os
import subprocess

import yaml


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _empty_git_metadata(*, patch_file: str, error: str | None = None) -> dict:
    metadata = {
        "available": False,
        "repo_root": None,
        "branch": None,
        "head_commit": None,
        "head_subject": None,
        "head_author_date": None,
        "head_commit_date": None,
        "status_porcelain": [],
        "dirty": False,
        "patch_file": patch_file,
        "patch_bytes": 0,
        "patch_sha256": EMPTY_SHA256,
        "skipped_files": [],
    }
    if error:
        metadata["error"] = error
    return metadata


def _run_git(args: list[str], *, cwd: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _git_stdout(args: list[str], *, cwd: str) -> bytes:
    result = _run_git(args, cwd=cwd)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or f"git {' '.join(args)} failed")
    return result.stdout


def _git_text(args: list[str], *, cwd: str) -> str:
    return _git_stdout(args, cwd=cwd).decode("utf-8", errors="replace")


def _nul_split(data: bytes) -> list[str]:
    return [
        part.decode("utf-8", errors="replace")
        for part in data.split(b"\0")
        if part
    ]


def _is_binary_tracked_diff(repo_root: str, path: str) -> bool:
    numstat = _git_stdout(
        ["diff", "--no-renames", "--numstat", "-z", "HEAD", "--", path],
        cwd=repo_root,
    )
    return numstat.startswith(b"-\t-\t")


def _is_binary_file(path: str) -> bool:
    with open(path, "rb") as f:
        sample = f.read(8192)
    return b"\0" in sample


def _collect_tracked_text_diff(repo_root: str) -> tuple[bytes, list[dict]]:
    changed_paths = _nul_split(
        _git_stdout(
            ["diff", "--no-renames", "--name-only", "-z", "HEAD", "--", "."],
            cwd=repo_root,
        )
    )
    patch_parts: list[bytes] = []
    skipped_files: list[dict] = []
    for path in changed_paths:
        if _is_binary_tracked_diff(repo_root, path):
            skipped_files.append({"path": path, "reason": "binary"})
            continue
        patch = _git_stdout(
            ["diff", "--no-renames", "--no-ext-diff", "HEAD", "--", path],
            cwd=repo_root,
        )
        if patch:
            patch_parts.append(patch)
    return b"".join(patch_parts), skipped_files


def _collect_untracked_text_diff(repo_root: str) -> tuple[bytes, list[dict]]:
    untracked_paths = _nul_split(
        _git_stdout(
            ["ls-files", "--others", "--exclude-standard", "-z"],
            cwd=repo_root,
        )
    )
    patch_parts: list[bytes] = []
    skipped_files: list[dict] = []
    for path in untracked_paths:
        abs_path = os.path.join(repo_root, path)
        if _is_binary_file(abs_path):
            skipped_files.append({"path": path, "reason": "binary"})
            continue
        result = _run_git(
            ["diff", "--no-index", "--no-ext-diff", "--", "/dev/null", path],
            cwd=repo_root,
        )
        if result.returncode not in (0, 1):
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(stderr or f"git diff --no-index failed for {path}")
        if result.stdout:
            patch_parts.append(result.stdout)
    return b"".join(patch_parts), skipped_files


def _collect_git_snapshot(repo_dir: str, *, patch_file: str) -> tuple[dict, bytes]:
    try:
        repo_root = _git_text(["rev-parse", "--show-toplevel"], cwd=repo_dir).strip()
        branch = _git_text(["branch", "--show-current"], cwd=repo_root).strip() or None
        head_commit = _git_text(["rev-parse", "HEAD"], cwd=repo_root).strip()
        head_subject = _git_text(
            ["show", "-s", "--format=%s", "HEAD"],
            cwd=repo_root,
        ).rstrip("\n")
        head_author_date = _git_text(
            ["show", "-s", "--format=%aI", "HEAD"],
            cwd=repo_root,
        ).strip()
        head_commit_date = _git_text(
            ["show", "-s", "--format=%cI", "HEAD"],
            cwd=repo_root,
        ).strip()
        status_porcelain = _git_text(
            ["status", "--porcelain"],
            cwd=repo_root,
        ).splitlines()
        tracked_patch, tracked_skipped = _collect_tracked_text_diff(repo_root)
        untracked_patch, untracked_skipped = _collect_untracked_text_diff(repo_root)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        return _empty_git_metadata(patch_file=patch_file, error=str(exc)), b""

    patch = tracked_patch
    if patch and untracked_patch and not patch.endswith(b"\n"):
        patch += b"\n"
    patch += untracked_patch
    patch_sha256 = hashlib.sha256(patch).hexdigest()
    metadata = {
        "available": True,
        "repo_root": repo_root,
        "branch": branch,
        "head_commit": head_commit,
        "head_subject": head_subject,
        "head_author_date": head_author_date,
        "head_commit_date": head_commit_date,
        "status_porcelain": status_porcelain,
        "dirty": bool(status_porcelain),
        "patch_file": patch_file,
        "patch_bytes": len(patch),
        "patch_sha256": patch_sha256,
        "skipped_files": tracked_skipped + untracked_skipped,
    }
    return metadata, patch


def save_experiment_config_snapshot(
    config: dict,
    *,
    root: str = "exp_configs",
    filename: str = "config.yaml",
    repo_dir: str | None = None,
) -> dict[str, str]:
    experiment_id = str(config.get("experiment_id") or "").strip()
    if not experiment_id:
        raise ValueError("Cannot save experiment config snapshot without experiment_id.")
    if not filename or os.path.basename(filename) != filename:
        raise ValueError(f"filename must be a plain file name, got {filename!r}")

    config_dir = os.path.join(root, experiment_id)
    git_path = os.path.join(config_dir, "git.yaml")
    patch_path = os.path.join(config_dir, "dirty.patch")
    git_metadata, dirty_patch = _collect_git_snapshot(
        repo_dir or os.getcwd(),
        patch_file="dirty.patch",
    )

    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, filename)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    with open(patch_path, "wb") as f:
        f.write(dirty_patch)
    with open(git_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(git_metadata, f, sort_keys=False, allow_unicode=True)
    return {
        "config": config_path,
        "git": git_path,
        "patch": patch_path,
    }
