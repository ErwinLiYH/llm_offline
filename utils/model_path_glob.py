from __future__ import annotations

import glob


def resolve_trailing_wildcard_path(path: str, *, field_name: str = "path") -> str:
    """Resolve a path that may contain one trailing '*' wildcard."""
    if not isinstance(path, str):
        raise TypeError(f"{field_name} must be a string, got {type(path).__name__}")
    if "*" not in path:
        return path
    if not path.endswith("*") or path.count("*") != 1:
        raise ValueError(
            f"{field_name} wildcard only supports a single trailing '*', got {path!r}"
        )

    pattern = glob.escape(path[:-1]) + "*"
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"{field_name} wildcard matched no paths: {path!r}")
    if len(matches) > 1:
        lines = [
            f"{field_name} wildcard matched multiple paths; use a more specific pattern.",
            f"pattern: {path!r}",
            "matches:",
        ]
        lines.extend(f"  {match}" for match in matches)
        raise ValueError("\n".join(lines))
    return matches[0]
