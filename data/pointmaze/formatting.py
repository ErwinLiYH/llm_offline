import re
import numpy as np


def format_obs(obs: np.ndarray, goal: np.ndarray) -> str:
    """Serialize PointMaze obs and goal into text for prompt insertion.

    obs: [x, y, vx, vy]
    goal: [gx, gy]
    """
    x, y, vx, vy = obs
    gx, gy = goal
    return (
        f"  Position: (x={x:.4f}, y={y:.4f})\n"
        f"  Velocity: (vx={vx:.4f}, vy={vy:.4f})\n"
        f"  Goal:     (gx={gx:.4f}, gy={gy:.4f})"
    )


def format_action(action: np.ndarray) -> str:
    """Serialize a 2D action vector into the training target text."""
    ax, ay = action
    return f"{int(np.clip(np.round(ax * 100), -100, 100))},{int(np.clip(np.round(ay * 100), -100, 100))}"


_ACTION_PATTERN = re.compile(
    r"[-+]?\d+\s*,\s*[-+]?\d+"
)


def parse_action(text: str) -> tuple[np.ndarray, bool]:
    """Parse model output text into a 2D action vector.

    Returns (action, success). On failure returns (zeros, False).
    """
    match = _ACTION_PATTERN.search(text)
    if match is None:
        return np.zeros(2, dtype=np.float32), False
    try:
        parts = match.group(0).split(",")
        ax = float(int(parts[0].strip())) / 100.0
        ay = float(int(parts[1].strip())) / 100.0
        action = np.array([ax, ay], dtype=np.float32)
        return action, True
    except (ValueError, IndexError):
        return np.zeros(2, dtype=np.float32), False


def validate_action(action: np.ndarray) -> bool:
    """Return True if all action components are within [-1, 1]."""
    return bool(np.all(action >= -1.0) and np.all(action <= 1.0))
