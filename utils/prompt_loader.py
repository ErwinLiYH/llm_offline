import os
import yaml
from typing import List


def load_templates(env_family: str, variant: str) -> List[str]:
    """Load all 5 prompt templates for the given env_family and variant.

    Returns a list of 5 template strings (indices 0-4).
    Templates 0-2 are English; 3-4 are Chinese.
    """
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "prompts",
        env_family,
        f"{variant}.yaml",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt template not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    templates = data["templates"]
    if len(templates) != 5:
        raise ValueError(
            f"Expected 5 templates in {path}, got {len(templates)}"
        )
    return templates
