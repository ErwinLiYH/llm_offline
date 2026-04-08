import os
from typing import List

import yaml


def load_templates(env_family: str, variant: str) -> List[str]:
    """Load prompt templates for the given env_family and variant.

    Returns the template strings exactly as listed in the variant YAML file.
    Evaluation uses template 0; training may use the first N templates.
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
    if not isinstance(templates, list) or not templates:
        raise ValueError(f"Expected a non-empty template list in {path}")
    return templates
