import os
from string import Formatter
from typing import List


def load_templates(env_family: str) -> List[str]:
    """Load shared prompt templates for an environment family.

    Templates live at prompts/<env_family>/<idx>.txt and are loaded in ascending
    numeric index order. Indices must be contiguous starting from 0.
    """
    prompt_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "prompts",
        env_family,
    )
    if not os.path.isdir(prompt_dir):
        raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")

    indexed_paths: list[tuple[int, str]] = []
    for name in os.listdir(prompt_dir):
        stem, ext = os.path.splitext(name)
        if ext != ".txt":
            continue
        if not stem.isdigit():
            raise ValueError(f"Prompt template filename must be numeric: {name}")
        indexed_paths.append((int(stem), os.path.join(prompt_dir, name)))

    if not indexed_paths:
        raise ValueError(f"No prompt templates found in {prompt_dir}")

    indexed_paths.sort()
    expected = list(range(len(indexed_paths)))
    actual = [idx for idx, _ in indexed_paths]
    if actual != expected:
        raise ValueError(
            f"Prompt template indices must be contiguous from 0 in {prompt_dir}; got {actual}"
        )

    templates = []
    for _, path in indexed_paths:
        with open(path, "r", encoding="utf-8") as f:
            templates.append(f.read())
    return templates


def render_template(template: str, prompt_vars: dict, **extra_vars) -> str:
    """Render a prompt template with strict missing-variable validation."""
    values = dict(prompt_vars)
    values.update(extra_vars)

    field_names = {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name
    }
    missing = sorted(name for name in field_names if name not in values)
    if missing:
        raise KeyError(f"Missing prompt template variables: {missing}")

    return template.format(**values)
