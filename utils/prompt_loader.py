import os
from string import Formatter
from typing import List


def _get_prompt_dir(env_family: str) -> str:
    prompt_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "prompts",
        env_family,
    )
    if not os.path.isdir(prompt_dir):
        raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")
    return prompt_dir


def load_template_map(env_family: str) -> dict[str, str]:
    """Load shared prompt templates keyed by prompt filename stem."""
    prompt_dir = _get_prompt_dir(env_family)

    templates: dict[str, str] = {}
    for name in os.listdir(prompt_dir):
        stem, ext = os.path.splitext(name)
        if ext != ".txt":
            continue
        path = os.path.join(prompt_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            templates[stem] = f.read()

    if not templates:
        raise ValueError(f"No prompt templates found in {prompt_dir}")
    return templates


def load_template_names(env_family: str) -> list[str]:
    """Return available prompt template names in deterministic order."""
    return sorted(load_template_map(env_family))


def load_named_templates(env_family: str, prompt_names: list[str]) -> list[str]:
    """Load prompt templates by filename stem, preserving the requested order."""
    templates_by_name = load_template_map(env_family)
    missing = [name for name in prompt_names if name not in templates_by_name]
    if missing:
        available = ", ".join(sorted(templates_by_name))
        raise ValueError(
            f"Unknown prompt template names for {env_family}: {missing}. Available: {available}"
        )
    return [templates_by_name[name] for name in prompt_names]


def load_templates(env_family: str) -> List[str]:
    """Load all shared prompt templates for an environment family by filename order."""
    templates_by_name = load_template_map(env_family)
    return [templates_by_name[name] for name in sorted(templates_by_name)]


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
