---
name: llm-offline-env-support
description: Use this skill when adding or updating environment-family support or variant support in the llm_offline project. It covers when to add a new variant versus a new env family, which files must change, how prompts are wired through shared family templates plus variant prompt_vars, how formatter/decoder logic is connected, and what to validate so training, evaluation, and dataset caching remain consistent.
---

# LLM Offline Env Support

## Overview

Use this skill when the user wants to add a new PointMaze variant, add a new environment family, or refactor how prompt metadata, formatting, and action decoding are connected.

This repo uses:
- family-level shared prompt templates in `prompts/<env_family>/<idx>.txt`
- variant metadata in `data/<env_family>/variants.py`
- family formatter/decoder functions in `data/<env_family>/formatting.py`
- family dataset construction in `data/<env_family>/dataset.py`
- family registration in `data/registry.py`

## Decision Rule

### Add only a new variant

Do this when the observation schema, action schema, parse/validate rules, and dataset construction logic are the same as an existing environment family.

Typical example:
- A new PointMaze map or reward variant that still uses the same obs format, action format, and decoder rules.

### Add a new environment family

Do this when any of these change:
- observation fields or how they should be serialized
- action target text format
- action parsing or validation logic
- dataset loading procedure
- prompt variable schema needed by shared templates

## Workflow: New Variant

When adding a new variant inside an existing family:

1. Update `data/<env_family>/variants.py`.
Add a new entry with exactly the family's expected top-level shape.
For PointMaze today that is:
- `dataset_id`
- `env_id`
- `prompt_vars`

2. Fill `prompt_vars` with all fields referenced by the family's shared prompt templates.
For PointMaze today that includes at least:
- `env_name`
- `reward_type`
- `maze_map`
- `reward_desc_en`
- `reward_desc_zh`
- `maze_shape`
- `maze_raw_matrix`
- `maze_visual`
- `structure_desc_en`
- `structure_desc_zh`

3. Do not add a new prompt file for the variant unless the user explicitly wants a prompt-system change.
The current design is that variants provide metadata and all variants in the family reuse `prompts/<env_family>/<idx>.txt`.

4. Verify the shared templates can render with the new `prompt_vars`.
Missing variables must fail loudly. Extra variables are allowed.

5. If the user wants the new variant available in single-variant training and in `variant=all` evaluation paths, make sure every place iterating the variant dict naturally includes it.

## Workflow: New Environment Family

When adding a new environment family:

1. Create `data/<env_family>/variants.py`.
Define the family's variant metadata dictionary. Keep per-variant facts here, especially prompt rendering metadata.

2. Create `data/<env_family>/formatting.py`.
Implement all four functions:
- `format_obs`
- `format_action`
- `parse_action`
- `validate_action`

These are the family boundary. If these are incomplete, training or evaluation is not done.

3. Create `data/<env_family>/dataset.py`.
The dataset must:
- load the offline data for the family
- split train/val at episode level when applicable
- load shared family templates from `prompts/<env_family>/<idx>.txt`
- render prompts with `render_template(template, prompt_vars, obs_text=...)`
- respect `prompt_template_count`
- use cache filenames that include `prompts<N>`

4. Create `prompts/<env_family>/0.txt`, `1.txt`, and any additional indexed templates.
Requirements:
- filenames must be numeric and contiguous from `0`
- template `0` is the evaluation template unless the code is intentionally changed
- templates should be style variants, not variant-specific copies

5. Register the family in `data/registry.py`.
Add:
- dataset class
- formatter module

6. Check `train.py` and `evaluate.py` only if the new family needs special-case behavior.
Prefer not to add special cases. Follow the existing family abstraction.

## Prompt Rules

The current prompt system is strict:
- shared prompt files live at `prompts/<env_family>/<idx>.txt`
- `utils/prompt_loader.py` loads templates by numeric index
- `render_template(...)` raises on missing variables
- evaluation uses template `0`
- training uses the first `prompt_template_count` templates

When updating prompts, avoid these mistakes:
- do not reintroduce per-variant duplicated prompt files
- do not hardcode the number of templates in code unless the user explicitly wants that
- do not move variant-only facts out of `variants.py`
- do not let templates silently render missing fields as empty strings

## Validation Checklist

After adding a variant or family, validate at minimum:
- `python -m py_compile` passes for the touched Python files
- shared templates load successfully
- one representative variant can render template `0` with `render_template(...)`
- `prompt_template_count=1` works
- `prompt_template_count` greater than available templates fails clearly
- training and evaluation still import the family without special-case edits unless intentionally required
- dataset cache naming still includes `prompts<N>`

## Project-Specific Notes

For this repo today:
- PointMaze uses family-shared templates plus per-variant `prompt_vars`
- PointMaze top-level variant entries should stay minimal
- if the user asks to simplify metadata, prefer moving derived prompt facts into `prompt_vars` rather than adding more top-level fields
- if prompt duplication appears, the preferred fix is to push shared wording into family templates and keep only environment-specific facts in `variants.py`
