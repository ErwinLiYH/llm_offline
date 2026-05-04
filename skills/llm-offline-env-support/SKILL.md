---
name: llm-offline-env-support
description: Use this skill when adding or updating environment-family support or variant support in the llm_offline project. It covers when to add a new variant versus a new env family, which files must change, how prompts are wired through shared family templates plus variant prompt_vars, how formatter/decoder logic is connected, and what to validate so training, evaluation, and dataset caching remain consistent.
---

# LLM Offline Env Support

## Overview

Use this skill when the user wants to add a new PointMaze variant, add a new environment family, or refactor how prompt metadata, formatting, and action decoding are connected.

This repo uses:
- family-level shared prompt templates in `prompts/<env_family>/<prompt_name>.txt`
- variant metadata in `data/<env_family>/variants.py`
- family text-mode formatter/decoder functions in `data/<env_family>/formatting.py`
- family dataset construction in `data/<env_family>/dataset.py`
- shared action-bin token/display/model-id logic in `utils/action_bins.py`
- family registration in `data/registry.py`

## Decision Rule

### Add only a new variant

Do this when the observation schema, action schema, parse/validate rules, and dataset construction logic are the same as an existing environment family.

Typical example:
- A new PointMaze map or reward variant that still uses the same obs format, action format, and decoder rules.

### Add a new environment family

Do this when any of these change:
- observation fields or how they should be serialized
- text-mode action target format
- text-mode action parsing or action validation logic
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
The current design is that variants provide metadata and all variants in the family reuse `prompts/<env_family>/<prompt_name>.txt`.

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

These are the family boundary for observation rendering, text-mode action format, and final action validation. If these are incomplete, training or evaluation is not done.

Do not add `format_action_bin_tokens` or `parse_action_bin_tokens` to new family formatters. Bin and gaussian-bin modes are shared:
- action discretization, display text such as `<act_00>`, model token ids/text, mapping hashes, and token-id decoding live in `utils.action_bins.ActionBinCodec`
- `new_token: false` reuses stable low-frequency tokenizer IDs internally and only displays `<act_XX>` in human-readable logs/cache JSONL
- `new_token: true` preserves the older path that registers `<act_XX>` as additional special tokens
- eval bin parsing uses generated token ids through the codec, not formatter text parsing
- the environment formatter still supplies `validate_action(action)` for both text and bin modes

3. Create `data/<env_family>/dataset.py`.
The dataset must:
- load the offline data for the family
- split train/val at episode level when applicable
- load shared family templates from `prompts/<env_family>/<prompt_name>.txt`
- render prompts with `render_template(template, prompt_vars, obs_text=...)`
- respect `prompt_templete_index` prompt-name selection
- for text mode, use the family formatter's `format_action(action)` as the assistant target
- for bin modes, use `ActionBinCodec` to compute bin indices, model action text, display action text, and model token ids
- write display action text such as `<act_24><act_37>` to human-readable JSONL/history, while tokenized PKL samples contain the real model token ids
- fill `action_bin_labels` only at action-token positions with the true bin index; all non-action and padding positions must be `-1`
- use cache filenames that include the selected prompt-name tag, action mode/range/bin count, `newtok<0|1>`, and the action-token schema hash
- store cache metadata for `new_token` and `action_token_schema_hash`, and fail loudly on schema mismatch rather than silently reusing stale tokenized samples

4. Create `prompts/<env_family>/<prompt_name>.txt` templates.
Requirements:
- filenames define prompt names; they do not need to be numeric
- evaluation uses the first template in filename order unless the code is intentionally changed
- templates should be style variants, not variant-specific copies

5. Register the family in `data/registry.py`.
Add:
- dataset class
- formatter module

6. Check `train.py` and `evaluate.py` only if the new family needs special-case behavior.
Prefer not to add special cases. Follow the existing family abstraction.

## Dataset Implementation Guide

New environment families should implement the `BaseOfflineDataset` interface in
`data/base_dataset.py`. There are two implementation levels:

- **Basic implementation**: enough for `train.py` to construct train/val loaders.
- **Advanced implementation**: optional PointMaze-style batching, cache reuse, multiprocessing, and file progress.

### Basic Dataset Implementation

Create `data/<env_family>/dataset.py` with a dataset class registered in
`data/registry.py`. The class must inherit `BaseOfflineDataset` and implement:

```python
@classmethod
def build_batch(cls, requests: list[DatasetBuildRequest]) -> list[BaseOfflineDataset]:
    ...

@classmethod
def collect_variant_episode_stats(
    cls,
    variant: str,
    episode_keep_num: int | None,
) -> VariantEpisodeStats:
    ...

def __len__(self) -> int:
    ...

def __getitem__(self, idx: int) -> TensorSample:
    ...
```

`train.py` calls only `dataset_cls.build_batch(dataset_requests)`. It expects:

- Return length equals `len(requests)`.
- Return item `i` corresponds exactly to request `i`.
- Each returned object is a loaded PyTorch `Dataset`.
- The class exposes `collate_fn`; using the inherited `BaseOfflineDataset.collate_fn` is usually enough.

`DatasetBuildRequest` is the full input contract from training. The important fields are:

- `variant`, `split`: requested dataset identity; `split` is normally `"train"` or `"val"`.
- `tokenizer`, `tokenizer_name_or_path`, `max_length`: tokenizer context for building model-ready samples.
- `num_workers`, `cache_dir`, `max_data_num`: offline construction controls.
- `prompt_templete_index` / `prompt_template_count`: prompt template selection.
- `train_data_ratio`, `episode_keep_num`, `sampling_seed`, `balance_variant_episode_count`, `balanced_train_episode_count`: episode sampling/split controls.
- `history_num`, `history_stride`: optional history prompt controls.
- `action_token_mode`, `action_num_bins`, `action_bin_min`, `action_bin_max`, `new_token`: action target encoding controls.

For a minimal `build_batch`:

1. Iterate through `requests` in order.
2. Load offline data for `request.variant`.
3. Split at episode level when the family has episode structure:
   - select up to `episode_keep_num` episodes, or all if `None`
   - use `sampling_seed` for reproducible random selection
   - use `floor(pool_size * train_data_ratio)` train episodes and the rest as val
4. Load templates from `prompts/<env_family>/<prompt_name>.txt` using `prompt_templete_index`.
5. For each selected timestep and each selected template:
   - call the family formatter's `format_obs(...)`
   - render the prompt with `render_template(...)`
   - in text mode, format the action target with the family formatter's `format_action(...)`
   - in bin modes, use `ActionBinCodec` for action bin indices, model action text, display text, and target token ids
   - wrap prompt/action using the tokenizer chat template, matching the existing training format
   - tokenize to `input_ids`, `attention_mask`, `labels`
   - fill `action_bin_labels` at the exact action token positions in bin modes
6. Return one dataset object per request, preserving request order.

Each `__getitem__` result must be:

```python
{
    "input_ids": torch.LongTensor[seq_len],
    "attention_mask": torch.LongTensor[seq_len],
    "labels": torch.LongTensor[seq_len],
    "action_bin_labels": torch.LongTensor[seq_len],
}
```

Padding and masking conventions:

- `input_ids`: padded with `0` by `collate_fn`
- `attention_mask`: padded with `0`
- `labels`: prompt/user positions are `-100`; padding is also `-100`
- `action_bin_labels`: action-token positions contain the true bin index; all other positions and padding are `-1`

For text-only action families, still return `action_bin_labels` as all `-1` so the shared loss path works.

`collect_variant_episode_stats(...)` is used before multi-variant training balance. It must return:

```python
{
    "variant": variant,
    "total_episodes": int,
    "total_steps": int,
    "initial_train_target": int,
    "sampled_episode_target": int,
}
```

For current semantics, `sampled_episode_target` should be
`total_episodes` when `episode_keep_num is None`, otherwise
`min(total_episodes, episode_keep_num)`. `initial_train_target` is kept for compatibility and should match the same target unless a family has a documented reason to differ.

### Advanced Dataset Features Used By PointMaze

PointMaze implements more than the minimum. New families may copy this pattern when offline tokenization is expensive or multi-variant training is common:

- **Single batch construction path**: `build_batch()` receives all selected variants and train/val requests at once, groups requests by variant, and builds every dataset in one call.
- **Shared process pool**: cache misses are tokenized with one `ProcessPoolExecutor`; worker payloads are episode-level, and each worker initializes its tokenizer once.
- **Multi-plan worker config**: each episode payload carries a `job_id`; workers use `job_id -> config` to select variant prompt vars, templates, history settings, and action encoding.
- **Joint file progress**: `MultiWorkerFileProgress` writes one total progress file plus per-worker sub-progress rows. `train.py` prints only the joint progress path.
- **Episode-level cache**: cache stores `episode_idx -> tokenized samples`, not split-level flattened samples.
- **Cache hit resampling**: `episode_keep_num`, `train_data_ratio`, `sampling_seed`, and variant balancing are reapplied after cache load. If the cache lacks any current sampled episode, rebuild and overwrite that variant cache.
- **Cache signature discipline**: include only tokenization-changing settings in cache filenames, such as data signature, tokenizer/model tag, `max_length`, selected prompt names, history settings, action encoding, `newtok<0|1>`, and the action-token schema hash. Do not include runtime sampling settings or `max_data_num`.
- **Debug truncation**: apply `max_data_num` only after final train/val samples are assembled; never truncate what is written to cache.

Important implementation notes for advanced mode:

- Use `spawn` multiprocessing unless there is a specific reason not to; this avoids unsafe fork state around CUDA/PyTorch/tokenizers.
- Require all pending jobs in the same process pool to share tokenizer and action-token schema. Raise a clear `ValueError` if they differ.
- Preserve request order exactly even when grouping by variant internally.
- Preserve episode boundaries in cache; a flattened cache cannot safely reapply later `episode_keep_num` or `train_data_ratio` changes.
- Keep cache files per variant and per tokenization signature. Do not merge variants into one cache file.
- Cache hit should not enter the process pool or count toward tokenization progress.
- If the family supports `bin` / `gaussian_bin`, initialize `ActionBinCodec` inside worker tokenizers with `get_action_bin_codec(..., ensure_registered=True)`. This registers special action tokens only when `new_token: true`; with `new_token: false` it selects reused tokenizer ids without growing the vocabulary. Fill `action_bin_labels` only at generated action-token positions.
- `.jsonl` debug cache should include enough provenance to inspect records, typically `episode_idx`, `timestep`, `prompt`, and `action`.
- `.jsonl` debug cache and history text should use display actions such as `<act_24><act_37>`; tokenized PKL samples should contain the true model token ids.

## Prompt Rules

The current prompt system is strict:
- shared prompt files live at `prompts/<env_family>/<prompt_name>.txt`
- `utils/prompt_loader.py` loads templates by filename stem
- `render_template(...)` raises on missing variables
- evaluation uses the first template in filename order
- training uses the templates named by `prompt_templete_index`

When updating prompts, avoid these mistakes:
- do not reintroduce per-variant duplicated prompt files
- do not hardcode the number of templates in code unless the user explicitly wants that
- do not move variant-only facts out of `variants.py`
- do not let templates silently render missing fields as empty strings

## Validation Checklist

After adding a variant or family, validate at minimum:
- `python -m py_compile` passes for the touched Python files
- shared templates load successfully
- one representative variant can render a selected prompt template with `render_template(...)`
- `prompt_templete_index: ["0"]` works for the current PointMaze prompts
- unknown prompt names fail clearly
- training and evaluation still import the family without special-case edits unless intentionally required
- dataset cache naming still includes the selected prompt-name tag
- text-mode eval uses the family formatter's `parse_action(...)`
- bin-mode eval uses generated token ids and `ActionBinCodec`, not formatter text parsing
- bin-mode dataset tokenization writes the correct number of non-negative `action_bin_labels` for the environment action dimension
- dataset cache naming includes `newtok<0|1>` and the action-token schema hash
- human-readable dataset JSONL and eval step logs display `<act_XX>` bins even when `new_token: false`
- `new_token: false` does not add special tokens, resize embeddings, or force `embed_tokens` / `lm_head` into LoRA target modules
- `new_token: true` still registers action special tokens and keeps the embedding/LoRA handling path valid

## Project-Specific Notes

For this repo today:
- PointMaze uses family-shared templates plus per-variant `prompt_vars`
- PointMaze top-level variant entries should stay minimal
- if the user asks to simplify metadata, prefer moving derived prompt facts into `prompt_vars` rather than adding more top-level fields
- if prompt duplication appears, the preferred fix is to push shared wording into family templates and keep only environment-specific facts in `variants.py`
