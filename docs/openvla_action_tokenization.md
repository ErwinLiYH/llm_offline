# OpenVLA Action Tokenization Notes

This note records the OpenVLA action-token implementation details that are
relevant to this project. It is a reference note, not a statement that this repo
must copy OpenVLA exactly.

## High-Level Design

OpenVLA represents continuous robot actions as ordinary language-model token
IDs. Its original action tokenizer does not add human-readable tokens such as
`<act_00>`. Instead, it reuses the least-used token IDs at the end of the base
LLaMA tokenizer vocabulary as action bins.

The important consequence is that OpenVLA avoids the usual "new token" workflow:

- no `tokenizer.add_special_tokens(...)`
- no tokenizer vocabulary growth
- no `resize_token_embeddings(...)` for action bins
- no separate saving/loading problem for newly added action-token embeddings

This repo now supports both paths. With `new_token: false` it follows the same
high-level idea as OpenVLA by reusing existing tokenizer IDs internally. With
`new_token: true` it preserves the older path where `<act_00> ... <act_N>` are
newly registered tokenizer tokens and therefore need matching model
embedding/output-head handling.

## Action Normalization

OpenVLA first normalizes robot actions into a common range before tokenization.
For each dataset/action dimension, the codebase stores action statistics used
for normalization and later unnormalization. The paper describes using the 1st
and 99th action quantiles as the effective discretization range, instead of raw
min/max, so outlier actions do not stretch the bins too far.

In code, the RLDS data path uses `NormalizationType.BOUNDS_Q99`, which maps
action values into `[-1, 1]` using the dataset statistics.

## Uniform Binning

The original `ActionTokenizer` uses uniform bins over `[-1, 1]`.

Conceptually:

```python
bins = linspace(-1, 1, n_bins)
bin_centers = midpoint(bins[:-1], bins[1:])
discrete = digitize(clip(action, -1, 1), bins)
token_id = tokenizer.vocab_size - discrete
```

For the default `n_bins=256`, each continuous action dimension becomes one
token ID from the last 256-ish IDs of the tokenizer vocabulary. Each action
vector is represented as a sequence with one token per action dimension.

One detail to keep in mind: with `np.linspace(-1, 1, 256)`, there are 256 bin
boundary values but 255 adjacent intervals/centers. The decoder clips the
derived center index into the valid center range, so edge cases map to the first
or last center.

## Token ID Mapping

OpenVLA defines the beginning of the action-token region near the end of the
base vocabulary:

```python
action_token_begin_idx = tokenizer.vocab_size - (n_bins + 1)
```

For a discretized bin index `d`, it maps to:

```python
token_id = tokenizer.vocab_size - d
```

The paper explains the reason for this design: the LLaMA tokenizer did not have
enough reserved special tokens for 256 action bins, so OpenVLA reused the last
256 least-used vocabulary tokens as action tokens.

This is not the same as adding `additional_special_tokens`. The token IDs
already exist in the base model; OpenVLA changes their effective meaning during
VLA training.

## Dataset Labels And Loss

OpenVLA's RLDS batch transform builds a prompt with a human turn and a GPT turn.
The GPT turn contains the action-token string produced by `ActionTokenizer`.

The labels are initially copied from `input_ids`, then the prompt prefix is
masked out with `-100`. Only action tokens, plus optionally the stop token, are
trained.

Conceptually:

```python
input_ids = tokenizer(rendered_prompt_with_action).input_ids
labels = input_ids.copy()
labels[prompt_prefix_positions] = -100
```

The training loop then calls the model with `labels=batch["labels"]` and uses
the model's standard causal language-model loss:

```python
loss = outputs.loss
```

So OpenVLA's vanilla action-bin training objective is ordinary full-vocabulary
next-token cross entropy on action-token positions. It does not use a Gaussian
soft-label action-bin loss.

## Training Metrics

During LoRA fine-tuning, OpenVLA also logs action-specific metrics:

- `action_accuracy`: argmax token equals the target action token at action-token
  label positions.
- `l1_loss`: predicted and target action token IDs are decoded back into
  continuous normalized actions, then compared with L1 loss.

The action-position mask is based on token ID range near the end of the
vocabulary, not on a separate `action_bin_labels` tensor.

## Inference

At inference, OpenVLA generates exactly `action_dim` new tokens. It takes the
last `action_dim` generated token IDs and decodes them back into normalized
continuous actions:

```python
discrete = vocab_size - predicted_token_ids
center_idx = clip(discrete - 1, 0, len(bin_centers) - 1)
normalized_action = bin_centers[center_idx]
```

Then it unnormalizes the normalized actions using the saved dataset action
statistics. Dimensions marked by the dataset mask are unnormalized with
quantile bounds; unmasked dimensions can remain in normalized form.

## Difference From This Repo

OpenVLA:

- reuses existing tokenizer IDs at the end of the vocabulary
- does not grow the tokenizer vocabulary for action bins
- avoids `embed_tokens` / `lm_head` resize and new-token save/load issues
- trains with standard full-vocabulary causal LM cross entropy
- decodes action tokens using `vocab_size - token_id`
- relies on dataset action statistics for normalization and unnormalization

This repo's `new_token: false` action-bin path:

- selects stable, non-special tokenizer IDs by scanning backward from
  `tokenizer.vocab_size - 1`
- validates that selected token IDs roundtrip through `decode(ids)` and
  tokenization before using them
- feeds the model the real selected token IDs/text during train and eval
- keeps all human-readable artifacts as display tokens such as
  `<act_00><act_37>`
- stores the tokenized dataset cache under a key that includes `new_token` and
  the action-token mapping hash
- parses generated bin actions from generated token IDs first, not from decoded
  low-frequency token text
- does not add special tokens, resize embeddings, or automatically train
  `embed_tokens` / `lm_head`

This repo's `new_token: true` action-bin path:

- adds explicit readable tokens such as `<act_00>`
- requires tokenizer mutation and model embedding/output-head resize
- must train and save the action-token embedding/output rows correctly
- supports a custom Gaussian soft-label loss for `gaussian_bin`
- decodes bins by looking up explicit action token IDs

The OpenVLA design is attractive because it avoids new-token mechanics. The
tradeoff is that it repurposes existing rare language tokens. This repo hides
that internal representation behind a display mapping so jsonl records, history
prompts, step logs, and probability logs still show readable `<act_XX>` bins.

## References

- OpenVLA paper, section 3.2 "OpenVLA Training Procedure":
  https://arxiv.org/html/2406.09246
- OpenVLA `ActionTokenizer` implementation:
  https://github.com/openvla/openvla/blob/main/prismatic/vla/action_tokenizer.py
- OpenVLA RLDS dataset transform and action-label masking:
  https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/datasets.py
- OpenVLA LoRA fine-tuning script:
  https://github.com/openvla/openvla/blob/main/vla-scripts/finetune.py
- OpenVLA runtime action prediction wrapper:
  https://github.com/openvla/openvla/blob/main/prismatic/models/vlas/openvla.py
- HuggingFace-style OpenVLA action prediction model:
  https://github.com/openvla/openvla/blob/main/prismatic/extern/hf/modeling_prismatic.py
