#!/usr/bin/env python
"""Smoke test PEFT LoRA training with newly added special tokens.

This mirrors the project action-bin token flow at a small scale:
- load tokenizer
- add special tokens after loading
- resize model token embeddings with len(tokenizer)
- train LoRA with embed_tokens/lm_head included
- save adapter + tokenizer
- reload adapter and verify new-token loss is preserved

The text samples come from the PEFT official notebook dataset
`smangrul/assistant_chatbot_dataset`, but the default model is a tiny random
Llama so this stays cheap.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
from enum import Enum
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator


class OfficialSpecialTokens(str, Enum):
    begin_target = "<|begintarget|>"
    end_target = "<|endtarget|>"
    begin_context = "<|begincontext|>"
    end_context = "<|endcontext|>"
    system = "<|system|>"
    user = "<|user|>"
    begin_last_user_utterance = "<|beginlastuserutterance|>"
    end_last_user_utterance = "<|endlastuserutterance|>"
    begin_dsts = "<|begindsts|>"
    end_dsts = "<|enddsts|>"
    begin_dst = "<|begindst|>"
    end_dst = "<|enddst|>"
    begin_belief = "<|beginbelief|>"
    end_belief = "<|endbelief|>"
    begin_response = "<|beginresponse|>"
    end_response = "<|endresponse|>"
    begin_action = "<|beginaction|>"
    end_action = "<|endaction|>"
    begin_user_action = "<|beginuseraction|>"
    end_user_action = "<|enduseraction|>"
    sys_actions = "<|sysactions|>"
    begin_intent = "<|beginintent|>"
    end_intent = "<|endintent|>"
    begin_requested_slots = "<|beginrequestedslots|>"
    end_requested_slots = "<|endrequestedslots|>"
    pad_token = "<|pad|>"
    bos_token = "<|startoftext|>"

    @classmethod
    def list(cls) -> list[str]:
        return [item.value for item in cls]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default="hf-internal-testing/tiny-random-LlamaForCausalLM",
        help="Small causal LM with embed_tokens/lm_head/q_proj/v_proj modules.",
    )
    parser.add_argument("--dataset-slice", default="train[:16]")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        default="/tmp/llm_offline_added_token_lora_smoke",
        help="Temporary adapter/tokenizer roundtrip directory.",
    )
    parser.add_argument("--keep-output", action="store_true")
    return parser.parse_args()


def add_official_tokens(tokenizer) -> int:
    old_len = len(tokenizer)
    tokenizer.add_special_tokens(
        {
            "pad_token": OfficialSpecialTokens.pad_token.value,
            "bos_token": OfficialSpecialTokens.bos_token.value,
            "eos_token": OfficialSpecialTokens.end_target.value,
            "additional_special_tokens": OfficialSpecialTokens.list(),
        }
    )
    return len(tokenizer) - old_len


def build_dataset(tokenizer, dataset_slice: str, max_length: int):
    dataset = load_dataset("smangrul/assistant_chatbot_dataset", split=dataset_slice)

    def preprocess(examples):
        contexts = tokenizer(examples["context"], add_special_tokens=False)
        targets = tokenizer(examples["target"], add_special_tokens=False)
        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []
        for context_ids, target_ids in zip(contexts["input_ids"], targets["input_ids"]):
            target_ids = target_ids + [tokenizer.eos_token_id]
            input_ids = context_ids + target_ids
            labels = [-100] * len(context_ids) + target_ids
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]
            attention_mask = [1] * len(input_ids)
            pad_len = max_length - len(input_ids)
            input_ids_batch.append([tokenizer.pad_token_id] * pad_len + input_ids)
            attention_mask_batch.append([0] * pad_len + attention_mask)
            labels_batch.append([-100] * pad_len + labels)
        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels_batch,
        }

    return dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )


def special_token_ids(tokenizer) -> set[int]:
    ids = {
        int(tokenizer.convert_tokens_to_ids(token))
        for token in OfficialSpecialTokens.list()
    }
    missing = [token for token in OfficialSpecialTokens.list() if tokenizer.convert_tokens_to_ids(token) is None]
    if missing:
        raise ValueError(f"Missing special tokens: {missing}")
    return ids


@torch.no_grad()
def special_token_metrics(model, dataloader, special_ids: set[int], device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits[:, :-1, :].float()
        labels = batch["labels"][:, 1:]
        mask = labels != -100
        if mask.any():
            special_mask = torch.zeros_like(mask)
            for token_id in special_ids:
                special_mask |= labels == token_id
            mask &= special_mask
        if not mask.any():
            continue
        selected_logits = logits[mask]
        selected_labels = labels[mask]
        total_loss += F.cross_entropy(selected_logits, selected_labels, reduction="sum").item()
        total_correct += (selected_logits.argmax(dim=-1) == selected_labels).sum().item()
        total_count += selected_labels.numel()
    if total_count == 0:
        raise RuntimeError("No supervised special-token labels found in the dataset slice.")
    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
        "count": float(total_count),
    }


def train(model, dataloader, *, steps: int, learning_rate: float, device: torch.device):
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    iterator = iter(dataloader)
    last_loss = math.nan
    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if step == 1 or step % max(steps // 4, 1) == 0 or step == steps:
            print(f"[train] step={step}/{steps} loss={last_loss:.4f}")
    return last_loss


def main():
    args = parse_args()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    added = add_official_tokens(tokenizer)
    print(f"[tokenizer] added={added}, len={len(tokenizer)}, pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id}")
    for token in OfficialSpecialTokens.list()[:6]:
        ids = tokenizer(token, add_special_tokens=False).input_ids
        print(f"[tokenizer] {token} -> {ids}")
        if len(ids) != 1:
            raise AssertionError(f"Token {token!r} is not a single token after registration: {ids}")

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=["embed_tokens", "lm_head", "q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.to(device)
    model.print_trainable_parameters()

    dataset = build_dataset(tokenizer, args.dataset_slice, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=default_data_collator)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=default_data_collator)
    ids = special_token_ids(tokenizer)

    before = special_token_metrics(model, eval_loader, ids, device)
    print(f"[before] special_loss={before['loss']:.4f} special_acc={before['accuracy']:.2%} count={int(before['count'])}")
    train(model, dataloader, steps=args.steps, learning_rate=args.learning_rate, device=device)
    after = special_token_metrics(model, eval_loader, ids, device)
    print(f"[after] special_loss={after['loss']:.4f} special_acc={after['accuracy']:.2%} count={int(after['count'])}")
    if after["loss"] >= before["loss"]:
        raise AssertionError(f"Special-token loss did not improve: before={before}, after={after}")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    reloaded_tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    reloaded_base = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    reloaded_base.resize_token_embeddings(len(reloaded_tokenizer))
    reloaded = PeftModel.from_pretrained(reloaded_base, output_dir)
    reloaded.to(device)
    loaded = special_token_metrics(reloaded, eval_loader, ids, device)
    print(f"[loaded] special_loss={loaded['loss']:.4f} special_acc={loaded['accuracy']:.2%} count={int(loaded['count'])}")
    if abs(loaded["loss"] - after["loss"]) > 1e-4:
        raise AssertionError(f"Reloaded loss drifted: after={after}, loaded={loaded}")

    print("[ok] added-token LoRA train/save/load smoke passed")
    if not args.keep_output:
        shutil.rmtree(output_dir)


if __name__ == "__main__":
    main()
