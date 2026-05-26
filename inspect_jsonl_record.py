"""Print one JSONL record as Prompt/Action for manual inspection.

Usage:
    micromamba run -n llm_offline python inspect_jsonl_record.py path/to/file.jsonl 0
"""

from __future__ import annotations

import argparse
import json

from utils.record_format import format_prompt_action_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path", type=str)
    parser.add_argument("record_index", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.record_index < 0:
        raise ValueError(f"record_index must be >= 0, got {args.record_index}")

    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx != args.record_index:
                continue
            record = json.loads(line)
            prompt = record.get("prompt", "")
            action = record.get("action", "")
            text = format_prompt_action_text(prompt, action)
            pht_text = record.get("place holder")
            if pht_text:
                text = f"{text}\n\nPlace Holder:\n{pht_text}"
            print(text)
            return

    raise IndexError(
        f"record_index {args.record_index} is out of range for file: {args.jsonl_path}"
    )


if __name__ == "__main__":
    main()
