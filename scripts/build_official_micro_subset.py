#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


TAG_RE = {
    "summary": re.compile(r"<SUMMARY>\s*(.*?)\s*</SUMMARY>", re.S),
    "caption": re.compile(r"<CAPTION>\s*(.*?)\s*</CAPTION>", re.S),
    "reasoning": re.compile(r"<REASONING>\s*(.*?)\s*</REASONING>", re.S),
    "conclusion": re.compile(r"<CONCLUSION>\s*(.*?)\s*</CONCLUSION>", re.S),
}


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_sections(text: str) -> dict[str, str] | None:
    out: dict[str, str] = {}
    for name, pattern in TAG_RE.items():
        match = pattern.search(text)
        if not match:
            return None
        out[name] = clean(match.group(1))
    return out


def iter_turns(path: Path, prefixes: set[str]):
    with path.open(encoding="utf-8") as handle:
        for record_idx, line in enumerate(handle):
            obj = json.loads(line)
            image = obj.get("image", "")
            task = image.split("/", 1)[0] if image else ""
            if task not in prefixes:
                continue
            conversations = obj.get("conversations", [])
            for turn_idx in range(0, len(conversations) - 1, 2):
                human = conversations[turn_idx]
                assistant = conversations[turn_idx + 1]
                if human.get("from") != "human" or assistant.get("from") != "gpt":
                    continue
                sections = extract_sections(assistant.get("value", ""))
                if not sections:
                    continue
                yield {
                    "id": f"{obj.get('id', record_idx)}::turn{turn_idx//2}",
                    "source_record_id": obj.get("id"),
                    "source_record_index": record_idx,
                    "turn_index": turn_idx // 2,
                    "task": task,
                    "image": image,
                    "question": clean(human.get("value", "")),
                    "summary": sections["summary"],
                    "caption": sections["caption"],
                    "reasoning": sections["reasoning"],
                    "answer": sections["conclusion"],
                    "whole_cot": clean(
                        sections["summary"]
                        + "\n"
                        + sections["caption"]
                        + "\n"
                        + sections["reasoning"]
                    ),
                }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/data/zxl/official_heima/datasets/LLaVA-CoT-100k/train.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1"))
    parser.add_argument("--tasks", nargs="+", default=["chartqa", "sqa"])
    parser.add_argument("--train-per-task", type=int, default=96)
    parser.add_argument("--val-per-task", type=int, default=24)
    parser.add_argument("--test-per-task", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in iter_turns(args.input, set(args.tasks)):
        by_task[row["task"]].append(row)

    splits = {"train": [], "validation": [], "test": []}
    required = args.train_per_task + args.val_per_task + args.test_per_task
    for task in args.tasks:
        rows = by_task.get(task, [])
        if len(rows) < required:
            raise SystemExit(f"Task {task} has only {len(rows)} usable turns; need {required}.")
        rng.shuffle(rows)
        train_end = args.train_per_task
        val_end = train_end + args.val_per_task
        splits["train"].extend(rows[:train_end])
        splits["validation"].extend(rows[train_end:val_end])
        splits["test"].extend(rows[val_end:required])

    for rows in splits.values():
        rng.shuffle(rows)

    for name, rows in splits.items():
        write_jsonl(args.out / f"{name}.jsonl", rows)

    image_rows = []
    seen_images = set()
    for name, rows in splits.items():
        for row in rows:
            if row["image"] in seen_images:
                continue
            seen_images.add(row["image"])
            image_rows.append({"image": row["image"], "task": row["task"], "required_by_split": name})
    write_jsonl(args.out / "image_manifest.jsonl", image_rows)

    counts = {name: Counter(row["task"] for row in rows) for name, rows in splits.items()}
    spec = {
        "name": args.out.name,
        "source": str(args.input),
        "seed": args.seed,
        "tasks": args.tasks,
        "splits": {name: {"num_turns": len(rows), "tasks": dict(counts[name])} for name, rows in splits.items()},
        "num_unique_images": len(seen_images),
        "fields": [
            "id",
            "task",
            "image",
            "question",
            "summary",
            "caption",
            "reasoning",
            "answer",
            "whole_cot",
        ],
        "heima_alignment": {
            "uses_official_llava_cot_json_schema": True,
            "preserves_official_section_targets": ["summary", "caption", "reasoning"],
            "preserves_official_image_path": True,
            "requires_images_for_true_mllm_encoder": True,
            "can_run_decoder_loss_framework_without_images": True,
        },
    }
    (args.out / "dataset_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(spec, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
