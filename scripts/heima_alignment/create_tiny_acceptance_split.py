#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.heima_alignment.data.image_resolver import ImageResolver

REQUIRED = ("image", "question", "summary", "caption", "reasoning", "answer")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_index"] = idx
            rows.append(row)
    return rows


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def validate_row(row: dict, split: str) -> None:
    missing = [k for k in REQUIRED if row.get(k) in (None, "")]
    if missing:
        raise ValueError(f"{split} row {row.get('_source_index')} missing {missing}")


def split_item(row: dict, split: str, resolver: ImageResolver) -> dict:
    validate_row(row, split)
    resolved = resolver.resolve(str(row["image"]))
    return {
        "split": split,
        "index": int(row["_source_index"]),
        "id": str(row.get("id", row["_source_index"])),
        "image_field": str(row["image"]),
        "resolved_image_path": resolved.resolved_path,
        "image_path_hash": resolved.sha256_16,
        "question_hash": sha16(str(row["question"])),
        "reasoning_hash": sha16(str(row["reasoning"])),
        "answer_hash": sha16(str(row["answer"])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-root", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1"))
    ap.add_argument("--out", type=Path, default=Path("/data/zxl/runs/heima_ab_loss1_tiny_acceptance_v1/data_split.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-size", type=int, default=192)
    ap.add_argument("--eval-size", type=int, default=48)
    ap.add_argument("--allow-smaller-available", action="store_true", help="write an available-image split when exact 192/48 is impossible")
    args = ap.parse_args()

    if "LLaVA-CoT-100k/datasets" in str(args.subset_root):
        raise ValueError("tiny acceptance must not use full LLaVA-CoT-100k train.jsonl")
    train_path = args.subset_root / "train.jsonl"
    eval_path = args.subset_root / "validation.jsonl"
    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(f"expected train/validation JSONL under {args.subset_root}")

    train_rows = read_jsonl(train_path)
    eval_rows = read_jsonl(eval_path)
    if len(train_rows) < args.train_size or len(eval_rows) < args.eval_size:
        raise ValueError(f"subset too small: train={len(train_rows)}, eval={len(eval_rows)}")

    resolver = ImageResolver(args.subset_root)

    def available_indices(rows: list[dict], split: str) -> tuple[list[int], list[dict]]:
        ok: list[int] = []
        missing: list[dict] = []
        for idx, row in enumerate(rows):
            validate_row(row, split)
            try:
                resolver.resolve(str(row["image"]))
                ok.append(idx)
            except FileNotFoundError as exc:
                missing.append({"index": int(row["_source_index"]), "id": str(row.get("id", row["_source_index"])), "image": str(row.get("image", "")), "error": str(exc)})
        return ok, missing

    train_available, train_missing = available_indices(train_rows, "train")
    eval_available, eval_missing = available_indices(eval_rows, "eval")
    exact_possible = len(train_available) >= args.train_size and len(eval_available) >= args.eval_size
    if not exact_possible and not args.allow_smaller_available:
        report = {
            "status": "stop_gate_failed",
            "reason": "requested exact tiny acceptance split cannot be built from locally accessible images",
            "requested_train": args.train_size,
            "requested_eval": args.eval_size,
            "available_train": len(train_available),
            "available_eval": len(eval_available),
            "missing_train_count": len(train_missing),
            "missing_eval_count": len(eval_missing),
            "missing_train_examples": train_missing[:20],
            "missing_eval_examples": eval_missing[:20],
            "no_download_performed": True,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        stop_path = args.out.with_name("data_split_stop_gate_missing_images.json")
        stop_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))

    rng = random.Random(args.seed)
    train_order = list(train_available)
    eval_order = list(eval_available)
    rng.shuffle(train_order)
    rng.shuffle(eval_order)
    train_target = args.train_size if exact_possible else len(train_order)
    eval_target = args.eval_size if exact_possible else len(eval_order)
    train_sel = train_order[:train_target]
    eval_sel = eval_order[:eval_target]

    train = [split_item(train_rows[i], "train", resolver) for i in train_sel]
    eval_ = [split_item(eval_rows[i], "eval", resolver) for i in eval_sel]

    train_ids = {x["id"] for x in train}
    eval_ids = {x["id"] for x in eval_}
    if train_ids & eval_ids:
        raise ValueError("train/eval id overlap")
    train_q = {x["question_hash"] for x in train}
    eval_q = {x["question_hash"] for x in eval_}

    out = {
        "name": "heima_ab_loss1_tiny_acceptance_v1",
        "seed": args.seed,
        "source_subset": str(args.subset_root),
        "train_file": str(train_path),
        "eval_file": str(eval_path),
        "requested_train_size": args.train_size,
        "requested_eval_size": args.eval_size,
        "train_size": len(train),
        "eval_size": len(eval_),
        "exact_requested_size": len(train) == args.train_size and len(eval_) == args.eval_size,
        "train_eval_question_hash_overlap": len(train_q & eval_q),
        "note": "Deterministic tiny real-image split from existing chartqa_sqa_v1 micro subset; does not use full 98582 train.jsonl.",
        "train": train,
        "eval": eval_,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "train": len(train), "eval": len(eval_), "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
