#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable

DATASET_ID = "a-m-team/AM-DeepSeek-R1-Distilled-1.4M"
DEFAULT_CONFIG = "am_0.5M"
DEFAULT_OUTPUT = "/data2/zhouxiaoling/latent_cot/am_deepseek_runs/AM_DEEPSEEK_R1_DISTILLED_90K_TRAIN_5K_EVAL/manifest_train90k_eval5k_floor_nocap.json"
VISUAL_RE = re.compile(r"\b(image|picture|photo|chart|diagram|shown|visual|bounding box|bbox|ocr|document displays|the image shows)\b", re.I)
THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.I | re.S)
FINAL_MARKERS = ["####", "The answer is", "Final answer:", "Answer:", "答案", "最终答案"]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".zst":
        import zstandard as zstd
        fh = open(path, "rb")
        reader = zstd.open(fh, "rt", encoding="utf-8")
        try:
            for line in reader:
                line = line.strip()
                if line:
                    yield json.loads(line)
        finally:
            reader.close()
            fh.close()
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_hf(config: str, split: str, cache_dir: str | None, streaming: bool):
    from datasets import load_dataset
    ds = load_dataset(DATASET_ID, config, split=split, cache_dir=cache_dir, streaming=streaming)
    for row in ds:
        yield row


def first_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    return str(v).strip()


def messages_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    msgs = row.get("messages") or row.get("conversations") or row.get("conversation") or []
    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs)
        except Exception:
            return []
    return msgs if isinstance(msgs, list) else []


def extract_question(row: dict[str, Any]) -> str:
    for k in ["question", "problem", "prompt", "input"]:
        if row.get(k):
            return first_text(row[k])
    for m in messages_from_row(row):
        role = str(m.get("role") or m.get("from") or "").lower()
        if role in {"user", "human"}:
            return first_text(m.get("content") or m.get("value"))
    return ""


def extract_info(row: dict[str, Any]) -> dict[str, Any]:
    info = row.get("info")
    if isinstance(info, dict):
        return info
    for m in messages_from_row(row):
        info = m.get("info")
        if isinstance(info, dict):
            return info
    return {}


def extract_assistant_text(row: dict[str, Any]) -> str:
    for k in ["response", "solution", "answer", "output", "completion"]:
        if row.get(k):
            return first_text(row[k])
    assistants = []
    for m in messages_from_row(row):
        role = str(m.get("role") or m.get("from") or "").lower()
        if role in {"assistant", "gpt", "model"}:
            assistants.append(first_text(m.get("content") or m.get("value")))
    return assistants[-1] if assistants else ""


def split_cot_and_answer(row: dict[str, Any]) -> tuple[str, str]:
    info = extract_info(row)
    assistant = extract_assistant_text(row)
    think_info = first_text(info.get("think_content"))
    ans_info = first_text(info.get("reference_answer") or info.get("answer_content") or row.get("reference_answer"))
    m = THINK_TAG_RE.search(assistant)
    if m:
        cot = m.group(1).strip()
        after = assistant[m.end():].strip()
        return cot, ans_info or after
    if think_info:
        return think_info, ans_info or assistant
    if ans_info and assistant:
        idx = assistant.rfind(ans_info)
        if idx > 0:
            return assistant[:idx].strip(), ans_info
    for marker in FINAL_MARKERS:
        idx = assistant.rfind(marker)
        if idx > 0:
            return assistant[:idx].strip(), ans_info or assistant[idx + len(marker):].strip()
    return assistant.strip(), ans_info


def token_count(tokenizer, text: str) -> int:
    if tokenizer is None:
        return max(1, len(text.split()))
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def normalize_row(row: dict[str, Any], idx: int, tokenizer, reject_visual: bool) -> dict[str, Any] | None:
    q = extract_question(row)
    cot, answer = split_cot_and_answer(row)
    info = extract_info(row)
    source = first_text(info.get("source") or row.get("source") or DATASET_ID)
    if not q or not cot or not answer:
        return None
    if reject_visual and VISUAL_RE.search(q + "\n" + cot):
        return None
    n_cot = token_count(tokenizer, cot)
    k = max(1, math.floor(0.5 * n_cot))
    return {
        "sample_id": f"am_deepseek_{idx}",
        "source": source,
        "dataset": DATASET_ID,
        "question": q,
        "gold_cot": cot,
        "answer": answer,
        "cot_token_count": n_cot,
        "raw_K": k,
        "latent_count": k,
        "k_rule": "floor(0.5*N_text_CoT), no cap",
    }


def load_tokenizer(model_id: str, revision: str | None, cache_dir: str | None):
    if not model_id:
        return None
    from transformers import AutoTokenizer
    kwargs = {"cache_dir": cache_dir, "trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision
    try:
        return AutoTokenizer.from_pretrained(model_id, local_files_only=True, **kwargs)
    except Exception:
        return AutoTokenizer.from_pretrained(model_id, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", default=None, help="Optional local AM jsonl/jsonl.zst; avoids HF dataset loading.")
    ap.add_argument("--dataset-config", default=DEFAULT_CONFIG)
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--train-size", type=int, default=90000)
    ap.add_argument("--eval-size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf-cache", default=os.environ.get("HF_HOME", "/weights2/zhouxiaoling/hf_cache"))
    ap.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--model-revision", default="916b56a44061fd5cd7d6a8fb632557ed4f724f60")
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--max-source-rows", type=int, default=0)
    ap.add_argument("--allow-visual-words", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    tokenizer = load_tokenizer(args.model_id, args.model_revision, args.hf_cache)
    need = args.train_size + args.eval_size
    rows: list[dict[str, Any]] = []
    source_iter = read_jsonl(Path(args.input_jsonl)) if args.input_jsonl else iter_hf(args.dataset_config, args.split, args.hf_cache, args.streaming)
    seen = missing = visual_rejected = 0
    for idx, raw in enumerate(source_iter):
        seen += 1
        item = normalize_row(raw, idx, tokenizer, reject_visual=not args.allow_visual_words)
        if item is None:
            raw_text = json.dumps(raw, ensure_ascii=False)[:4000]
            if VISUAL_RE.search(raw_text):
                visual_rejected += 1
            else:
                missing += 1
        else:
            rows.append(item)
        if len(rows) >= need:
            break
        if args.max_source_rows and seen >= args.max_source_rows:
            break
    if len(rows) < need:
        raise SystemExit(f"Not enough valid AM rows: got {len(rows)}, need {need}, seen {seen}, missing {missing}, visual_rejected {visual_rejected}")
    random.shuffle(rows)
    train = rows[: args.train_size]
    eval_rows = rows[args.train_size: args.train_size + args.eval_size]
    all_rows = train + eval_rows
    visual_count = sum(bool(VISUAL_RE.search(r["question"] + "\n" + r["gold_cot"])) for r in all_rows)
    meta = {
        "dataset_id": DATASET_ID,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "source": args.input_jsonl or DATASET_ID,
        "train": len(train),
        "eval": len(eval_rows),
        "seed": args.seed,
        "seen_source_rows": seen,
        "missing_or_unparseable_rows": missing,
        "visual_rejected_rows": visual_rejected,
        "visual_word_rows_after_filter": visual_count,
        "visual_word_rate_after_filter": visual_count / max(len(all_rows), 1),
        "k_protocol": "K=floor(0.5*cot_token_count), no hard cap",
        "train_K_floor_min": min(r["raw_K"] for r in train),
        "train_K_floor_median": sorted(r["raw_K"] for r in train)[len(train)//2],
        "train_K_floor_mean": sum(r["raw_K"] for r in train) / len(train),
        "train_K_floor_max": max(r["raw_K"] for r in train),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps({"train": train, "eval": eval_rows, "meta": meta}, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out)
    print(json.dumps({"output": str(out), "meta": meta}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
