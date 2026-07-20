#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_data_small_vlm_official_sections import (  # noqa: E402
    SECTIONS,
    THINKING_TOKENS,
    evaluate,
    image_path,
    load_vlm_a,
    read_jsonl,
    train_s1,
    write_json,
)
from src.htext.heima_reuse import backend_resolution_snapshot  # noqa: E402


def shell(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def latest_recover_checkpoint(root: Path) -> Path:
    candidates = sorted(root.glob("seed*/20*/checkpoints/recover_all.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no recover_all.pt under {root}")
    return candidates[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--recover-checkpoint", type=Path, default=None)
    p.add_argument("--progressive-root", type=Path, default=Path("/data/zxl/runs/heima_small_vlm_progressive_sections"))
    p.add_argument("--model-a-path", default="/data/zxl/small_models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--model-b-path", default="/data/zxl/small_models/Qwen2.5-0.5B-Instruct")
    p.add_argument("--subset", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_available_images_v1"))
    p.add_argument("--image-root", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1/image_files"))
    p.add_argument("--out", type=Path, default=Path("/data/zxl/runs/heima_small_vlm_progressive_interpreters"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--s1-steps", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-samples", type=int, default=8)
    p.add_argument("--lr-a", type=float, default=1e-5)
    p.add_argument("--lr-b", type=float, default=2e-5)
    p.add_argument("--lr-projector", type=float, default=1e-4)
    p.add_argument("--lambda1", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["adafactor", "adamw"], default="adafactor")
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--max-q", type=int, default=160)
    p.add_argument("--max-target", type=int, default=160)
    p.add_argument("--max-image-side", type=int, default=336)
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()
    args.recover_checkpoint = args.recover_checkpoint or latest_recover_checkpoint(args.progressive_root)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.out / f"seed{args.seed}" / run_id
    train = read_jsonl(args.subset / "train.jsonl")
    val = read_jsonl(args.subset / "validation.jsonl")
    test = read_jsonl(args.subset / "test.jsonl")
    missing = [str(image_path(args, r)) for r in train + val + test if not image_path(args, r).exists()]
    if missing:
        raise FileNotFoundError(f"missing images: {missing[:5]} count={len(missing)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, tokenizer_a, model_a = load_vlm_a(args, device)
    ckpt = torch.load(args.recover_checkpoint, map_location=device)
    model_a.load_state_dict(ckpt["model_a"], strict=True)
    manifest = {
        "status": "running",
        "task": "Freeze progressive recover_all MLLM A and train three official-section interpreters",
        "run_id": run_id,
        "host": shell(["hostname"]),
        "gpu": shell(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader"]),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "recover_checkpoint": str(args.recover_checkpoint),
        "sections": SECTIONS,
        "typed_tokens": THINKING_TOKENS,
        "args": vars(args),
    }
    write_json(run_dir / "experiment_manifest.json", manifest)
    initial = evaluate(model_a, processor, tokenizer_a, {}, {}, None, [], args) if False else None
    started = time.time()
    tokenizer_b, decoders, projectors = train_s1(args, run_dir, processor, tokenizer_a, model_a, train, val)
    final_val = evaluate(model_a, processor, tokenizer_a, decoders, projectors, tokenizer_b, val[: args.eval_samples], args)
    final_test = evaluate(model_a, processor, tokenizer_a, decoders, projectors, tokenizer_b, test[: args.eval_samples], args)
    summary = {
        "status": "complete",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": time.time() - started,
        "recover_checkpoint": str(args.recover_checkpoint),
        "initial": initial,
        "validation": final_val,
        "test": final_test,
        "backend_resolution": backend_resolution_snapshot(),
    }
    write_json(run_dir / "summary.json", summary)
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = summary["completed_at_utc"]
    write_json(run_dir / "experiment_manifest.json", manifest)
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
