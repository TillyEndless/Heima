#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from scripts.run_data_small_vlm_official_sections import (
    SECTIONS,
    THINKING_TOKENS,
    backend_resolution_snapshot,
    evaluate,
    image_path,
    load_vlm_a,
    read_jsonl,
    shell,
    train_s1,
    write_json,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder-checkpoint", type=Path, required=True)
    p.add_argument("--source-group", required=True)
    p.add_argument("--model-a-path", default="/data/zxl/small_models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--model-b-path", default="/data/zxl/small_models/Qwen2.5-0.5B-Instruct")
    p.add_argument("--subset", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_available_images_v1"))
    p.add_argument("--image-root", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1/image_files"))
    p.add_argument("--out", type=Path, default=Path("/data/zxl/runs/heima_small_vlm_fresh_b_eval"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--s1-steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-samples", type=int, default=16)
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
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--train-latent-marker-ntp", action="store_true")
    args = p.parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.out / args.source_group / run_id
    train = read_jsonl(args.subset / "train.jsonl")
    val = read_jsonl(args.subset / "validation.jsonl")
    test = read_jsonl(args.subset / "test.jsonl")
    missing = [str(image_path(args, r)) for r in train + val + test if not image_path(args, r).exists()]
    if missing:
        raise FileNotFoundError(f"missing images: {missing[:5]} count={len(missing)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, tokenizer_a, model_a = load_vlm_a(args, device)
    ckpt = torch.load(args.encoder_checkpoint, map_location=device)
    model_a.load_state_dict(ckpt["model_a"], strict=True)

    manifest = {
        "status": "running",
        "task": "Fresh B evaluator for frozen Model A latent readability",
        "source_group": args.source_group,
        "encoder_checkpoint": str(args.encoder_checkpoint),
        "run_id": run_id,
        "host": shell(["hostname"]),
        "gpu": shell(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader"]),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_a": {"path": args.model_a_path, "frozen": True},
        "model_b": {"path": args.model_b_path, "fresh_init": True},
        "sections": SECTIONS,
        "typed_tokens": THINKING_TOKENS,
        "strict_core": {
            "thinking_state_mode": "predictor",
            "projector": "HeimaOfficialAbstractProjection",
            "replacement": "official_embedding_replacement",
            "loss": "heima_ce_loss / CEWithChunkedOutputLoss when available",
        },
        "args": vars(args),
    }
    write_json(run_dir / "experiment_manifest.json", manifest)
    cfg = copy.deepcopy(manifest)
    write_json(run_dir / "groups" / "fresh_b_eval" / "config_full.json", cfg)

    started = time.time()
    tokenizer_b, decoders, projectors = train_s1(args, run_dir, processor, tokenizer_a, model_a, train, val)
    validation = evaluate(model_a, processor, tokenizer_a, decoders, projectors, tokenizer_b, val[: args.eval_samples], args)
    test_metrics = evaluate(model_a, processor, tokenizer_a, decoders, projectors, tokenizer_b, test[: args.eval_samples], args)
    summary = {
        "status": "complete",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": time.time() - started,
        "source_group": args.source_group,
        "encoder_checkpoint": str(args.encoder_checkpoint),
        "validation": validation,
        "test": test_metrics,
        "backend_resolution": backend_resolution_snapshot(),
    }
    write_json(run_dir / "summary.json", summary)
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = summary["completed_at_utc"]
    write_json(run_dir / "experiment_manifest.json", manifest)
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
