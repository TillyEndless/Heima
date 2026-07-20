#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_data_small_vlm_official_sections import backend_resolution_snapshot, read_jsonl, train_joint, write_json  # noqa: E402


def shell(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def latest_interpreter_run(root: Path) -> Path:
    candidates = sorted(root.glob("seed*/20*/checkpoints/s1_staged_sections.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no s1_staged_sections.pt under {root}")
    return candidates[0].parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--s1-run-dir", type=Path, default=None)
    p.add_argument("--interpreter-root", type=Path, default=Path("/data/zxl/runs/heima_small_vlm_progressive_interpreters"))
    p.add_argument("--model-a-path", default="/data/zxl/small_models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--model-b-path", default="/data/zxl/small_models/Qwen2.5-0.5B-Instruct")
    p.add_argument("--subset", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_available_images_v1"))
    p.add_argument("--image-root", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1/image_files"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--joint-steps", type=int, default=30)
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
    args.s1_run_dir = args.s1_run_dir or latest_interpreter_run(args.interpreter_root)

    train = read_jsonl(args.subset / "train.jsonl")
    val = read_jsonl(args.subset / "validation.jsonl")
    manifest = {
        "status": "running",
        "task": "Progressive recover A + staged interpreters joint Loss1 comparison",
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": shell(["hostname"]),
        "gpu": shell(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader"]),
        "s1_run_dir": str(args.s1_run_dir),
        "only_difference": "detach_encoder_latent true vs false",
        "loss": "Lmain + lambda1 * (L1_summary + L1_caption + L1_reasoning)",
        "args": vars(args),
    }
    write_json(args.s1_run_dir / "progressive_joint_manifest.json", manifest)
    detach = train_joint(args, args.s1_run_dir, train, val, True)
    ours = train_joint(args, args.s1_run_dir, train, val, False)
    summary = {
        "status": "complete",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "s1_run_dir": str(args.s1_run_dir),
        "joint_detach": detach,
        "ours_l1_no_detach": ours,
        "backend_resolution": backend_resolution_snapshot(),
    }
    write_json(args.s1_run_dir / "progressive_joint_summary.json", summary)
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = summary["completed_at_utc"]
    write_json(args.s1_run_dir / "progressive_joint_manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
