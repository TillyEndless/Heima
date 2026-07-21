#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from src.heima_aligned.protocol import (
    HeimaRecord,
    STAGES,
    THINKING_TOKENS,
    build_decoder_sample,
    build_encoder_sample,
    config_hash,
    load_heima_records,
    mode_plan,
    split_hash,
)
from src.heima_aligned.tiny_acceptance_eval import build_tiny_acceptance_eval_manifest

STAGE_ORDER = [
    "prepare_data",
    "explicit_cot_sft",
    "progressive_summary",
    "progressive_caption",
    "progressive_reasoning",
    "recover",
    "train_interpreter_summary",
    "train_interpreter_caption",
    "train_interpreter_reasoning",
    "ours_joint_summary",
    "ours_joint_caption",
    "ours_joint_reasoning",
    "ours_recover",
    "eval_encoder",
    "eval_decoder",
    "eval_causal",
]


def read_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    # The shipped configs are JSON-compatible enough for tests only when PyYAML is unavailable.
    raise RuntimeError("PyYAML is required to read heima_aligned YAML configs")


def shell(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def completion_marker(run_dir: Path, stage: str) -> Path:
    return run_dir / "stages" / stage / "COMPLETED"


def create_micro_fixture(run_dir: Path) -> Path:
    fixture = run_dir / "fixtures" / "micro.jsonl"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": "micro-0",
            "image": "micro0.jpg",
            "question": "What is the object doing?",
            "summary": "Identify the main object and action.",
            "caption": "The image shows a person holding an object.",
            "reasoning": "The question asks for the action, so visual evidence is mapped to a concise answer.",
            "answer": "The object is being held.",
        },
        {
            "id": "micro-1",
            "image": "micro1.jpg",
            "question": "What color is the sign?",
            "summary": "Locate the sign in the image.",
            "caption": "A sign is visible near the center of the scene.",
            "reasoning": "After locating the sign, inspect its dominant color before answering.",
            "answer": "The sign is red.",
        },
    ]
    with fixture.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return fixture


def resolve_records(cfg: dict[str, Any], *, smoke: bool, run_dir: Path) -> tuple[list[HeimaRecord], dict[str, Any]]:
    data_cfg = cfg.get("data", {})
    report: dict[str, Any] = {"smoke": smoke}
    use_configured_for_dry_run = bool(data_cfg.get("use_configured_data_for_dry_run", False))
    if smoke and not use_configured_for_dry_run:
        path = create_micro_fixture(run_dir)
        records = load_heima_records(path)
        report.update({"source": str(path), "split_hash": split_hash(records), "count": len(records)})
        return records, report
    full = Path(data_cfg.get("prepared_json_dir", "")) / data_cfg.get("train_file", "")
    if not full.exists():
        missing = {
            "status": "missing_full_data",
            "expected_train_file": str(full),
            "message": "Full LLaVA-CoT-100k prepared JSON is not present. Run scripts/heima_aligned/prepare_data.sh after approving download/extraction space.",
        }
        write_json(run_dir / "data_missing_report.json", missing)
        raise FileNotFoundError(missing["message"])
    records = load_heima_records(full)
    report.update({"source": str(full), "split_hash": split_hash(records), "count": len(records)})
    return records, report


def selected_stages(mode: str, from_stage: str | None, to_stage: str | None) -> list[dict[str, Any]]:
    plan = mode_plan(mode)
    names = [p["stage"] for p in plan]
    if from_stage:
        plan = plan[names.index(from_stage):]
        names = [p["stage"] for p in plan]
    if to_stage:
        plan = plan[: names.index(to_stage) + 1]
    return plan


def run_stage(stage_spec: dict[str, Any], run_dir: Path, records: list[HeimaRecord], *, dry_run: bool, smoke: bool) -> dict[str, Any]:
    stage = stage_spec["stage"]
    stage_dir = run_dir / "stages" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    sample_artifact: dict[str, Any] | None = None
    if stage.startswith("progressive") or stage in {"explicit_cot_sft", "recover"}:
        encoder_stage = stage_spec.get("encoder_stage", "stage_0_explicit" if stage == "explicit_cot_sft" else "stage_4_recover")
        sample_artifact = build_encoder_sample(records[0], encoder_stage)
    elif stage.startswith("train_interpreter"):
        section = stage_spec.get("section", stage.rsplit("_", 1)[-1])
        sample_artifact = build_decoder_sample(records[0], section)
    elif stage.startswith("joint_reasoning"):
        sample_artifact = {
            "encoder_sample": build_encoder_sample(records[0], "stage_reasoning_only"),
            "decoder_sample": build_decoder_sample(records[0], "reasoning"),
            "detach_encoder_latent": bool(stage_spec.get("detach_encoder_latent", False)),
            "interventions_required": ["q_only", "correct", "shuffle", "zero"],
            "metrics_required": [
                "reasoning_reconstruction_full_nll",
                "reasoning_content_token_nll",
                "numeric_token_accuracy",
                "entity_token_accuracy",
                "answer_token_accuracy",
                "generation_exact_match",
                "answer_accuracy",
            ],
            "semantic_diagnostics": build_tiny_acceptance_eval_manifest({"reasoning": records[0].reasoning, "answer": records[0].answer}),
        }
    elif stage.startswith("eval"):
        sample_artifact = build_tiny_acceptance_eval_manifest({"reasoning": records[0].reasoning, "answer": records[0].answer})
        sample_artifact.update({"evaluation_stage": stage, "profiles": ["reasoning_acceptance", "causal_deterministic"]})
    else:
        sample_artifact = {"training_stage": stage, "mode_specific": stage_spec}
    log = {
        "stage": stage,
        "dry_run": dry_run,
        "smoke": smoke,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage_spec": stage_spec,
        "sample_artifact": sample_artifact,
        "note": "Protocol dry-run/smoke validates stage wiring only; full training is intentionally not launched by this task.",
    }
    write_json(stage_dir / "stage_manifest.json", log)
    completion_marker(run_dir, stage).write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output-root", type=Path, default=Path("/data/zxl/runs"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--from-stage")
    ap.add_argument("--to-stage")
    ap.add_argument("--gpus", default="")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    cfg = read_config(args.config)
    cfg_hash = config_hash(cfg)
    run_dir = args.output_root / f"heima_aligned_{args.run_id}"
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"output exists and --resume was not passed: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_cfg_hash = run_dir / "config.sha256"
    if args.resume and existing_cfg_hash.exists() and existing_cfg_hash.read_text().strip() != cfg_hash:
        raise RuntimeError("--resume config hash mismatch")
    existing_cfg_hash.write_text(cfg_hash + "\n")
    write_json(run_dir / "resolved_config.json", cfg)
    records, data_report = resolve_records(cfg, smoke=args.smoke or args.dry_run, run_dir=run_dir)
    official_reference = ROOT / "docs" / "heima_alignment" / "official_reference.json"
    launch = {
        "run_id": args.run_id,
        "mode": args.mode,
        "cli": sys.argv,
        "cwd": str(Path.cwd()),
        "git_commit": shell(["git", "rev-parse", "HEAD"]),
        "official_reference": str(official_reference),
        "config_hash": cfg_hash,
        "data": data_report,
        "gpus": args.gpus,
        "environment": {"hostname": shell(["hostname"]), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
    }
    write_json(run_dir / "launch_manifest.json", launch)
    plan = selected_stages(args.mode, args.from_stage, args.to_stage)
    if args.skip_train:
        plan = [p for p in plan if p["stage"].startswith("eval")]
    if args.skip_eval:
        plan = [p for p in plan if not p["stage"].startswith("eval")]
    completed = []
    for spec in plan:
        marker = completion_marker(run_dir, spec["stage"])
        if args.resume and marker.exists():
            completed.append({"stage": spec["stage"], "status": "skipped_completed"})
            continue
        completed.append(run_stage(spec, run_dir, records, dry_run=args.dry_run, smoke=args.smoke))
    summary = {"status": "complete", "mode": args.mode, "run_dir": str(run_dir), "completed": completed}
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:4000])

if __name__ == "__main__":
    main()
