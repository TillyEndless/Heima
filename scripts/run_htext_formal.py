#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.htext.formal_eval import (
    evaluate_h0_checkpoint,
    evaluate_h1_interventions,
    generate_h1_samples,
    write_json,
)
from src.htext.synthetic_data import generate_synthetic_split, read_jsonl, write_jsonl
from src.htext.trainer import train_h0, train_h1


def h0_overfit_pass(report: dict, train_eval: dict) -> tuple[bool, list[str]]:
    reasons = []
    logs = report.get("logs", [])
    if not logs:
        reasons.append("missing H0 logs")
        return False, reasons
    if not all(log.get("model_a_grad_finite", False) for log in logs):
        reasons.append("non-finite H0 gradients")
    initial = logs[0]["loss_main"]
    final = logs[-1]["loss_main"]
    if final >= initial:
        reasons.append(f"H0 loss did not decrease: initial={initial}, final={final}")
    if train_eval["thinking_token_accuracy"] < 0.95:
        reasons.append(f"thinking token accuracy below 0.95: {train_eval['thinking_token_accuracy']}")
    if train_eval["answer_nll"] > initial:
        reasons.append(f"train answer NLL worse than initial main loss: {train_eval['answer_nll']} > {initial}")
    return not reasons, reasons


def summarize_generation(samples: list[dict]) -> dict:
    out = {}
    for variant in ["normal", "shuffled"]:
        rows = [sample["generations"][variant] for sample in samples]
        out[variant] = {
            key: sum(bool(row[key]) for row in rows) / max(len(rows), 1)
            for key in ["number_match", "expression_match", "intermediate_result_match", "operation_type_match"]
        }
    return out


def delta(a: dict, b: dict, path: list[str]) -> float | None:
    cur_a = a
    cur_b = b
    for key in path:
        cur_a = cur_a.get(key) if isinstance(cur_a, dict) else None
        cur_b = cur_b.get(key) if isinstance(cur_b, dict) else None
    if cur_a is None or cur_b is None:
        return None
    return cur_a - cur_b


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument("--h0-steps", type=int, default=300)
    parser.add_argument("--h1-steps", type=int, default=300)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--intervention-samples", type=int, default=64)
    parser.add_argument("--generation-samples", type=int, default=16)
    args = parser.parse_args()

    reports_dir = Path("experiments/htext_gpt2/reports")
    checkpoints_dir = Path("experiments/htext_gpt2/checkpoints")
    data_dir = Path("experiments/htext_gpt2/data")
    reports_dir.mkdir(parents=True, exist_ok=True)

    h0_summary = {}
    h1_summaries = {"q": {}, "z": {}, "qz": {}}
    intervention_summary = {}
    sample_lines = []

    for seed in args.seeds:
        train, val = generate_synthetic_split(args.train_size, args.validation_size, seed)
        train_path = data_dir / f"synthetic_htext_train_seed_{seed}.jsonl"
        val_path = data_dir / f"synthetic_htext_validation_seed_{seed}.jsonl"
        write_jsonl(train, train_path)
        write_jsonl(val, val_path)

        h0_ckpt = checkpoints_dir / f"h0_seed_{seed}"
        h0_report_path = reports_dir / f"h0_seed_{seed}.json"
        h0_overrides = {
            "seed": seed,
            "train_path": str(train_path),
            "validation_path": str(val_path),
            "max_steps": args.h0_steps,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": 1,
            "eval_samples": min(args.validation_size, 128),
            "report_path": str(h0_report_path),
            "checkpoint_dir": str(h0_ckpt),
            "eval_batch_size": args.eval_batch_size,
            "intervention_samples": args.intervention_samples,
        }
        h0_report = train_h0("experiments/htext_gpt2/configs/h0.yaml", h0_overrides)
        train_eval = evaluate_h0_checkpoint(h0_report["config"], h0_ckpt, train)
        val_eval = evaluate_h0_checkpoint(h0_report["config"], h0_ckpt, val)
        ok, reasons = h0_overfit_pass(h0_report, train_eval)
        h0_summary[str(seed)] = {
            "report_path": str(h0_report_path),
            "checkpoint_dir": str(h0_ckpt),
            "initial_main_loss": h0_report["logs"][0]["loss_main"],
            "final_main_loss": h0_report["logs"][-1]["loss_main"],
            "train": train_eval,
            "validation": val_eval,
            "gate_pass": ok,
            "gate_reasons": reasons,
            "peak_gpu_memory_mb": max((log.get("peak_gpu_memory_mb", 0.0) for log in h0_report.get("logs", [])), default=0.0),
            "runtime_sec": h0_report["runtime_sec"],
        }
        write_json(reports_dir / "h0_summary.json", h0_summary)
        if not ok:
            write_json(reports_dir / "h1_cross_seed_summary.json", {"status": "stopped_after_h0", "h0_summary": h0_summary})
            return 2

        for mode in ["q", "z", "qz"]:
            h1_ckpt_dir = checkpoints_dir / f"h1_{mode}_seed_{seed}"
            h1_report_path = reports_dir / f"h1_{mode}_seed_{seed}.json"
            h1_overrides = {
                "seed": seed,
                "h1_mode": mode,
                "train_path": str(train_path),
                "validation_path": str(val_path),
                "h0_checkpoint_dir": str(h0_ckpt),
                "max_steps": args.h1_steps,
                "micro_batch_size": args.micro_batch_size,
                "gradient_accumulation_steps": 1,
                "eval_samples": min(args.validation_size, 128),
                "report_path": str(h1_report_path),
                "checkpoint_dir": str(h1_ckpt_dir),
                "eval_batch_size": args.eval_batch_size,
                "intervention_samples": args.intervention_samples,
            }
            h1_report = train_h1("experiments/htext_gpt2/configs/h1.yaml", h1_overrides)
            if any(log.get("model_a_frozen_grad_norm", 0.0) != 0.0 for log in h1_report["logs"]):
                raise RuntimeError(f"H1-{mode} seed {seed} produced A gradients")
            eval_records = read_jsonl(val_path)[: args.intervention_samples]
            interventions = evaluate_h1_interventions(
                h1_report["config"],
                h0_ckpt,
                Path(h1_report["config"]["checkpoint_dir"]) / "h1.pt",
                eval_records,
                mode,
            )
            h1_summaries[mode][str(seed)] = {
                "report_path": str(h1_report_path),
                "checkpoint_dir": str(h1_ckpt_dir),
                "first_loss1": h1_report["logs"][0]["loss1"],
                "final_loss1": h1_report["logs"][-1]["loss1"],
                "eval": h1_report["eval"],
                "interventions": interventions,
                "runtime_sec": h1_report["runtime_sec"],
            }
            write_json(reports_dir / f"h1_{mode}_summary.json", h1_summaries[mode])
            if mode == "qz":
                samples = generate_h1_samples(
                    h1_report["config"],
                    h0_ckpt,
                    Path(h1_report["config"]["checkpoint_dir"]) / "h1.pt",
                    eval_records[: args.generation_samples],
                    mode,
                )
                intervention_summary[str(seed)] = {
                    "interventions": interventions,
                    "generation": summarize_generation(samples),
                }
                for sample in samples:
                    sample_lines.extend([
                        f"seed={seed} id={sample['id']}",
                        f"Q: {sample['question']}",
                        f"Gold: {sample['gold_cot']}",
                        f"Normal: {sample['generations']['normal']['text']}",
                        f"Shuffled: {sample['generations']['shuffled']['text']}",
                        "",
                    ])
                write_json(reports_dir / "h1_intervention_summary.json", intervention_summary)
                (reports_dir / "h1_sample_decodes.txt").write_text("\n".join(sample_lines), encoding="utf-8")

    cross = {"seeds": args.seeds, "h0": h0_summary, "h1": h1_summaries, "h1_qz_intervention": intervention_summary}
    direction = {}
    for seed in args.seeds:
        s = str(seed)
        q = h1_summaries["q"][s]["interventions"]["normal"]["nll"]
        qz = h1_summaries["qz"][s]["interventions"]["normal"]["nll"]
        shuffle = h1_summaries["qz"][s]["interventions"]["shuffled"]["nll"]
        direction[s] = {
            "delta_QZ_over_Q": {
                key: delta(q, qz, [key])
                for key in ["full", "first_8_tokens", "numeric_tokens", "intermediate_tokens"]
            },
            "delta_normal_over_shuffle": {
                key: delta(shuffle, qz, [key])
                for key in ["full", "first_8_tokens", "numeric_tokens", "intermediate_tokens"]
            },
            "generation_normal_number_minus_shuffle": intervention_summary[s]["generation"]["normal"]["number_match"] - intervention_summary[s]["generation"]["shuffled"]["number_match"],
        }
    cross["directional_tests"] = direction
    cross["allow_h2"] = False
    cross["allow_h2_reason"] = "This script only establishes H0/H1 baseline; H2 requires manual review of directional tests."
    write_json(reports_dir / "h1_cross_seed_summary.json", cross)
    print(json.dumps({"status": "complete", "reports_dir": str(reports_dir), "seeds": args.seeds}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

