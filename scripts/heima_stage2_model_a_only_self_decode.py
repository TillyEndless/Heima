#!/usr/bin/env python3
"""Model-A-only online self-decode Stage2 entrypoint.

This script intentionally does not load Model B, create interpreter/projector
parameters, or save B checkpoints. It reuses the strict Heima VLM first-pass
format to extract continuous section latents from Model A, then calls the same
Model A N times to reconstruct section CoTs from those latents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_data_small_vlm_official_sections as base
from src.heima_stage2.model_a_only_self_decode import (
    AOnlySelfDecodeMode,
    FirstPassOutput,
    evaluate_self_decode_interventions,
    run_a_only_train_step,
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def parse_sections(raw: str) -> tuple[str, ...]:
    sections = tuple(s.strip() for s in raw.split(",") if s.strip())
    if not sections:
        raise argparse.ArgumentTypeError("--sections must contain at least one section")
    return sections


def load_stage0_checkpoint_if_present(model_a, stage0_checkpoint: str | None, device: torch.device, report_path: Path | None = None) -> dict:
    report = {"stage0_checkpoint": stage0_checkpoint, "loaded": False}
    if not stage0_checkpoint:
        return report
    payload = torch.load(stage0_checkpoint, map_location=device)
    state = payload.get("model_a", payload)
    target = model_a.state_dict()
    candidates = [("raw", state)]
    candidates.append(("strip_model_prefix_per_key", {
        (k[len("model."):] if isinstance(k, str) and k.startswith("model.") else k): v
        for k, v in state.items()
    }))
    qwen25_bridge = {}
    for k, v in state.items():
        if isinstance(k, str) and k.startswith("model.visual."):
            qwen25_bridge[k[len("model."):]] = v
        elif isinstance(k, str) and k.startswith("model.language_model."):
            qwen25_bridge["model." + k[len("model.language_model."):]] = v
        else:
            qwen25_bridge[k] = v
    candidates.append(("qwen25_vl_model_bridge", qwen25_bridge))

    best_name, best_state, best_matches = None, None, []
    for name, candidate in candidates:
        matches = [k for k, v in candidate.items() if k in target and tuple(v.shape) == tuple(target[k].shape)]
        if len(matches) > len(best_matches):
            best_name, best_state, best_matches = name, candidate, matches
    if best_state is None or not best_matches:
        raise RuntimeError(f"Stage0 checkpoint has no compatible tensors: {stage0_checkpoint}")

    filtered = {k: best_state[k] for k in best_matches}
    missing, unexpected = model_a.load_state_dict(filtered, strict=False)
    report = {
        "stage0_checkpoint": stage0_checkpoint,
        "loaded": True,
        "key_transform": best_name,
        "checkpoint_tensors": len(state),
        "loaded_tensors": len(filtered),
        "target_tensors": len(target),
        "missing_after_partial_load": len(missing),
        "unexpected_after_partial_load": len(unexpected),
        "first_missing": list(missing)[:20],
        "first_unexpected": list(unexpected)[:20],
    }
    if report_path is not None:
        write_json(report_path, report)
    return report


def make_first_pass_fn(processor, tokenizer_a, args):
    def first_pass(model_a, records):
        main, _logits, _labels, z, _trace = base.encoder_forward(model_a, processor, tokenizer_a, list(records), args)
        return FirstPassOutput(main_loss=main, latents=z)

    return first_pass


def manifest(args, status: str) -> dict:
    return {
        "status": status,
        "framework": "Model-A-only online self-decode supervision",
        "model_A": args.model_a_path,
        "model_B": None,
        "has_model_b": False,
        "optimizer_contains_model_b": False,
        "use_projector": False,
        "use_role_embedding": False,
        "extra_trainable_params_except_A": 0,
        "dataset": str(args.dataset_path),
        "image_root": str(args.image_root),
        "sections": list(args.sections),
        "lambda_self": args.lambda_self,
        "self_decode_with_image": args.self_decode_with_image,
        "stage0_checkpoint": args.stage0_checkpoint,
        "forward_contract": {
            "first_pass": "A(image, question, latent_cot_1..latent_cot_N, answer) -> L_main and z_i from last hidden state at latent markers",
            "second_pass": "A(explain_prompt_i, question/context, continuous z_i, section_prefix_i, text_cot_i) -> L_cot_i",
            "expected_forward_count_per_batch": len(args.sections) + 1,
        },
        "loss_contract": {
            "a_only_main_baseline": "L_total = L_main; self-decode forwards are eval/log only with detached z and no gradient contribution",
            "a_only_self_decode": "L_total = L_main + lambda_self * mean_i(L_cot_i); z is not detached",
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_component_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        "args": vars(args),
    }


def run_smoke_backward_only(args, run_dir: Path) -> dict:
    base.set_seed(args.seed)
    train = read_jsonl(args.dataset_path / "train.jsonl")[: max(1, args.max_train_samples or 1)]
    missing = [str(base.image_path(args, row)) for row in train if not base.image_path(args, row).exists()]
    if missing:
        raise FileNotFoundError(f"missing images: {missing[:5]} count={len(missing)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, tokenizer_a, model_a = base.load_vlm_a(args, device)
    load_stage0_checkpoint_if_present(model_a, args.stage0_checkpoint, device, run_dir / "stage0_load_report.json")
    optimizer = base.make_optimizer([{"params": model_a.parameters(), "lr": args.lr_a}], args)
    first_pass = make_first_pass_fn(processor, tokenizer_a, args)
    records = train[: args.batch_size]
    outputs = {}
    for mode in (AOnlySelfDecodeMode.A_ONLY_MAIN_BASELINE, AOnlySelfDecodeMode.A_ONLY_SELF_DECODE):
        step_out = run_a_only_train_step(
            model_a=model_a,
            optimizer_a=optimizer,
            records=records,
            tokenizer=tokenizer_a,
            first_pass_fn=first_pass,
            mode=mode,
            lambda_self=args.lambda_self,
            sections=args.sections,
            max_q=args.max_q,
            max_target=args.max_target,
            step_optimizer=False,
        )
        outputs[mode.value] = step_out.__dict__ | {"mode": step_out.mode.value}
    write_json(run_dir / "smoke_backward_only.json", outputs)
    return outputs


def train_stage2(args, run_dir: Path) -> dict:
    base.set_seed(args.seed)
    train = read_jsonl(args.dataset_path / "train.jsonl")
    val = read_jsonl(args.dataset_path / "validation.jsonl")[: args.max_eval_samples]
    if args.max_train_samples:
        train = train[: args.max_train_samples]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, tokenizer_a, model_a = base.load_vlm_a(args, device)
    load_stage0_checkpoint_if_present(model_a, args.stage0_checkpoint, device, run_dir / "stage0_load_report.json")
    optimizer = base.make_optimizer([{"params": model_a.parameters(), "lr": args.lr_a}], args)
    first_pass = make_first_pass_fn(processor, tokenizer_a, args)
    logs = []
    started = time.time()
    for step in range(1, args.max_steps + 1):
        records = base.batch_rows(train, args.batch_size, step - 1)
        out = run_a_only_train_step(
            model_a=model_a,
            optimizer_a=optimizer,
            records=records,
            tokenizer=tokenizer_a,
            first_pass_fn=first_pass,
            mode=args.mode,
            lambda_self=args.lambda_self,
            sections=args.sections,
            max_q=args.max_q,
            max_target=args.max_target,
            step_optimizer=True,
        )
        logs.append(out.__dict__ | {"mode": out.mode.value, "step": step})
        if args.save_every and step % args.save_every == 0:
            base.save_ckpt(run_dir / "checkpoints" / f"model_a_step{step}.pt", model_a=model_a.state_dict())
    interventions = evaluate_self_decode_interventions(
        model_a=model_a,
        tokenizer=tokenizer_a,
        records=val,
        first_pass_fn=first_pass,
        sections=args.sections,
        max_q=args.max_q,
        max_target=args.max_target,
    ) if val else {}
    result = {
        "mode": args.mode,
        "runtime_sec": time.time() - started,
        "logs": logs,
        "latent_intervention_eval": interventions,
        "has_model_b": False,
        "optimizer_contains_model_b": False,
        "use_projector": False,
        "use_role_embedding": False,
        "extra_trainable_params_except_A": 0,
    }
    write_json(run_dir / "result.json", result)
    if args.save_every:
        base.save_ckpt(run_dir / "checkpoints" / "model_a_final.pt", model_a=model_a.state_dict())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a_path", "--model-a-path", default="/data/zxl/small_models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--dataset_path", "--dataset-path", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_available_images_v1"))
    parser.add_argument("--image_root", "--image-root", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1/image_files"))
    parser.add_argument("--output_dir", "--output-dir", type=Path, default=Path("/data/zxl/runs/model_a_only_self_decode_v0"))
    parser.add_argument("--sections", type=parse_sections, default=parse_sections("summary,caption,reasoning"))
    parser.add_argument("--lambda_self", "--lambda-self", type=float, default=0.05)
    parser.add_argument("--stage0_checkpoint", "--stage0-checkpoint", default=None)
    parser.add_argument("--max_train_samples", "--max-train-samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", "--max-eval-samples", type=int, default=8)
    parser.add_argument("--max_steps", "--max-steps", type=int, default=1)
    parser.add_argument("--eval_every", "--eval-every", type=int, default=0)
    parser.add_argument("--save_every", "--save-every", type=int, default=0)
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument("--smoke_backward_only", "--smoke-backward-only", action="store_true")
    parser.add_argument("--mode", choices=[m.value for m in AOnlySelfDecodeMode], default=AOnlySelfDecodeMode.A_ONLY_SELF_DECODE.value)
    parser.add_argument("--self_decode_with_image", "--self-decode-with-image", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-a", type=float, default=1e-5)
    parser.add_argument("--optimizer", choices=["adafactor", "adamw"], default="adafactor")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--max-q", type=int, default=160)
    parser.add_argument("--max-target", type=int, default=160)
    parser.add_argument("--max-image-side", type=int, default=336)
    parser.add_argument("--torch-dtype", default="bfloat16")
    args = SimpleNamespace(**vars(parser.parse_args()))
    args.model_a_path = args.model_a_path
    args.image_root = args.image_root
    args.dataset_path = args.dataset_path
    args.self_decode_with_image = bool(args.self_decode_with_image)
    if args.self_decode_with_image:
        raise SystemExit("self_decode_with_image=true is reserved for a future ablation; v0 default must be false")

    run_dir = args.output_dir / f"seed{args.seed}" / time.strftime("%Y%m%d_%H%M%S")
    write_json(run_dir / "experiment_manifest.json", manifest(args, "dry_run" if args.dry_run else "running"))
    if args.dry_run:
        print(json.dumps(manifest(args, "dry_run"), indent=2, ensure_ascii=False, sort_keys=True, default=str))
        return 0
    if args.smoke_backward_only:
        print(json.dumps(run_smoke_backward_only(args, run_dir), indent=2, ensure_ascii=False, sort_keys=True, default=str))
        write_json(run_dir / "experiment_manifest.json", manifest(args, "smoke_complete"))
        return 0
    result = train_stage2(args, run_dir)
    write_json(run_dir / "experiment_manifest.json", manifest(args, "completed"))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
