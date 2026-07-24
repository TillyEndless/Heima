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
    causal_lm_loss,
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


def load_stage0_checkpoint_if_present(model_a, stage0_checkpoint: str | None, device: torch.device) -> None:
    if not stage0_checkpoint:
        return
    payload = torch.load(stage0_checkpoint, map_location=device)
    state = payload.get("model_a", payload)
    model_a.load_state_dict(state, strict=True)


def make_first_pass_fn(processor, tokenizer_a, args):
    def first_pass(model_a, records):
        main, logits, _labels, z, trace = base.encoder_forward(model_a, processor, tokenizer_a, list(records), args)
        latent_labels = torch.full(logits.shape[:2], -100, dtype=torch.long, device=logits.device)
        for section in args.sections:
            token_id = tokenizer_a.convert_tokens_to_ids(base.THINKING_TOKENS[section])
            for row, pos in enumerate(trace[section]["thinking_pos"]):
                latent_labels[row, int(pos)] = int(token_id)
        latent_token_loss = causal_lm_loss(logits, latent_labels)
        return FirstPassOutput(main_loss=main, latents=z, latent_token_loss=latent_token_loss)

    return first_pass


def trainable_parameter_report(model_a) -> dict:
    named = list(model_a.named_parameters())
    trainable = [(name, p) for name, p in named if p.requires_grad]

    def grad_norm(params):
        total = 0.0
        has_grad = False
        for p in params:
            if p.grad is not None:
                has_grad = True
                total += float(p.grad.detach().float().pow(2).sum().item())
        return total ** 0.5, has_grad

    def module_param_stats(module):
        if module is None:
            return {"exists": False, "matched_names": [], "trainable_param_count": 0, "has_grad": False, "grad_norm": 0.0}
        module_param_ids = {id(p) for p in module.parameters(recurse=True)}
        matched = [(n, p) for n, p in trainable if id(p) in module_param_ids]
        norm, has_grad = grad_norm([p for _n, p in matched])
        return {
            "exists": True,
            "module_class": module.__class__.__name__,
            "matched_names": [n for n, _p in matched][:50],
            "trainable_param_count": int(sum(p.numel() for _n, p in matched)),
            "has_grad": has_grad,
            "grad_norm": norm,
        }

    def name_group_stats(patterns):
        matched = [(n, p) for n, p in trainable if any(pat in n for pat in patterns)]
        norm, has_grad = grad_norm([p for _n, p in matched])
        return {
            "matched_names": [n for n, _p in matched][:50],
            "trainable_param_count": int(sum(p.numel() for _n, p in matched)),
            "has_grad": has_grad,
            "grad_norm": norm,
        }

    input_emb = model_a.get_input_embeddings() if hasattr(model_a, "get_input_embeddings") else None
    output_emb = model_a.get_output_embeddings() if hasattr(model_a, "get_output_embeddings") else None
    input_ids = {id(p) for p in input_emb.parameters(recurse=True)} if input_emb is not None else set()
    output_ids = {id(p) for p in output_emb.parameters(recurse=True)} if output_emb is not None else set()
    peft_config = getattr(model_a, "peft_config", None)
    return {
        "total_param_count": int(sum(p.numel() for _n, p in named)),
        "trainable_param_count": int(sum(p.numel() for _n, p in trainable)),
        "trainable_name_count": len(trainable),
        "has_peft_config": peft_config is not None,
        "peft_config": str(peft_config) if peft_config is not None else None,
        "input_embeddings": module_param_stats(input_emb),
        "output_embeddings_or_lm_head": module_param_stats(output_emb),
        "output_tied_to_input_embeddings": bool(input_ids and output_ids and bool(input_ids & output_ids)),
        "lm_head_name_fallback": name_group_stats(("lm_head", "language_model.lm_head")),
        "sample_trainable_names": [n for n, _p in trainable[:100]],
        "all_name_count": len(named),
    }



@torch.no_grad()
def evaluate_answer_accuracy(model_a, processor, tokenizer_a, args, records: list[dict]) -> dict:
    if not records:
        return {"accuracy": None, "samples": 0}
    device = next(model_a.parameters()).device
    hits = 0
    total = 0
    examples = []
    for rec in records:
        batch = base.vlm_inputs(processor, args, [rec], include_answer=False).to(device)
        generated = model_a.generate(
            **batch,
            do_sample=False,
            max_new_tokens=32,
            pad_token_id=tokenizer_a.pad_token_id,
            eos_token_id=tokenizer_a.eos_token_id,
        )
        prompt_len = int(batch["attention_mask"][0].sum().item())
        text = tokenizer_a.decode(generated[0, prompt_len:], skip_special_tokens=True).strip()
        gold = str(rec.get("answer", "")).strip()
        ok = bool(gold and gold.lower() in text.lower())
        hits += int(ok)
        total += 1
        if len(examples) < 8:
            examples.append({"question": rec.get("question"), "gold_answer": gold, "generated": text, "correct": ok})
    return {"accuracy": hits / total if total else None, "samples": total, "match_rule": "gold answer substring in greedy generation", "examples": examples}

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
        "latent_token_loss_weight": args.latent_token_loss_weight,
        "self_decode_with_image": args.self_decode_with_image,
        "stage0_checkpoint": args.stage0_checkpoint,
        "forward_contract": {
            "first_pass": "A(image, question, latent_cot_1..latent_cot_N, answer) -> L_main and z_i from last hidden state at latent markers",
            "second_pass": "A(explain_prompt_i, question/context, continuous z_i, section_prefix_i, text_cot_i) -> L_cot_i",
            "expected_forward_count_per_batch": len(args.sections) + 1,
        },
        "loss_contract": {
            "a_only_main_baseline": "L_total = L_main + latent_token_loss_weight * L_latent_token; self-decode forwards are eval/log only with detached z",
            "a_only_self_decode": "L_total = L_main + lambda_self * mean_i(L_cot_i) + latent_token_loss_weight * L_latent_token; z is not detached",
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
    load_stage0_checkpoint_if_present(model_a, args.stage0_checkpoint, device)
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
            lambda_latent=args.latent_token_loss_weight,
            sections=args.sections,
            max_q=args.max_q,
            max_target=args.max_target,
            step_optimizer=False,
        )
        outputs[mode.value] = step_out.__dict__ | {"mode": step_out.mode.value}
    write_json(run_dir / "smoke_backward_only.json", outputs)
    write_json(run_dir / "trainable_parameter_report.json", trainable_parameter_report(model_a))
    return outputs


def train_stage2(args, run_dir: Path) -> dict:
    base.set_seed(args.seed)
    train = read_jsonl(args.dataset_path / "train.jsonl")
    val = read_jsonl(args.dataset_path / "validation.jsonl")[: args.max_eval_samples]
    if args.max_train_samples:
        train = train[: args.max_train_samples]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, tokenizer_a, model_a = base.load_vlm_a(args, device)
    load_stage0_checkpoint_if_present(model_a, args.stage0_checkpoint, device)
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
            lambda_latent=args.latent_token_loss_weight,
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
    answer_accuracy = evaluate_answer_accuracy(model_a, processor, tokenizer_a, args, val) if val else {"accuracy": None, "samples": 0}
    result = {
        "mode": args.mode,
        "runtime_sec": time.time() - started,
        "logs": logs,
        "answer_accuracy": answer_accuracy,
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
    parser.add_argument("--latent_token_loss_weight", "--latent-token-loss-weight", type=float, default=0.05)
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
