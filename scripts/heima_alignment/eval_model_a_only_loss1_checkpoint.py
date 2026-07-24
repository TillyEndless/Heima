#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_data_small_vlm_official_sections as base
from scripts.heima_stage2_model_a_only_self_decode import make_first_pass_fn, read_jsonl
from src.heima_stage2.model_a_only_self_decode import (
    build_self_decode_features,
    evaluate_self_decode_interventions,
    run_a_only_train_step,
    self_decode_forward,
)

SECTIONS = ("summary", "caption", "reasoning")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def load_model_checkpoint(model, checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model_a", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {"checkpoint": str(checkpoint), "loaded_tensors": len(state), "missing": len(missing), "unexpected": len(unexpected), "first_missing": list(missing)[:20]}


def mean(xs):
    return float(sum(xs) / max(1, len(xs)))


def token_accuracy(logits, labels) -> tuple[int, int]:
    pred = logits[:, :-1, :].argmax(dim=-1)
    gold = labels[:, 1:]
    mask = gold.ne(-100)
    correct = int((pred[mask] == gold[mask]).sum().item()) if mask.any() else 0
    total = int(mask.sum().item())
    return correct, total


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"the answer is", "", text)
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_match(pred: str, gold: str) -> bool:
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if not g:
        return False
    return p == g or g in p or p in g


def main_nll_and_answer(model, processor, tokenizer, rows, args, max_gen: int) -> dict:
    losses = []
    correct = 0
    gen_rows = []
    model.eval()
    for rec in rows:
        with torch.no_grad():
            loss, _logits, _labels, _z, _trace = base.encoder_forward(model, processor, tokenizer, [rec], args)
        losses.append(float(loss.detach().cpu().item()))
        generated = ""
        if max_gen > 0:
            inputs = base.vlm_inputs(processor, args, [rec], include_answer=False).to(next(model.parameters()).device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_gen, do_sample=False, use_cache=False)
            prompt_len = int(inputs["input_ids"].shape[1])
            generated = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()
            correct += int(answer_match(generated, rec["answer"]))
        gen_rows.append({"question": rec["question"], "gold_answer": rec["answer"], "generated_answer": generated, "condition": "normal_answer"})
    return {"validation_main_nll": mean(losses), "answer_generation_accuracy": correct / max(1, len(rows)) if max_gen > 0 else None, "answer_rows": gen_rows}


def self_decode_generate(model, tokenizer, rec, section: str, z, args, max_new: int) -> str:
    features = build_self_decode_features(model_a=model, tokenizer=tokenizer, records=[rec], section=section, z=z, max_q=args.max_q, max_target=1, include_latent=z is not None)
    # Drop the one dummy target token; keep prompt + latent + section prefix.
    keep = features.prompt_lengths[0] + (1 if z is not None else 0) + features.prefix_lengths[0]
    embeds = features.inputs_embeds[:, :keep, :]
    mask = features.attention_mask[:, :keep]
    with torch.no_grad():
        out = model.generate(inputs_embeds=embeds, attention_mask=mask, max_new_tokens=max_new, do_sample=False, use_cache=False)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def loss1_and_intervention(model, tokenizer, rows, first_pass, args) -> dict:
    per_section_losses = {s: [] for s in SECTIONS}
    per_section_acc = {s: [0, 0] for s in SECTIONS}
    margins = {s: [] for s in SECTIONS}
    totals = {s: {"correct": [], "shuffle": [], "zero": [], "q_only": [], "shuffle_margin": [], "zero_margin": [], "q_gain": []} for s in SECTIONS}
    model.eval()
    for start in range(0, len(rows), args.eval_batch_size):
        batch = rows[start:start + args.eval_batch_size]
        first = first_pass(model, batch)
        for section in SECTIONS:
            z = first.latents[section]
            variants = {
                "correct": z,
                "shuffle": torch.roll(z, shifts=1, dims=0) if z.size(0) > 1 else torch.zeros_like(z),
                "zero": torch.zeros_like(z),
                "q_only": None,
            }
            losses = {}
            for name, zvar in variants.items():
                with torch.no_grad():
                    loss, logits, labels, _features = self_decode_forward(model_a=model, tokenizer=tokenizer, records=batch, section=section, z=zvar, max_q=args.max_q, max_target=args.max_target, include_latent=zvar is not None)
                losses[name] = float(loss.detach().cpu().item())
                if name == "correct":
                    c, t = token_accuracy(logits, labels)
                    per_section_acc[section][0] += c
                    per_section_acc[section][1] += t
                    per_section_losses[section].append(losses[name])
            totals[section]["correct"].append(losses["correct"])
            totals[section]["shuffle"].append(losses["shuffle"])
            totals[section]["zero"].append(losses["zero"])
            totals[section]["q_only"].append(losses["q_only"])
            totals[section]["shuffle_margin"].append(losses["shuffle"] - losses["correct"])
            totals[section]["zero_margin"].append(losses["zero"] - losses["correct"])
            totals[section]["q_gain"].append(losses["q_only"] - losses["correct"])
            margins[section].append(losses["shuffle"] - losses["correct"])
    metrics = {}
    for section in SECTIONS:
        metrics[section] = {
            "ce_loss": mean(per_section_losses[section]),
            "token_accuracy": per_section_acc[section][0] / max(1, per_section_acc[section][1]),
        } | {k: mean(v) for k, v in totals[section].items()}
    all_margins = [x for s in SECTIONS for x in margins[s]]
    return {"loss1_reconstruction": metrics, "bootstrap": bootstrap_ci(all_margins, args.bootstrap_samples, args.seed), "margins": margins}


def bootstrap_ci(values, n: int, seed: int) -> dict:
    rng = random.Random(seed)
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    samples = []
    for _ in range(n):
        draw = [values[rng.randrange(len(values))] for _ in values]
        samples.append(mean(draw))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return {"mean": mean(values), "ci_low": lo, "ci_high": hi, "n": len(values)}


def gradient_audit(model, tokenizer, first_pass, row, args) -> dict:
    optimizer = base.make_optimizer([{"params": model.parameters(), "lr": args.lr_a}], args)
    out = run_a_only_train_step(model_a=model, optimizer_a=optimizer, records=[row], tokenizer=tokenizer, first_pass_fn=first_pass, mode="a_only_self_decode", lambda_self=args.lambda_self, sections=SECTIONS, max_q=args.max_q, max_target=args.max_target, step_optimizer=False)
    model.zero_grad(set_to_none=True)
    return {
        "grad_z_summary": out.grad_z_norm.get("summary", 0.0),
        "grad_z_caption": out.grad_z_norm.get("caption", 0.0),
        "grad_z_reasoning": out.grad_z_norm.get("reasoning", 0.0),
        "grad_A_from_loss1": out.grad_A_from_self_decode_norm,
        "grad_A_total": out.grad_A_total_norm,
        "finite": out.finite,
        "pass": out.grad_z_norm.get("summary", 0.0) > 0 and out.grad_z_norm.get("caption", 0.0) > 0 and out.grad_z_norm.get("reasoning", 0.0) > 0 and out.grad_A_from_self_decode_norm > 0,
    }


def generation_files(model, tokenizer, rows, first_pass, args, out_dir: Path) -> None:
    rows = rows[:args.generation_samples]
    normal, shuffle, zero = [], [], []
    for batch_start in range(0, len(rows), 2):
        batch = rows[batch_start:batch_start + 2]
        first = first_pass(model, batch)
        z = first.latents["reasoning"]
        z_shuffle = torch.roll(z, shifts=1, dims=0) if z.size(0) > 1 else torch.zeros_like(z)
        z_zero = torch.zeros_like(z)
        for i, rec in enumerate(batch):
            base_row = {"question": rec["question"], "gold_answer": rec["answer"]}
            normal.append(base_row | {"condition": "correct", "generated_reasoning": self_decode_generate(model, tokenizer, rec, "reasoning", z[i:i+1], args, args.max_generation_tokens), "generated_answer": ""})
            shuffle.append(base_row | {"condition": "shuffle", "generated_reasoning": self_decode_generate(model, tokenizer, rec, "reasoning", z_shuffle[i:i+1], args, args.max_generation_tokens), "generated_answer": ""})
            zero.append(base_row | {"condition": "zero", "generated_reasoning": self_decode_generate(model, tokenizer, rec, "reasoning", z_zero[i:i+1], args, args.max_generation_tokens), "generated_answer": ""})
    write_jsonl(out_dir / "normal_generation.jsonl", normal)
    write_jsonl(out_dir / "shuffle_generation.jsonl", shuffle)
    write_jsonl(out_dir / "zero_generation.jsonl", zero)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--dataset-path", type=Path, default=Path("/data/zxl/runs/model_a_only_loss1_formal/formal_split"))
    p.add_argument("--image-root", type=Path, default=Path("/data/zxl/runs/model_a_only_loss1_formal/image_files"))
    p.add_argument("--model-a-path", default="/data/zxl/small_models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--eval-samples", type=int, default=512)
    p.add_argument("--generation-samples", type=int, default=64)
    p.add_argument("--max-generation-tokens", type=int, default=96)
    p.add_argument("--answer-max-new-tokens", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda-self", type=float, default=0.1)
    p.add_argument("--lr-a", type=float, default=1e-5)
    p.add_argument("--optimizer", choices=["adafactor", "adamw"], default="adafactor")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-q", type=int, default=160)
    p.add_argument("--max-target", type=int, default=160)
    p.add_argument("--max-image-side", type=int, default=336)
    p.add_argument("--torch-dtype", default="bfloat16")
    ns = p.parse_args()
    args = SimpleNamespace(**vars(ns))
    args.model_a_path = args.model_a_path
    args.sections = SECTIONS
    args.self_decode_with_image = False
    args.batch_size = 1

    base.set_seed(args.seed)
    eval_rows = read_jsonl(args.dataset_path / "validation.jsonl")[:args.eval_samples]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, tokenizer, model = base.load_vlm_a(args, device)
    ckpt_report = load_model_checkpoint(model, args.checkpoint)
    first_pass = make_first_pass_fn(processor, tokenizer, args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    main_metrics = main_nll_and_answer(model, processor, tokenizer, eval_rows, args, args.answer_max_new_tokens)
    write_jsonl(args.out_dir / "answer_generation.jsonl", main_metrics.pop("answer_rows"))
    loss_metrics = loss1_and_intervention(model, tokenizer, eval_rows, first_pass, args)
    grad = gradient_audit(model, tokenizer, first_pass, eval_rows[0], args)
    generation_files(model, tokenizer, eval_rows, first_pass, args, args.out_dir)

    metrics = {
        "step": args.step,
        "checkpoint": ckpt_report,
        "eval_samples": len(eval_rows),
        "generation_samples": min(args.generation_samples, len(eval_rows)),
        "main": main_metrics,
        "loss1_reconstruction": loss_metrics["loss1_reconstruction"],
        "bootstrap_shuffle_margin": loss_metrics["bootstrap"],
        "gradient_audit": grad,
    }
    write_json(args.out_dir / "metrics.json", metrics)
    write_json(args.out_dir / f"gradient_step{args.step}.json", grad)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True, default=str))

if __name__ == "__main__":
    main()
