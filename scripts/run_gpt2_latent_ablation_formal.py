#!/usr/bin/env python3
"""Formal GPT2-small G2/G3 latent-token ablation training.

Runs G2 and G3 on the same deterministic split, optimizer, and step schedule.
Only model weights/config/metrics are saved; optimizer state is intentionally not
checkpointed.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

from run_gpt2_latent_ablation import (
    Example,
    build_split,
    compute_losses,
    evaluate_intervention,
    exact_answer_accuracy,
    generate_condition,
    load_hf,
    token_overlap_summary,
    trainable_parameter_report,
    write_jsonl,
)

CHECKPOINT_STEPS = (500, 1000, 2500, 5000)


def load_split(path: Path) -> Dict[str, List[Example]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "train": [Example(**x) for x in raw["train"]],
        "eval": [Example(**x) for x in raw["eval"]],
    }


def batched(records: Sequence[Example], batch_size: int) -> Iterable[List[Example]]:
    for i in range(0, len(records), batch_size):
        yield list(records[i : i + batch_size])


def mean_metrics(items: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({k for item in items for k in item})
    out = {}
    for key in keys:
        vals = [item[key] for item in items if isinstance(item.get(key), (int, float))]
        if vals:
            out[key] = sum(vals) / len(vals)
        else:
            out[key] = None
    return out


def eval_batched(model, tokenizer, records: Sequence[Example], group: str, max_length: int, eval_batch_size: int, device: torch.device) -> Dict[str, float]:
    metrics = []
    for batch in batched(records, eval_batch_size):
        metrics.append(evaluate_intervention(model, tokenizer, batch, group, max_length, device))
    return mean_metrics(metrics)


def generation_eval(model, tokenizer, records: Sequence[Example], group_dir: Path, step: int, max_length: int, device: torch.device) -> Dict[str, float]:
    gen_dir = group_dir / f"eval_step{step}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    normal = generate_condition(model, tokenizer, records, max_length, device, "normal")
    shuffle = generate_condition(model, tokenizer, records, max_length, device, "shuffle")
    zero = generate_condition(model, tokenizer, records, max_length, device, "zero")
    write_jsonl(gen_dir / "normal_generation.jsonl", normal)
    write_jsonl(gen_dir / "shuffle_generation.jsonl", shuffle)
    write_jsonl(gen_dir / "zero_generation.jsonl", zero)
    return {
        "generation_acc": exact_answer_accuracy([x["generated_text"] for x in normal], records),
        "generation_token_overlap_normal_shuffle": token_overlap_summary(normal, shuffle),
        "generation_samples": len(records),
    }


def save_weights(model, tokenizer, group_dir: Path, step: int, group: str, args: argparse.Namespace) -> str:
    ckpt_dir = group_dir / f"step{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "group": group, "step": step, "tokenizer_len": len(tokenizer)}, ckpt_dir / "model_weights.pt")
    config = {
        "group": group,
        "step": step,
        "model_path": args.model_path,
        "lambda_self": args.lambda_self,
        "lambda_latent": args.lambda_latent,
        "seed": args.seed,
        "optimizer": "AdamW",
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "optimizer_checkpoint_saved": False,
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return str(ckpt_dir)


def train_group(group: str, split: Dict[str, List[Example]], args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    group_dir = Path(args.output_dir) / group
    group_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_hf(args.model_path, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train = split["train"]
    eval_records = split["eval"]
    gen_records = eval_records[: args.generation_samples]
    progress = []
    final_summary: Dict[str, object] = {}
    start = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        offset = ((step - 1) * args.batch_size) % len(train)
        batch = train[offset : offset + args.batch_size]
        if len(batch) < args.batch_size:
            batch = batch + train[: args.batch_size - len(batch)]
        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses(model, tokenizer, batch, group, args.lambda_self, args.lambda_latent, args.max_length, device)
        losses.total_loss.backward()
        optimizer.step()

        if step % args.log_every == 0:
            row = {
                "step": step,
                "loss_total": float(losses.total_loss.detach().cpu()),
                "main_loss": float(losses.main_loss.detach().cpu()),
                "self_decode_loss": float(losses.self_decode_loss.detach().cpu()),
                "latent_token_loss": float(losses.latent_token_loss.detach().cpu()),
                "latent_token_accuracy": float(losses.latent_token_accuracy.detach().cpu()),
                "elapsed_sec": time.time() - start,
            }
            progress.append(row)
            with open(group_dir / "train_progress.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({"group": group, **row}), flush=True)

        if step in CHECKPOINT_STEPS:
            ckpt = save_weights(model, tokenizer, group_dir, step, group, args)
            eval_metrics = eval_batched(model, tokenizer, eval_records, group, args.max_length, args.eval_batch_size, device)
            gen_metrics = generation_eval(model, tokenizer, gen_records, group_dir, step, args.max_length, device)
            eval_metrics.update(gen_metrics)
            eval_metrics["checkpoint_path"] = ckpt
            eval_metrics["step"] = step
            (group_dir / f"eval_step{step}.json").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")
            final_summary[f"step{step}"] = eval_metrics
            print(json.dumps({"group": group, "eval": eval_metrics}), flush=True)
            trainable_parameter_report(model, group_dir / f"trainable_parameter_report_step{step}.json")

    (group_dir / "summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    return final_summary


def compact_output(all_summary: Dict[str, Dict[str, object]], step: int) -> Dict[str, Dict[str, float]]:
    out = {}
    for group in ("G2", "G3"):
        metrics = all_summary.get(group, {}).get(f"step{step}", {})
        out[group] = {
            "shuffle_margin": metrics.get("shuffle_margin"),
            "latent_acc": metrics.get("latent_token_accuracy"),
            "generation_acc": metrics.get("generation_acc"),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/data/zxl/official_heima/datasets/LLaVA-CoT-100k/train.jsonl")
    parser.add_argument("--model-path", default="/data/zxl/models/openai-community-gpt2")
    parser.add_argument("--output-dir", default="/data/zxl/runs/gpt2_latent_token_ablation_formal")
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--generation-samples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--lambda-self", type=float, default=0.1)
    parser.add_argument("--lambda-latent", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--groups", default="G2,G3")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    if "gpt2-xl" in args.model_path.lower():
        raise SystemExit("GPT2-xl is forbidden for this experiment")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = out_dir / "data_split.json"
    if not split_path.exists():
        split_info = build_split(args.dataset, out_dir, args.train_samples, args.eval_samples, args.seed)
    else:
        split_info = {"split_path": str(split_path), "reused": True}
    (out_dir / "formal_config.json").write_text(json.dumps(vars(args) | {"split_info": split_info, "checkpoint_steps": list(CHECKPOINT_STEPS)}, indent=2), encoding="utf-8")
    split = load_split(split_path)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    summaries = {}
    for group in [g.strip() for g in args.groups.split(",") if g.strip()]:
        if group not in {"G2", "G3"}:
            raise SystemExit(f"Formal run only supports G2/G3, got {group}")
        summaries[group] = train_group(group, split, args, device)
    compact = compact_output(summaries, args.steps)
    (out_dir / "formal_comparison.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    main()
