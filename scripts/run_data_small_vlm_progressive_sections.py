#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import Adafactor, AutoModelForImageTextToText, AutoProcessor

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.htext.formal_eval import hidden_geometry
from src.htext.heima_reuse import backend_resolution_snapshot, extract_thinking_state, heima_ce_loss

SECTIONS = ("summary", "caption", "reasoning")
TOKENS = {
    "summary": "<THINKING_OF_SUMMARY>",
    "caption": "<THINKING_OF_CAPTION>",
    "reasoning": "<THINKING_OF_REASONING>",
}
STAGES = ("explicit", "p1_summary", "p2_summary_caption", "recover_all")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def shell(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_rows(rows: list[dict], batch_size: int, step: int) -> list[dict]:
    start = (step * batch_size) % len(rows)
    return (rows + rows)[start : start + batch_size]


def image_path(args, rec: dict) -> Path:
    p = Path(rec["image"])
    return p if p.is_absolute() else args.image_root / p


def load_image(path: Path, max_side: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def load_model(args, device: torch.device):
    dtype = getattr(torch, args.torch_dtype) if args.torch_dtype else None
    processor = AutoProcessor.from_pretrained(args.model_a_path, local_files_only=True, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": list(TOKENS.values())})
    kwargs = {"local_files_only": True, "trust_remote_code": True}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForImageTextToText.from_pretrained(args.model_a_path, **kwargs)
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)
    return processor, tokenizer, model


def stage_parts(rec: dict, stage: str) -> list[str]:
    if stage == "explicit":
        return [rec["summary"], rec["caption"], rec["reasoning"]]
    if stage == "p1_summary":
        return [TOKENS["summary"], rec["caption"], rec["reasoning"]]
    if stage == "p2_summary_caption":
        return [TOKENS["summary"], TOKENS["caption"], rec["reasoning"]]
    if stage == "recover_all":
        return [TOKENS["summary"], TOKENS["caption"], TOKENS["reasoning"]]
    raise ValueError(stage)


def sample_text(rec: dict, stage: str, include_answer: bool) -> str:
    text = "Question:\n" + rec["question"] + "\n"
    for section, part in zip(SECTIONS, stage_parts(rec, stage)):
        text += f"{section.upper()}:\n{part}\n"
    text += "Answer:"
    if include_answer:
        text += " " + rec["answer"]
    return text


def vlm_inputs(processor, args, records: list[dict], stage: str, include_answer: bool):
    texts, images = [], []
    for rec in records:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": sample_text(rec, stage, include_answer)},
            ],
        }]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        images.append(load_image(image_path(args, rec), args.max_image_side))
    return processor(text=texts, images=images, padding=True, return_tensors="pt")


def forward(model, processor, tokenizer, records: list[dict], stage: str, args):
    device = next(model.parameters()).device
    full = vlm_inputs(processor, args, records, stage, True).to(device)
    prefix = vlm_inputs(processor, args, records, stage, False)
    labels = torch.full_like(full["input_ids"], -100)
    for i in range(len(records)):
        prefix_len = int(prefix["attention_mask"][i].sum().item())
        full_len = int(full["attention_mask"][i].sum().item())
        # Supervise all stage outputs and answer, but not image/question/chat prefix.
        label_start = max(0, prefix_len - sum(part in TOKENS.values() for part in stage_parts(records[i], stage)) - 1)
        labels[i, label_start:full_len] = full["input_ids"][i, label_start:full_len]
    out = model(**full, output_hidden_states=True, use_cache=False)
    loss = heima_ce_loss(out.logits, labels)
    z, trace = {}, {}
    for section in SECTIONS:
        if TOKENS[section] not in stage_parts(records[0], stage):
            continue
        tid = tokenizer.convert_tokens_to_ids(TOKENS[section])
        state = extract_thinking_state(
            input_ids=full["input_ids"],
            last_hidden_state=out.hidden_states[-1],
            thinking_token_id=tid,
            mode="predictor",
        )
        z[section] = state.hidden
        trace[section] = {
            "thinking_pos": state.thinking_positions.detach().cpu().tolist(),
            "selected_pos": state.selected_positions.detach().cpu().tolist(),
            "semantics": state.semantics,
        }
    return loss, z, trace


def grad_norm(params) -> tuple[float, bool]:
    total, finite = 0.0, True
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        finite = finite and bool(torch.isfinite(g).all().item())
        total += float(g.float().pow(2).sum().item())
    return total**0.5, finite


@torch.no_grad()
def evaluate(model, processor, tokenizer, rows: list[dict], stage: str, args):
    model.eval()
    total = 0.0
    zs = {s: [] for s in SECTIONS}
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        loss, z, _trace = forward(model, processor, tokenizer, batch, stage, args)
        total += float(loss.item()) * len(batch)
        for s, value in z.items():
            zs[s].append(value.detach().cpu())
    return {
        "main_nll": total / len(rows),
        "latent_geometry": {s: hidden_geometry(torch.cat(chunks, dim=0)) for s, chunks in zs.items() if chunks},
    }


def make_optimizer(params, args):
    if args.optimizer == "adafactor":
        return Adafactor([{"params": params, "lr": args.lr_a, "weight_decay": args.weight_decay}], scale_parameter=False, relative_step=False, warmup_init=False)
    return torch.optim.AdamW(params, lr=args.lr_a, weight_decay=args.weight_decay)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-a-path", default="/data/zxl/small_models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--subset", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_available_images_v1"))
    p.add_argument("--image-root", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1/image_files"))
    p.add_argument("--out", type=Path, default=Path("/data/zxl/runs/heima_small_vlm_progressive_sections"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps-per-stage", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-samples", type=int, default=8)
    p.add_argument("--lr-a", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["adafactor", "adamw"], default="adafactor")
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--max-image-side", type=int, default=336)
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--log-every", type=int, default=5)
    args = p.parse_args()

    set_seed(args.seed)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.out / f"seed{args.seed}" / run_id
    train = read_jsonl(args.subset / "train.jsonl")
    val = read_jsonl(args.subset / "validation.jsonl")
    test = read_jsonl(args.subset / "test.jsonl")
    missing = [str(image_path(args, r)) for r in train + val + test if not image_path(args, r).exists()]
    if missing:
        raise FileNotFoundError(f"missing images: {missing[:5]} count={len(missing)}")
    manifest = {
        "status": "running",
        "task": "MLLM progressive/recovering official-section A-only schedule",
        "run_id": run_id,
        "host": shell(["hostname"]),
        "gpu": shell(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader"]),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_a": {"type": "VLM", "path": args.model_a_path, "sees": ["image", "question"]},
        "stages": STAGES,
        "typed_tokens": TOKENS,
        "args": vars(args),
    }
    write_json(run_dir / "experiment_manifest.json", manifest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, tokenizer, model = load_model(args, device)
    opt = make_optimizer(model.parameters(), args)
    stage_results = {}
    started = time.time()
    for stage in STAGES:
        logs, first_trace = [], None
        initial = evaluate(model, processor, tokenizer, val[: args.eval_samples], stage, args)
        for step in range(1, args.steps_per_stage + 1):
            batch = batch_rows(train, args.batch_size, step - 1)
            opt.zero_grad(set_to_none=True)
            loss, _z, trace = forward(model, processor, tokenizer, batch, stage, args)
            if first_trace is None:
                first_trace = trace
            loss.backward()
            g, finite = grad_norm(model.parameters())
            if not finite:
                raise RuntimeError(f"non-finite grad at {stage}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            if step == 1 or step == args.steps_per_stage or step % args.log_every == 0:
                logs.append({"step": step, "main": float(loss.item()), "grad_A": g})
        final_val = evaluate(model, processor, tokenizer, val[: args.eval_samples], stage, args)
        stage_results[stage] = {"initial_validation": initial, "final_validation": final_val, "logs": logs, "position_trace_first_batch": first_trace}
        write_json(run_dir / "stages" / stage / "result.json", stage_results[stage])
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save({"model_a": model.state_dict(), "stage": stage, "seed": args.seed}, run_dir / "checkpoints" / f"{stage}.pt")
    final_test = evaluate(model, processor, tokenizer, test[: args.eval_samples], "recover_all", args)
    summary = {
        "status": "complete",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": time.time() - started,
        "stage_results": stage_results,
        "recover_test": final_test,
        "backend_resolution": backend_resolution_snapshot(),
    }
    write_json(run_dir / "summary.json", summary)
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = summary["completed_at_utc"]
    write_json(run_dir / "experiment_manifest.json", manifest)
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
