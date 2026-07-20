#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import Adafactor

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_data_small_vlm_official_sections import (  # noqa: E402
    SECTIONS,
    THINKING_TOKENS,
    encoder_forward,
    image_path,
    load_vlm_a,
    read_jsonl,
)
from src.htext.self_decoder import (  # noqa: E402
    SelfDecodeLatentInterface,
    build_self_decoder_batch,
    forward_model_a_text_only_self_decoder,
    prepare_self_decoder_latent,
)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


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


def model_dim(model) -> int:
    cfg = model.config
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return int(cfg.text_config.hidden_size)
    raise AttributeError("cannot infer hidden size")


def grad_norm(params) -> tuple[float, bool]:
    total, finite = 0.0, True
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        finite = finite and bool(torch.isfinite(g).all().item())
        total += float(g.float().pow(2).sum().item())
    return total**0.5, finite


def make_optimizer(params, args):
    if args.optimizer == "adafactor":
        return Adafactor(params, scale_parameter=False, relative_step=False, warmup_init=False)
    return torch.optim.AdamW(params, weight_decay=args.weight_decay)


def make_interface(model_a, tokenizer, args):
    token_ids = {stage: tokenizer.convert_tokens_to_ids(THINKING_TOKENS[stage]) for stage in SECTIONS}
    embed_weight = model_a.get_input_embeddings().weight
    return SelfDecodeLatentInterface(
        model_dim(model_a),
        SECTIONS,
        adapter_type=args.self_decode_adapter,
        role_mode=args.self_decode_role_embedding,
        token_embedding_weight=embed_weight,
        stage_token_ids=token_ids,
    ).to(device=embed_weight.device, dtype=embed_weight.dtype)


def self_loss_for_batch(model_a, tokenizer, interface, records, z, args):
    losses = {}
    audits = {}
    total = torch.zeros((), device=next(model_a.parameters()).device)
    if args.self_decode_forward_mode == "sequential":
        for stage in SECTIONS:
            z_stage = prepare_self_decoder_latent(z[stage], detach_latent=args.self_decode_detach_latent)
            injected, audit = interface(stage, z_stage)
            batch = build_self_decoder_batch(
                tokenizer=tokenizer,
                records=records,
                stage=stage,
                thinking_token=THINKING_TOKENS[stage],
                target_key=stage,
                label_mode=args.self_decode_label_mode,
                device=next(model_a.parameters()).device,
                max_target_tokens=args.max_target,
            )
            out = forward_model_a_text_only_self_decoder(
                model_a=model_a,
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                replacement_mask=batch.replacement_mask,
                injected_latent=injected,
            )
            total = total + out.loss
            losses[stage] = float(out.loss.detach().item())
            audits[stage] = {**audit, **out.audit}
        return total, losses, audits
    if args.self_decode_forward_mode == "batched":
        for stage in SECTIONS:
            z_stage = prepare_self_decoder_latent(z[stage], detach_latent=args.self_decode_detach_latent)
            injected, audit = interface(stage, z_stage)
            batch = build_self_decoder_batch(
                tokenizer=tokenizer,
                records=records,
                stage=stage,
                thinking_token=THINKING_TOKENS[stage],
                target_key=stage,
                label_mode=args.self_decode_label_mode,
                device=next(model_a.parameters()).device,
                max_target_tokens=args.max_target,
            )
            out = forward_model_a_text_only_self_decoder(
                model_a=model_a,
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                replacement_mask=batch.replacement_mask,
                injected_latent=injected,
            )
            total = total + out.loss
            losses[stage] = float(out.loss.detach().item())
            audits[stage] = {**audit, **out.audit}
        return total, losses, audits
    raise ValueError(args.self_decode_forward_mode)


def attribution(model_a, tokenizer, interface, records, args):
    model_a.zero_grad(set_to_none=True)
    interface.zero_grad(set_to_none=True)
    main, _logits, _labels, _z, _trace = encoder_forward(model_a, args.processor, tokenizer, records, args)
    main.backward()
    grad_main, finite_main = grad_norm(model_a.parameters())

    model_a.zero_grad(set_to_none=True)
    interface.zero_grad(set_to_none=True)
    _main, _logits, _labels, z, _trace = encoder_forward(model_a, args.processor, tokenizer, records, args)
    self_loss, _per, _audit = self_loss_for_batch(model_a, tokenizer, interface, records, z, args)
    self_loss.backward()
    grad_self, finite_self = grad_norm(model_a.parameters())
    grad_adapter, finite_adapter = grad_norm(interface.parameters())
    if args.self_decode_detach_latent and not (grad_self > 0):
        raise RuntimeError("detach path should still give decoder-side Model A gradients")
    if (not args.self_decode_detach_latent) and not (grad_self > 0 and finite_self):
        raise RuntimeError("no-detach self loss produced no finite Model A gradient")
    return {
        "grad_norm_A_from_main": grad_main,
        "grad_norm_A_from_self": grad_self,
        "grad_norm_adapter": grad_adapter,
        "finite_main": finite_main,
        "finite_self": finite_self,
        "finite_adapter": finite_adapter,
    }


@torch.no_grad()
def intervention_eval(model_a, tokenizer, interface, rows, args):
    model_a.eval()
    interface.eval()
    records = rows[: args.eval_samples]
    _main, _logits, _labels, z, _trace = encoder_forward(model_a, args.processor, tokenizer, records, args)
    out = {"sections": {}}
    for stage in SECTIONS:
        normal_z = z[stage]
        shuffle_z = torch.roll(normal_z, shifts=1, dims=0) if normal_z.shape[0] > 1 else torch.zeros_like(normal_z)
        zero_z = torch.zeros_like(normal_z)
        rand = torch.randn_like(normal_z)
        rand = rand / rand.float().norm(dim=-1, keepdim=True).clamp_min(1e-6) * normal_z.float().norm(dim=-1, keepdim=True).to(rand.dtype)
        vals = {}
        for name, value in [("normal", normal_z), ("shuffle", shuffle_z), ("zero", zero_z), ("random", rand)]:
            injected, _audit = interface(stage, value)
            batch = build_self_decoder_batch(
                tokenizer=tokenizer,
                records=records,
                stage=stage,
                thinking_token=THINKING_TOKENS[stage],
                target_key=stage,
                label_mode=args.self_decode_label_mode,
                device=next(model_a.parameters()).device,
                max_target_tokens=args.max_target,
            )
            pred = forward_model_a_text_only_self_decoder(
                model_a=model_a,
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                replacement_mask=batch.replacement_mask,
                injected_latent=injected,
            )
            vals[f"nll_{name}"] = float(pred.loss.item())
        vals["shuffle_margin"] = vals["nll_shuffle"] - vals["nll_normal"]
        vals["zero_margin"] = vals["nll_zero"] - vals["nll_normal"]
        vals["random_margin"] = vals["nll_random"] - vals["nll_normal"]
        out["sections"][stage] = vals
    return out


def qwen_text_only_parity(model_a, tokenizer, args):
    model_a.eval()
    text = "Question:\nWhat is 2 plus 2?\n<THINKING_OF_REASONING>\nAnswer: 4"
    enc = tokenizer([text], padding=True, return_tensors="pt").to(next(model_a.parameters()).device)
    labels = enc["input_ids"].clone()
    labels[:, :2] = -100
    with torch.no_grad():
        input_path = model_a(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels, output_hidden_states=True, use_cache=False, return_dict=True)
        embeds = model_a.get_input_embeddings()(enc["input_ids"])
        embed_path = model_a(inputs_embeds=embeds, attention_mask=enc["attention_mask"], labels=labels, pixel_values=None, image_grid_thw=None, use_cache=False, output_hidden_states=True, return_dict=True)
    return {
        "logits_shape_input_ids": list(input_path.logits.shape),
        "logits_shape_inputs_embeds": list(embed_path.logits.shape),
        "max_abs_diff": float((input_path.logits.float() - embed_path.logits.float()).abs().max().item()),
        "loss_input_ids": float(input_path.loss.item()),
        "loss_inputs_embeds": float(embed_path.loss.item()),
        "loss_abs_diff": float(abs(input_path.loss.item() - embed_path.loss.item())),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-a-path", default="/data/zxl/small_models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--subset", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_available_images_v1"))
    p.add_argument("--image-root", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1/image_files"))
    p.add_argument("--out", type=Path, default=Path("/data/zxl/runs/heima_small_vlm_self_a_smoke"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-samples", type=int, default=2)
    p.add_argument("--lr-a", type=float, default=1e-6)
    p.add_argument("--lr-adapter", type=float, default=1e-5)
    p.add_argument("--lambda-self", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["adafactor", "adamw"], default="adafactor")
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--max-q", type=int, default=160)
    p.add_argument("--max-target", type=int, default=128)
    p.add_argument("--max-image-side", type=int, default=336)
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--self-decode-detach-latent", action="store_true")
    p.add_argument("--self-decode-label-mode", choices=["text_only", "latent_and_text"], default="text_only")
    p.add_argument("--self-decode-forward-mode", choices=["sequential", "batched"], default="sequential")
    p.add_argument("--self-decode-adapter", choices=["identity", "ln_linear"], default="ln_linear")
    p.add_argument("--self-decode-role-embedding", choices=["none", "typed"], default="typed")
    args = p.parse_args()

    set_seed(args.seed)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.out / run_id
    train = read_jsonl(args.subset / "train.jsonl")
    val = read_jsonl(args.subset / "validation.jsonl")
    missing = [str(image_path(args, r)) for r in train[: args.steps] + val[: args.eval_samples] if not image_path(args, r).exists()]
    if missing:
        raise FileNotFoundError(f"missing images: {missing[:5]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, tokenizer, model_a = load_vlm_a(args, device)
    args.processor = processor
    interface = make_interface(model_a, tokenizer, args)
    params = [{"params": model_a.parameters(), "lr": args.lr_a}]
    if list(interface.parameters()):
        params.append({"params": interface.parameters(), "lr": args.lr_adapter})
    opt = make_optimizer(params, args)

    manifest = {
        "status": "running",
        "decoder_arch": "self_a",
        "no_model_b": True,
        "same_model_a_object": True,
        "model_a_path": args.model_a_path,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "host": shell(["hostname"]),
        "gpu": shell(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader"]),
        "args": {k: v for k, v in vars(args).items() if k != "processor"},
        "self_decoder_call": "model_a(inputs_embeds=..., attention_mask=..., labels=..., pixel_values=None, image_grid_thw=None)",
    }
    write_json(run_dir / "experiment_manifest.json", manifest)
    write_json(run_dir / "qwen_text_only_parity.json", qwen_text_only_parity(model_a, tokenizer, args))
    attr = attribution(model_a, tokenizer, interface, batch_rows(train, args.batch_size, 0), args)

    logs = []
    started = time.time()
    peak = 0
    for step in range(1, args.steps + 1):
        model_a.train()
        interface.train()
        batch = batch_rows(train, args.batch_size, step - 1)
        opt.zero_grad(set_to_none=True)
        main_loss, _logits, _labels, z, _trace = encoder_forward(model_a, processor, tokenizer, batch, args)
        self_loss, per_stage, audits = self_loss_for_batch(model_a, tokenizer, interface, batch, z, args)
        total = main_loss + args.lambda_self * self_loss
        total.backward()
        grad_a, finite_a = grad_norm(model_a.parameters())
        grad_interface, finite_interface = grad_norm(interface.parameters())
        if not (torch.isfinite(total).item() and finite_a and finite_interface):
            raise RuntimeError("non-finite self-A smoke loss/gradient")
        torch.nn.utils.clip_grad_norm_([p for group in params for p in group["params"]], args.clip_grad)
        opt.step()
        if torch.cuda.is_available():
            peak = max(peak, int(torch.cuda.max_memory_allocated()))
        logs.append(
            {
                "step": step,
                "loss_main": float(main_loss.detach().item()),
                "loss_self_total": float(self_loss.detach().item()),
                "loss_self_by_stage": per_stage,
                "loss_total": float(total.detach().item()),
                "grad_norm_A": grad_a,
                "grad_norm_adapter": grad_interface,
                "audit": audits,
            }
        )

    eval_metrics = intervention_eval(model_a, tokenizer, interface, val, args)
    summary = {
        "status": "complete",
        "runtime_sec": time.time() - started,
        "gradient_attribution": attr,
        "logs": logs,
        "intervention_eval": eval_metrics,
        "peak_cuda_memory_bytes": peak,
    }
    write_json(run_dir / "summary.json", summary)
    manifest["status"] = "complete"
    write_json(run_dir / "experiment_manifest.json", manifest)
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
