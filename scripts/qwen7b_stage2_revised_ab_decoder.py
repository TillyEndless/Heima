#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

import qwen7b_stage1_r64_lr5_gate_sweep as base

LR = base.LR
SEED = base.SEED
LAMBDA_DECODE = 1.0
DEFAULT_STAGE1_50K = "/data2/zhouxiaoling/latent_cot/runs/QWEN7B_90K_NOCAP_THINKSTART_LALL1_R64_LR5e-5_STAGE1_GATE_5K_10K_25K_50K_90K_NOSTOP_20260811/checkpoints/stage1_step50000"
DEFAULT_RUN = "/data2/zhouxiaoling/latent_cot/runs/QWEN7B_90K_NOCAP_THINKSTART_LALL1_R64_LR5e-5_REVISED_AB_DECODER_FROM_S1_50K_S2_25K_20260813"


def now() -> float:
    return time.time()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo), text=True).strip()
    except Exception:
        return "unknown"


def tokenizer_for_ckpt(stage1_ckpt: Path) -> Path:
    return stage1_ckpt.parent / "tokenizer"


def load_a_b(stage1_ckpt: Path, trainable: bool = True):
    tok_path = tokenizer_for_ckpt(stage1_ckpt)
    adapter = stage1_ckpt / "adapter"
    projector = stage1_ckpt / "projector.pt"
    if not adapter.exists():
        raise FileNotFoundError(adapter)
    if not tok_path.exists():
        raise FileNotFoundError(tok_path)
    tok, model_a, projector_a = base.load_model_and_projector(trainable, adapter_path=adapter, tokenizer_path=tok_path, projector_path=projector)
    _, model_b, _ = base.load_model_and_projector(trainable, adapter_path=adapter, tokenizer_path=tok_path, projector_path=projector)
    return tok, model_a, model_b, projector_a


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def grad_norm(parameters) -> float:
    sq = 0.0
    for p in parameters:
        if p.grad is not None:
            n = float(p.grad.detach().float().norm().cpu())
            sq += n * n
    return math.sqrt(sq)


def lm_head_grad_norm(model) -> float:
    mod = model.get_output_embeddings()
    return grad_norm(list(mod.parameters())) if mod is not None else 0.0


def stage1_total_loss(c: dict) -> torch.Tensor:
    return c["answer_loss"] + c["start_loss"] + c["sync_loss"] + c["end_loss"] + c["marker_loss"]


def main_raw_loss_sum(c: dict) -> float:
    pairs = [
        ("answer_loss", "answer_active_token_count"),
        ("start_loss", "start_active_token_count"),
        ("sync_loss", "sync_active_token_count"),
        ("end_loss", "end_active_token_count"),
        ("marker_loss", "marker_active_token_count"),
    ]
    total = 0.0
    for loss_key, count_key in pairs:
        total += float(c[loss_key].detach().cpu()) * int(c[count_key])
    return total


def component_record(c: dict) -> dict:
    d = base.component_record(c)
    d["loss_main"] = float(stage1_total_loss(c).detach().cpu())
    d["num_supervised_tokens_main"] = int(c["answer_active_token_count"] + c["start_active_token_count"] + c["sync_active_token_count"] + c["end_active_token_count"] + c["marker_active_token_count"])
    d["raw_loss_sum_main"] = main_raw_loss_sum(c)
    d["mean_loss_main_if_token_mean"] = d["raw_loss_sum_main"] / max(d["num_supervised_tokens_main"], 1)
    if c.get("decode") is not None:
        d["loss_decode"] = float(c["decode"]["loss"].detach().cpu())
        d["decode_token_acc"] = c["decode"]["token_acc"]
        d["num_supervised_tokens_decode"] = int(c["decode"]["active_count"])
        d["raw_loss_sum_decode"] = d["loss_decode"] * max(d["num_supervised_tokens_decode"], 0)
        d["mean_loss_decode"] = d["loss_decode"]
    return d


def final_hidden_norm_module(model):
    candidates = [
        ["base_model", "model", "model", "norm"],
        ["base_model", "model", "model", "model", "norm"],
        ["model", "model", "norm"],
        ["model", "norm"],
    ]
    for path in candidates:
        obj = model
        ok = True
        for name in path:
            if not hasattr(obj, name):
                ok = False
                break
            obj = getattr(obj, name)
        if ok:
            return obj, ".".join(path)
    raise AttributeError(f"Could not locate final norm module on {type(model)}")


def forward_with_final_hidden(model, input_ids, attention_mask):
    captured = {}
    norm, norm_path = final_hidden_norm_module(model)
    def hook(_module, _inputs, output):
        captured["hidden"] = output[0] if isinstance(output, tuple) else output
    handle = norm.register_forward_hook(hook)
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError(f"final hidden hook did not fire at {norm_path}")
    return out, captured["hidden"], norm_path


def ce_by_positions_chunked(logits: torch.Tensor, input_ids: torch.Tensor, active_positions: list[int], chunk_size: int = 16) -> tuple[torch.Tensor, float, int]:
    positions = [p for p in active_positions if p > 0]
    if not positions:
        z = logits.sum() * 0.0
        return z, float("nan"), 0
    total_loss = logits.sum() * 0.0
    total_correct = 0.0
    total_count = 0
    for i in range(0, len(positions), chunk_size):
        chunk = positions[i:i + chunk_size]
        pred_pos = torch.tensor([p - 1 for p in chunk], device=logits.device)
        labels = input_ids[0, chunk]
        selected = logits[0, pred_pos, :].float()
        losses = torch.nn.functional.cross_entropy(selected, labels, reduction="sum")
        total_loss = total_loss + losses
        with torch.no_grad():
            total_correct += float(selected.argmax(dim=-1).eq(labels).float().sum().cpu())
            total_count += int(labels.numel())
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1), total_count


def forward_revised(tok, model_a, model_b, row: dict, detach_z: bool = False):
    seq, meta = base.main_sequence(tok, row)
    input_ids = torch.tensor([seq], dtype=torch.long, device="cuda")
    attn = torch.ones_like(input_ids)
    out, hidden, hidden_hook_path = forward_with_final_hidden(model_a, input_ids, attn)

    start_loss, start_acc, start_count = base.ce_by_positions(out.logits, input_ids, [meta["start_position"]])
    sync_loss, sync_acc, sync_count = base.ce_by_positions(out.logits, input_ids, meta["think_positions"])
    end_loss, end_acc, end_count = base.ce_by_positions(out.logits, input_ids, [meta["end_position"]])
    marker_loss, marker_acc, marker_count = base.ce_by_positions(out.logits, input_ids, [meta["answer_marker_position"]])
    answer_loss, answer_acc, answer_count = base.ce_by_positions(out.logits, input_ids, meta["answer_positions"])
    boundary_loss, boundary_acc, boundary_count = base.ce_by_positions(out.logits, input_ids, meta["boundary_positions"])
    first_sync_loss, first_sync_acc, _ = base.ce_by_positions(out.logits, input_ids, meta["think_positions"][:1])
    cont_sync_loss, cont_sync_acc, cont_sync_count = base.ce_by_positions(out.logits, input_ids, meta["think_positions"][1:])

    z = hidden[:, meta["think_positions"], :]
    if z.requires_grad:
        z.retain_grad()
    z_for_b = z.detach() if detach_z else z
    if detach_z:
        z_for_b.requires_grad_(True)
        z_for_b.retain_grad()

    dseq, dmeta = base.decoder_sequence(tok, row, meta["k"])
    decoder_ids = torch.tensor([dseq], dtype=torch.long, device="cuda")
    decoder_attn = torch.ones_like(decoder_ids)
    embeds = model_b.get_input_embeddings()(decoder_ids)
    embeds = embeds.clone()
    embeds[:, dmeta["placeholder_positions"], :] = z_for_b.to(dtype=embeds.dtype)
    dout = model_b(inputs_embeds=embeds, attention_mask=decoder_attn, use_cache=False)
    decode_loss, decode_acc, decode_count = ce_by_positions_chunked(dout.logits, decoder_ids, dmeta["decode_label_positions"], chunk_size=8)

    return {
        "start_loss": start_loss,
        "sync_loss": sync_loss,
        "end_loss": end_loss,
        "marker_loss": marker_loss,
        "answer_loss": answer_loss,
        "boundary_loss": boundary_loss,
        "first_sync_loss": first_sync_loss,
        "continuation_sync_loss": cont_sync_loss,
        "start_acc": start_acc,
        "sync_acc": sync_acc,
        "end_acc": end_acc,
        "marker_acc": marker_acc,
        "answer_acc": answer_acc,
        "boundary_acc": boundary_acc,
        "first_sync_acc": first_sync_acc,
        "continuation_sync_acc": cont_sync_acc,
        "start_active_token_count": start_count,
        "sync_active_token_count": sync_count,
        "end_active_token_count": end_count,
        "marker_active_token_count": marker_count,
        "answer_active_token_count": answer_count,
        "boundary_active_token_count": boundary_count,
        "continuation_sync_active_token_count": cont_sync_count,
        "main_logits": out.logits,
        "main_input_ids": input_ids,
        "z": z,
        "z_for_b": z_for_b,
        "decode": {
            "loss": decode_loss,
            "token_acc": decode_acc,
            "active_count": decode_count,
            "decoder_seq_len": len(dseq),
            "placeholder_positions": dmeta["placeholder_positions"],
            "decode_label_positions": dmeta["decode_label_positions"],
            "cot_token_count": dmeta["cot_token_count"],
        },
        "main_seq_len": len(seq),
        "meta": meta,
        "hidden_hook_path": hidden_hook_path,
    }


def total_loss(c: dict) -> torch.Tensor:
    return stage1_total_loss(c) + LAMBDA_DECODE * c["decode"]["loss"]


def save_checkpoint(run: Path, model_a, model_b, tok, name: str, step: int) -> str:
    path = run / "checkpoints" / name
    path.mkdir(parents=True, exist_ok=True)
    model_a.save_pretrained(path / "adapter")
    model_b.save_pretrained(path / "model_b_adapter")
    tok.save_pretrained(run / "checkpoints" / "tokenizer")
    write_json(path / "checkpoint_meta.json", {
        "step": step,
        "adapter": str(path / "adapter"),
        "model_a_adapter": str(path / "adapter"),
        "model_b_adapter": str(path / "model_b_adapter"),
        "projector": None,
        "routing": "decode_loss: model_b textual LM path -> injected h_z -> model_a latent path",
    })
    return str(path)


def validation(tok, model_a, model_b, rows: list[dict], n: int = 64) -> dict:
    vals = []
    model_a.eval(); model_b.eval()
    with torch.no_grad():
        for r in rows[:n]:
            c = forward_revised(tok, model_a, model_b, r)
            vals.append({
                "answer": float(c["answer_loss"].detach().cpu()),
                "start": float(c["start_loss"].detach().cpu()),
                "sync": float(c["sync_loss"].detach().cpu()),
                "first_sync": float(c["first_sync_loss"].detach().cpu()),
                "continuation_sync": float(c["continuation_sync_loss"].detach().cpu()),
                "end": float(c["end_loss"].detach().cpu()),
                "answer_marker": float(c["marker_loss"].detach().cpu()),
                "boundary": float(c["boundary_loss"].detach().cpu()),
                "answer_acc": c["answer_acc"],
                "decode": float(c["decode"]["loss"].detach().cpu()),
                "decode_token_acc": c["decode"]["token_acc"],
            })
    model_a.train(); model_b.train()
    def mean(k):
        xs = [x[k] for x in vals if k in x and math.isfinite(x[k])]
        return sum(xs) / len(xs) if xs else None
    return {"n": len(vals), **{f"val_{k}": mean(k) for k in ["answer", "start", "sync", "first_sync", "continuation_sync", "end", "answer_marker", "boundary", "answer_acc", "decode", "decode_token_acc"]}}


def grad_sanity(args):
    run = Path(args.run_dir)
    audit_dir = run / "audit"
    train, _, data_meta = base.load_split(Path(args.manifest))
    stage1_ckpt = Path(args.stage1_ckpt)
    tok, model_a, model_b, _ = load_a_b(stage1_ckpt, trainable=True)
    row = train[0]
    a_params = trainable_params(model_a)
    b_params = trainable_params(model_b)

    report = {"stage1_checkpoint": str(stage1_ckpt), "data_meta": data_meta, "lambda_decode": LAMBDA_DECODE}

    # Check 1: decode only, no detach.
    model_a.zero_grad(set_to_none=True); model_b.zero_grad(set_to_none=True)
    c = forward_revised(tok, model_a, model_b, row, detach_z=False)
    c["decode"]["loss"].backward()
    report["check_decode_only"] = {
        "grad_norm_model_a": grad_norm(a_params),
        "grad_norm_model_a_lm_head": lm_head_grad_norm(model_a),
        "grad_norm_model_b": grad_norm(b_params),
        "grad_norm_h_z": float(c["z"].grad.detach().float().norm().cpu()) if c["z"].grad is not None else 0.0,
        "h_z_requires_grad": bool(c["z"].requires_grad),
        **component_record(c),
    }

    # Check 2: decode only, detached z.
    model_a.zero_grad(set_to_none=True); model_b.zero_grad(set_to_none=True)
    c_det = forward_revised(tok, model_a, model_b, row, detach_z=True)
    c_det["decode"]["loss"].backward()
    report["check_decode_detached_z"] = {
        "grad_norm_model_a": grad_norm(a_params),
        "grad_norm_model_a_lm_head": lm_head_grad_norm(model_a),
        "grad_norm_model_b": grad_norm(b_params),
        "grad_norm_h_z_original": float(c_det["z"].grad.detach().float().norm().cpu()) if c_det["z"].grad is not None else 0.0,
        "grad_norm_z_for_b_detached_leaf": float(c_det["z_for_b"].grad.detach().float().norm().cpu()) if c_det["z_for_b"].grad is not None else 0.0,
    }

    # Check 3: main only.
    model_a.zero_grad(set_to_none=True); model_b.zero_grad(set_to_none=True)
    c_main = forward_revised(tok, model_a, model_b, row, detach_z=False)
    stage1_total_loss(c_main).backward()
    report["check_main_only"] = {
        "grad_norm_model_a": grad_norm(a_params),
        "grad_norm_model_a_lm_head": lm_head_grad_norm(model_a),
        "grad_norm_model_b": grad_norm(b_params),
        "grad_norm_h_z": float(c_main["z"].grad.detach().float().norm().cpu()) if c_main["z"].grad is not None else 0.0,
        **component_record(c_main),
    }

    report["loss_normalization"] = {
        "main": "sum of independently mean-normalized CE components: L_answer + L_start + L_sync + L_end + L_marker",
        "decode": "mean CE over textual CoT/EOS target token positions",
        "total": "main component sum + lambda_decode * decode token mean",
    }
    write_json(audit_dir / "revised_gradient_routing_sanity.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def run_stage2(args):
    repo = Path(args.repo)
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    train, eval_rows, data_meta = base.load_split(Path(args.manifest))
    stage1_ckpt = Path(args.stage1_ckpt)
    tok, model_a, model_b, _ = load_a_b(stage1_ckpt, trainable=True)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    a_params = trainable_params(model_a)
    b_params = trainable_params(model_b)
    opt = torch.optim.AdamW(a_params + b_params, lr=LR)
    cfg = {
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=str(repo), text=True).strip(),
        "commit": git_sha(repo),
        "manifest": str(args.manifest),
        "stage1_checkpoint": str(stage1_ckpt),
        "seed": SEED,
        "lr": LR,
        "lambda_decode": LAMBDA_DECODE,
        "model_a_init": "Stage1 50K adapter",
        "model_b_init": "deepcopy/same Stage1 50K adapter, independent parameters",
        "routing": {
            "main_loss": "model A only: Q + THINK_START + K*THINK + THINK_END + ANSWER + gold answer",
            "decode_loss": "model B textual CoT LM path with model A h_z injected into B placeholders",
            "decode_to_A_interface": "only h_z last hidden states; z_detach False",
            "projector": "none; A hidden size equals B input embedding size",
        },
        "loss_normalization": {
            "main": "sum of independently mean-normalized CE components",
            "decode": "mean CE over textual CoT/EOS tokens",
            "total": "main + lambda_decode * decode",
        },
        "data_meta": data_meta,
    }
    write_json(run / "stage2_config.json", cfg)
    log = run / "stage2_train.jsonl"
    status = run / "status.json"
    checkpoint_steps = {0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000}
    save_checkpoint(run, model_a, model_b, tok, "stage2_step0", 0)
    append_jsonl(log, {"event": "stage2_checkpoint", "step": 0, "checkpoint": str(run / "checkpoints" / "stage2_step0")})
    t0 = now()
    for step in range(1, args.stage2_steps + 1):
        r = train[(step - 1) % len(train)]
        torch.cuda.reset_peak_memory_stats()
        s0 = now()
        model_a.zero_grad(set_to_none=True); model_b.zero_grad(set_to_none=True)
        c = forward_revised(tok, model_a, model_b, r, detach_z=False)
        loss = total_loss(c)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(a_params + b_params, 1.0)
        opt.step()
        rec = {
            "phase": "stage2_revised_ab_decoder",
            "step": step,
            "sample_id": r.get("sample_id"),
            "loss_total": float(loss.detach().cpu()),
            **component_record(c),
            "z_grad_norm_after_backward": float(c["z"].grad.detach().float().norm().cpu()) if c["z"].grad is not None else 0.0,
            "step_runtime": now() - s0,
            "elapsed": now() - t0,
            "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
        if step == 1 or step % 500 == 0:
            rec["grad_norm_A"] = grad_norm(a_params)
            rec["grad_norm_A_lm_head"] = lm_head_grad_norm(model_a)
            rec["grad_norm_B"] = grad_norm(b_params)
        append_jsonl(log, rec)
        if step % 100 == 0 or step == 1:
            write_json(status, {"status": "running", "phase": "stage2_revised_ab_decoder", "step": step, "last": rec, "elapsed": now() - t0})
        if step in checkpoint_steps:
            ckpt = save_checkpoint(run, model_a, model_b, tok, f"stage2_step{step}", step)
            val = validation(tok, model_a, model_b, eval_rows, n=args.eval_n)
            ga = base.generation_protocol_audit(tok, model_a, eval_rows, n=args.gen_audit_n)
            evt = {"event": "stage2_checkpoint_eval", "step": step, "checkpoint": ckpt, "validation": val, "generation_protocol": ga, "elapsed": now() - t0}
            append_jsonl(log, evt)
            write_json(run / "eval" / f"stage2_step{step}.json", evt)
    final = save_checkpoint(run, model_a, model_b, tok, "stage2_final", args.stage2_steps)
    write_json(status, {"status": "stage2_complete", "step": args.stage2_steps, "final_checkpoint": final, "elapsed": now() - t0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["grad-sanity", "stage2"], required=True)
    ap.add_argument("--manifest", default=base.DEFAULT_MANIFEST)
    ap.add_argument("--stage1-ckpt", default=DEFAULT_STAGE1_50K)
    ap.add_argument("--run-dir", default=DEFAULT_RUN)
    ap.add_argument("--repo", default=str(Path.cwd()))
    ap.add_argument("--stage2-steps", type=int, default=25000)
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--gen-audit-n", type=int, default=32)
    args = ap.parse_args()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.mode == "grad-sanity":
        grad_sanity(args)
    else:
        run_stage2(args)


if __name__ == "__main__":
    main()
