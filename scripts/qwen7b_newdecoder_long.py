#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_REVISION = "916b56a44061fd5cd7d6a8fb632557ed4f724f60"
HF_HOME = "/weights2/zhouxiaoling/hf_cache"
THINK = "<THINK>"
THINK_START = "<THINK_START>"
THINK_END = "<THINK_END>"
ANSWER = "<ANSWER>"
SPECIAL_TOKENS = [THINK_START, THINK, THINK_END, ANSWER]
EXPAND_PROMPT = "Please expand the latent token into textual information\n"
DEFAULT_MANIFEST = "/data2/zhouxiaoling/latent_cot/nocap_runs/LLaVA_COT_90K_NOCAP_STAGE1_90K_STAGE2_90K/manifest_train90k_eval5k_floor_nocap.json"
DEFAULT_RUN = "/data2/zhouxiaoling/latent_cot/runs/QWEN7B_90K_NOCAP_THINKSTART_LALL1_NEWDECODER_S1_25K_S2_35K_20260809"
SEED = 42
LR = 2e-4
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 1
NUM_GPUS = 1
EOS_IN_DECODE_LOSS = True

NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?")
CHOICE_RE = re.compile(r"\b([A-E])\b", re.I)


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


def load_split(path: Path) -> tuple[list[dict], list[dict], dict]:
    obj = json.loads(path.read_text())
    if isinstance(obj, dict):
        train = obj.get("train") or obj.get("samples") or []
        eval_rows = obj.get("eval") or []
        meta = obj.get("meta") or {}
    else:
        train, eval_rows, meta = obj, [], {}
    for rows in (train, eval_rows):
        for r in rows:
            r["raw_K"] = int(r.get("raw_K") or r.get("latent_count") or max(1, math.floor(float(r.get("cot_token_count") or 2) * 0.5)))
            r["latent_count"] = int(r.get("latent_count") or r["raw_K"])
            r["k_rule"] = "floor(0.5*N_text_CoT), no cap"
    return train, eval_rows, meta


def norm_num(s: str | None) -> str | None:
    if s is None:
        return None
    s = str(s).strip().replace(",", "")
    return s[:-2] if s.endswith(".0") else s


def parse_num(text: str) -> str | None:
    nums = NUM_RE.findall(str(text or ""))
    return norm_num(nums[-1]) if nums else None


def answer_correct(gold: str, pred: str) -> bool:
    gnum, pnum = parse_num(gold), parse_num(pred)
    if gnum is not None and pnum is not None:
        return gnum == pnum
    gc, pc = CHOICE_RE.findall(gold or ""), CHOICE_RE.findall(pred or "")
    if gc and pc:
        return gc[-1].upper() == pc[-1].upper()
    def nw(x: str) -> str:
        return re.sub(r"\s+", " ", str(x or "").strip()).lower()
    return nw(gold) == nw(pred)


class IdentityLinearProjector(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


def load_tokenizer(tokenizer_path: str | Path | None = None):
    if tokenizer_path and Path(tokenizer_path).exists():
        tok = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True, local_files_only=True)
    else:
        tok = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=HF_HOME,
            trust_remote_code=True,
            local_files_only=True,
        )
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_and_projector(trainable: bool = True, adapter_path: str | Path | None = None, tokenizer_path: str | Path | None = None, projector_path: str | Path | None = None):
    tok = load_tokenizer(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=HF_HOME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.resize_token_embeddings(len(tok))
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if adapter_path:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=trainable)
    elif trainable:
        model = get_peft_model(model, lora_config())
    model.to("cuda")
    hidden = int(model.config.hidden_size)
    projector = IdentityLinearProjector(hidden).to("cuda", dtype=torch.bfloat16)
    if projector_path and Path(projector_path).exists():
        projector.load_state_dict(torch.load(str(projector_path), map_location="cuda"))
    return tok, model, projector


def ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def q_prefix(row: dict) -> str:
    return "Question:\n" + str(row["question"]).strip() + "\n\n"


def answer_text(tok, row: dict) -> str:
    return str(row.get("answer") or row.get("reference_answer") or "").strip() + tok.eos_token


def main_sequence(tok, row: dict) -> tuple[list[int], dict[str, list[int] | int]]:
    start_id = tok.convert_tokens_to_ids(THINK_START)
    think_id = tok.convert_tokens_to_ids(THINK)
    end_id = tok.convert_tokens_to_ids(THINK_END)
    answer_id = tok.convert_tokens_to_ids(ANSWER)
    q = ids(tok, q_prefix(row))
    k = int(row["raw_K"])
    ans = ids(tok, answer_text(tok, row))
    seq = q + [start_id] + [think_id] * k + [end_id, answer_id] + ans
    q_len = len(q)
    start_position = q_len
    think_positions = list(range(q_len + 1, q_len + 1 + k))
    end_position = q_len + 1 + k
    answer_marker_position = q_len + 2 + k
    answer_positions = list(range(q_len + k + 3, len(seq)))
    return seq, {
        "q_len": q_len,
        "k": k,
        "start_position": start_position,
        "think_positions": think_positions,
        "end_position": end_position,
        "answer_marker_position": answer_marker_position,
        "boundary_positions": [end_position, answer_marker_position],
        "answer_positions": answer_positions,
    }


def decoder_sequence(tok, row: dict, k: int) -> tuple[list[int], dict[str, list[int]]]:
    think_id = tok.convert_tokens_to_ids(THINK)
    prompt_ids = ids(tok, EXPAND_PROMPT)
    cot_ids = ids(tok, str(row["gold_cot"]).strip())
    eos_ids = [tok.eos_token_id] if EOS_IN_DECODE_LOSS and tok.eos_token_id is not None else []
    seq = prompt_ids + [think_id] * k + cot_ids + eos_ids
    placeholder_positions = list(range(len(prompt_ids), len(prompt_ids) + k))
    decode_label_positions = list(range(len(prompt_ids) + k, len(seq)))
    return seq, {"placeholder_positions": placeholder_positions, "decode_label_positions": decode_label_positions, "prompt_len": len(prompt_ids), "cot_token_count": len(cot_ids)}


def ce_by_positions(logits: torch.Tensor, input_ids: torch.Tensor, active_positions: list[int]) -> tuple[torch.Tensor, float, int]:
    # active_positions are token positions whose token id is the CE target; prediction comes from previous position.
    positions = [p for p in active_positions if p > 0]
    if not positions:
        z = logits.sum() * 0.0
        return z, float("nan"), 0
    pred_pos = torch.tensor([p - 1 for p in positions], device=logits.device)
    labels = input_ids[0, positions]
    selected = logits[0, pred_pos, :].float()
    losses = F.cross_entropy(selected, labels, reduction="none")
    preds = selected.argmax(dim=-1)
    return losses.mean(), float(preds.eq(labels).float().mean().detach().cpu()), int(labels.numel())


def token_transition_grad(logits: torch.Tensor, token_position: int, token_id: int) -> dict:
    pred_pos = token_position - 1
    if logits.grad is None or pred_pos < 0:
        return {
            "token_position": token_position,
            "pred_position": pred_pos,
            "token_id": token_id,
            "grad": None,
            "grad_abs_sum_at_pred_position": None,
        }
    grad_slice = logits.grad[0, pred_pos, :].detach().float()
    return {
        "token_position": token_position,
        "pred_position": pred_pos,
        "token_id": token_id,
        "grad": float(grad_slice[token_id].cpu()),
        "grad_abs_sum_at_pred_position": float(grad_slice.abs().sum().cpu()),
    }


def forward_components(tok, model, projector, row: dict, need_hidden: bool = True):
    seq, meta = main_sequence(tok, row)
    input_ids = torch.tensor([seq], dtype=torch.long, device="cuda")
    attn = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=need_hidden, use_cache=False)
    start_loss, start_acc, start_count = ce_by_positions(out.logits, input_ids, [meta["start_position"]])
    sync_loss, sync_acc, sync_count = ce_by_positions(out.logits, input_ids, meta["think_positions"])
    end_loss, end_acc, end_count = ce_by_positions(out.logits, input_ids, [meta["end_position"]])
    marker_loss, marker_acc, marker_count = ce_by_positions(out.logits, input_ids, [meta["answer_marker_position"]])
    answer_loss, answer_acc, answer_count = ce_by_positions(out.logits, input_ids, meta["answer_positions"])
    boundary_loss, boundary_acc, boundary_count = ce_by_positions(out.logits, input_ids, meta["boundary_positions"])
    first_sync_loss, first_sync_acc, _ = ce_by_positions(out.logits, input_ids, meta["think_positions"][:1])
    cont_sync_loss, cont_sync_acc, cont_sync_count = ce_by_positions(out.logits, input_ids, meta["think_positions"][1:])
    z = None
    decode = None
    if need_hidden:
        hidden = out.hidden_states[-1]
        z = hidden[:, meta["think_positions"], :]
        z.retain_grad()
        dseq, dmeta = decoder_sequence(tok, row, meta["k"])
        decoder_ids = torch.tensor([dseq], dtype=torch.long, device="cuda")
        decoder_attn = torch.ones_like(decoder_ids)
        embeds = model.get_input_embeddings()(decoder_ids)
        projected = projector(z).to(dtype=embeds.dtype)
        embeds = embeds.clone()
        embeds[:, dmeta["placeholder_positions"], :] = projected
        dout = model(inputs_embeds=embeds, attention_mask=decoder_attn, use_cache=False)
        decode_loss, decode_acc, decode_count = ce_by_positions(dout.logits, decoder_ids, dmeta["decode_label_positions"])
        decode = {
            "loss": decode_loss,
            "token_acc": decode_acc,
            "active_count": decode_count,
            "decoder_seq_len": len(dseq),
            "placeholder_positions": dmeta["placeholder_positions"],
            "decode_label_positions": dmeta["decode_label_positions"],
            "cot_token_count": dmeta["cot_token_count"],
        }
    comps = {
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
        "decode": decode,
        "main_seq_len": len(seq),
        "meta": meta,
    }
    return comps


def grad_norm(parameters) -> float:
    sq = 0.0
    for p in parameters:
        if p.grad is not None:
            sq += float(p.grad.detach().float().norm().cpu()) ** 2
    return math.sqrt(sq)


def trainable_report(tok, model, projector) -> dict:
    emb = model.get_input_embeddings().weight
    lm = model.get_output_embeddings().weight
    return {
        "think_start_token_string": THINK_START,
        "think_start_token_id": tok.convert_tokens_to_ids(THINK_START),
        "sync_token_string": THINK,
        "sync_token_id": tok.convert_tokens_to_ids(THINK),
        "think_end_token_id": tok.convert_tokens_to_ids(THINK_END),
        "answer_token_id": tok.convert_tokens_to_ids(ANSWER),
        "special_tokens": SPECIAL_TOKENS,
        "vocab_size": len(tok),
        "embedding_requires_grad": bool(emb.requires_grad),
        "lm_head_requires_grad": bool(lm.requires_grad),
        "projector_trainable_parameters": sum(p.numel() for p in projector.parameters() if p.requires_grad),
        "model_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "per_device_batch": PER_DEVICE_BATCH,
        "num_gpus": NUM_GPUS,
        "grad_accum": GRAD_ACCUM,
        "effective_global_batch": PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM,
    }


def stage1_total_loss(c: dict) -> torch.Tensor:
    return c["answer_loss"] + c["start_loss"] + c["sync_loss"] + c["end_loss"] + c["marker_loss"]


def stage2_total_loss(c: dict) -> torch.Tensor:
    return stage1_total_loss(c) + c["decode"]["loss"]


def component_record(c: dict) -> dict:
    return {
        "loss_answer": float(c["answer_loss"].detach().cpu()),
        "loss_start": float(c["start_loss"].detach().cpu()),
        "loss_sync": float(c["sync_loss"].detach().cpu()),
        "loss_end": float(c["end_loss"].detach().cpu()),
        "loss_answer_marker": float(c["marker_loss"].detach().cpu()),
        "start_token_nll": float(c["start_loss"].detach().cpu()),
        "first_THINK_nll": float(c["first_sync_loss"].detach().cpu()),
        "continuation_THINK_nll": float(c["continuation_sync_loss"].detach().cpu()),
        "END_nll": float(c["end_loss"].detach().cpu()),
        "ANSWER_marker_nll": float(c["marker_loss"].detach().cpu()),
        "answer_active_token_count": c["answer_active_token_count"],
        "start_active_token_count": c["start_active_token_count"],
        "sync_active_token_count": c["sync_active_token_count"],
        "end_active_token_count": c["end_active_token_count"],
        "answer_marker_active_token_count": c["marker_active_token_count"],
        "K": c["meta"]["k"],
        "main_seq_len": c["main_seq_len"],
    }


def run_audit_smoke(args):
    run = Path(args.run_dir)
    audit_dir = run / "audit"
    train, eval_rows, meta = load_split(Path(args.manifest))
    tok, model, projector = load_model_and_projector(True)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    report = {"dataset_meta": meta, "trainable": trainable_report(tok, model, projector), "samples": []}
    params = [p for p in model.parameters() if p.requires_grad]
    proj_params = [p for p in projector.parameters() if p.requires_grad]
    row = train[0]
    comps = forward_components(tok, model, projector, row, need_hidden=True)
    z = comps["z"]
    comps["main_logits"].retain_grad()
    # sync gradient path
    model.zero_grad(set_to_none=True); projector.zero_grad(set_to_none=True)
    comps["sync_loss"].backward(retain_graph=True)
    sync_model_grad = grad_norm(params)
    # decode gradient path
    model.zero_grad(set_to_none=True); projector.zero_grad(set_to_none=True)
    if z.grad is not None:
        z.grad.zero_()
    comps["decode"]["loss"].backward(retain_graph=True)
    decode_z_grad = float(z.grad.detach().float().norm().cpu()) if z.grad is not None else 0.0
    decode_model_grad = grad_norm(params)
    decode_projector_grad = grad_norm(proj_params)
    model.zero_grad(set_to_none=True); projector.zero_grad(set_to_none=True)
    if comps["main_logits"].grad is not None:
        comps["main_logits"].grad.zero_()
    full_stage1_loss = stage1_total_loss(comps)
    full_stage1_loss.backward(retain_graph=True)
    start_id = tok.convert_tokens_to_ids(THINK_START)
    think_id = tok.convert_tokens_to_ids(THINK)
    end_id = tok.convert_tokens_to_ids(THINK_END)
    answer_id = tok.convert_tokens_to_ids(ANSWER)
    meta_main = comps["meta"]
    first_answer_pos = meta_main["answer_positions"][0] if meta_main["answer_positions"] else None
    first_answer_id = int(comps["main_input_ids"][0, first_answer_pos].detach().cpu()) if first_answer_pos is not None else None
    causal_audit = {
        "Question_last_to_THINK_START": {
            "source_position": meta_main["start_position"] - 1,
            "target_position": meta_main["start_position"],
            "target_token": THINK_START,
            "target_id": start_id,
            "loss_component": "L_start",
            "gradient": token_transition_grad(comps["main_logits"], meta_main["start_position"], start_id),
        },
        "THINK_START_to_first_THINK": {
            "source_position": meta_main["think_positions"][0] - 1,
            "target_position": meta_main["think_positions"][0],
            "target_token": THINK,
            "target_id": think_id,
            "loss_component": "first THINK slice of L_sync",
            "gradient": token_transition_grad(comps["main_logits"], meta_main["think_positions"][0], think_id),
        },
        "THINK_i_to_next_THINK": {
            "source_position": meta_main["think_positions"][1] - 1 if len(meta_main["think_positions"]) > 1 else None,
            "target_position": meta_main["think_positions"][1] if len(meta_main["think_positions"]) > 1 else None,
            "target_token": THINK,
            "target_id": think_id,
            "loss_component": "continuation THINK slice of L_sync",
            "gradient": token_transition_grad(comps["main_logits"], meta_main["think_positions"][1], think_id) if len(meta_main["think_positions"]) > 1 else None,
        },
        "last_THINK_to_THINK_END": {
            "source_position": meta_main["end_position"] - 1,
            "target_position": meta_main["end_position"],
            "target_token": THINK_END,
            "target_id": end_id,
            "loss_component": "L_end",
            "gradient": token_transition_grad(comps["main_logits"], meta_main["end_position"], end_id),
        },
        "THINK_END_to_ANSWER": {
            "source_position": meta_main["answer_marker_position"] - 1,
            "target_position": meta_main["answer_marker_position"],
            "target_token": ANSWER,
            "target_id": answer_id,
            "loss_component": "L_marker",
            "gradient": token_transition_grad(comps["main_logits"], meta_main["answer_marker_position"], answer_id),
        },
        "ANSWER_to_first_gold_answer_token": {
            "source_position": first_answer_pos - 1 if first_answer_pos is not None else None,
            "target_position": first_answer_pos,
            "target_token_id": first_answer_id,
            "loss_component": "L_answer",
            "gradient": token_transition_grad(comps["main_logits"], first_answer_pos, first_answer_id) if first_answer_pos is not None else None,
        },
    }
    sample_info = {
        "N_CoT": int(row.get("cot_token_count") or comps["decode"]["cot_token_count"]),
        "K_target": comps["meta"]["k"],
        "z_shape": list(z.shape),
        "start_position": comps["meta"]["start_position"],
        "sync_positions_head_tail": [comps["meta"]["think_positions"][:5], comps["meta"]["think_positions"][-5:]],
        "end_position": comps["meta"]["end_position"],
        "answer_marker_position": comps["meta"]["answer_marker_position"],
        "answer_positions_head_tail": [comps["meta"]["answer_positions"][:5], comps["meta"]["answer_positions"][-5:]],
        "decoder_placeholder_positions_head_tail": [comps["decode"]["placeholder_positions"][:5], comps["decode"]["placeholder_positions"][-5:]],
        "decode_label_positions_head_tail": [comps["decode"]["decode_label_positions"][:5], comps["decode"]["decode_label_positions"][-5:]],
        "main_sequence_length": comps["main_seq_len"],
        "decoder_sequence_length": comps["decode"]["decoder_seq_len"],
        **component_record(comps),
        "decode_active_token_count": comps["decode"]["active_count"],
        "loss_decode_mean": float(comps["decode"]["loss"].detach().cpu()),
        "grad_sync_to_main_model": sync_model_grad,
        "grad_decode_to_z": decode_z_grad,
        "grad_decode_to_main_model": decode_model_grad,
        "grad_decode_to_projector": decode_projector_grad,
        "causal_position_audit": causal_audit,
    }
    report["samples"].append(sample_info)
    grad_checks = [
        causal_audit["Question_last_to_THINK_START"]["gradient"]["grad_abs_sum_at_pred_position"],
        causal_audit["last_THINK_to_THINK_END"]["gradient"]["grad_abs_sum_at_pred_position"],
        causal_audit["THINK_END_to_ANSWER"]["gradient"]["grad_abs_sum_at_pred_position"],
    ]
    passed = all(
        x > 0 and math.isfinite(x)
        for x in [sync_model_grad, decode_z_grad, decode_model_grad, decode_projector_grad, *grad_checks]
    )
    # 50-step smoke on a fresh model instance in memory, discarded after report.
    opt = torch.optim.AdamW(params + proj_params, lr=LR)
    smoke = []
    for step in range(1, args.smoke_steps + 1):
        r = train[(step - 1) % len(train)]
        torch.cuda.reset_peak_memory_stats()
        t0 = now()
        model.zero_grad(set_to_none=True); projector.zero_grad(set_to_none=True)
        c = forward_components(tok, model, projector, r, need_hidden=True)
        loss = stage2_total_loss(c)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite smoke loss at step {step}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params + proj_params, 1.0)
        opt.step()
        rec = {
            "step": step,
            "loss_total": float(loss.detach().cpu()),
            **component_record(c),
            "L_decode": float(c["decode"]["loss"].detach().cpu()),
            "decode_active_token_count": c["decode"]["active_count"],
            "step_runtime": now() - t0,
            "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
        smoke.append(rec)
    report["smoke"] = smoke
    if len(smoke) >= 20:
        report["smoke_decode_loss_first10_mean"] = sum(x["L_decode"] for x in smoke[:10]) / 10
        report["smoke_decode_loss_last10_mean"] = sum(x["L_decode"] for x in smoke[-10:]) / 10
    report["free_generation_smoke"] = generation_protocol_audit(tok, model, eval_rows, n=min(args.gen_audit_n, 16))
    report["passed"] = bool(passed and all(math.isfinite(x["loss_total"]) for x in smoke))
    write_json(audit_dir / "pretrain_audit_smoke.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(2)


def save_checkpoint(run: Path, model, projector, tok, name: str, step: int) -> str:
    path = run / "checkpoints" / name
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path / "adapter")
    torch.save(projector.state_dict(), path / "projector.pt")
    tok.save_pretrained(run / "checkpoints" / "tokenizer")
    write_json(path / "checkpoint_meta.json", {"step": step, "projector": "identity_linear_trainable", "adapter": str(path / "adapter")})
    return str(path)


def validation_nll(tok, model, projector, rows: list[dict], n: int = 64, stage2: bool = False) -> dict:
    vals = []
    model.eval(); projector.eval()
    with torch.no_grad():
        for r in rows[:n]:
            c = forward_components(tok, model, projector, r, need_hidden=stage2)
            d = {
                "answer": float(c["answer_loss"].detach().cpu()),
                "start": float(c["start_loss"].detach().cpu()),
                "sync": float(c["sync_loss"].detach().cpu()),
                "first_sync": float(c["first_sync_loss"].detach().cpu()),
                "continuation_sync": float(c["continuation_sync_loss"].detach().cpu()),
                "end": float(c["end_loss"].detach().cpu()),
                "answer_marker": float(c["marker_loss"].detach().cpu()),
                "boundary": float(c["boundary_loss"].detach().cpu()),
                "answer_acc": c["answer_acc"],
            }
            if stage2:
                d["decode"] = float(c["decode"]["loss"].detach().cpu())
            vals.append(d)
    model.train(); projector.train()
    def mean(k):
        xs = [x[k] for x in vals if k in x and math.isfinite(x[k])]
        return sum(xs)/len(xs) if xs else None
    return {"n": len(vals), **{f"val_{k}": mean(k) for k in ["answer", "start", "sync", "first_sync", "continuation_sync", "end", "answer_marker", "boundary", "answer_acc", "decode"]}}


def generation_protocol_audit(tok, model, rows: list[dict], n: int = 32, max_new_limit: int = 512) -> dict:
    start_id = tok.convert_tokens_to_ids(THINK_START)
    think_id = tok.convert_tokens_to_ids(THINK)
    end_id = tok.convert_tokens_to_ids(THINK_END)
    answer_id = tok.convert_tokens_to_ids(ANSWER)
    results = []
    model.eval()
    for r in rows[:n]:
        q_ids = ids(tok, q_prefix(r))
        inp = torch.tensor([q_ids], dtype=torch.long, device="cuda")
        attn = torch.ones_like(inp)
        with torch.no_grad():
            logits = model(input_ids=inp, attention_mask=attn, use_cache=False).logits[0, -1].float()
            logp = F.log_softmax(logits, dim=-1)
            first_start_prob = float(logp[start_id].exp().cpu())
            first_start_rank = int((logits > logits[start_id]).sum().detach().cpu()) + 1
            target_k = int(r["raw_K"])
            max_new = min(max(target_k + 96, 128), max_new_limit)
            out = model.generate(
                input_ids=inp,
                attention_mask=attn,
                max_new_tokens=max_new,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
                temperature=None,
                top_p=None,
            )
        gen = out[0, inp.shape[1]:].detach().cpu().tolist()
        start_seen = start_id in gen
        start_pos = gen.index(start_id) if start_seen else -1
        end_pos = gen.index(end_id) if end_id in gen else len(gen)
        before_end = gen[start_pos + 1:end_pos] if start_seen else gen[:end_pos]
        gen_think = sum(1 for x in before_end if x == think_id)
        answer_seen = answer_id in gen
        text = tok.decode(gen, skip_special_tokens=False)
        results.append({
            "sample_id": r.get("sample_id"),
            "target_k": target_k,
            "gold_n": int(r.get("cot_token_count") or target_k * 2),
            "think_start_seen": start_seen,
            "first_token_is_THINK_START": bool(gen and gen[0] == start_id),
            "generated_think": gen_think,
            "generated_think_over_gold_n": gen_think / max(int(r.get("cot_token_count") or 1), 1),
            "generated_think_over_target_k": gen_think / max(target_k, 1),
            "first_THINK_START_probability": first_start_prob,
            "first_THINK_START_rank": first_start_rank,
            "think_end_seen": end_id in gen,
            "answer_seen": answer_seen,
            "answer_correct": answer_correct(str(r.get("answer") or r.get("reference_answer") or ""), text),
            "preview": text[:500],
        })
    model.train()
    def mean(key):
        xs = [x[key] for x in results]
        return sum(xs) / len(xs) if xs else None
    return {
        "n": len(results),
        "target_K_mean": mean("target_k"),
        "THINK_START_rate": sum(x["think_start_seen"] for x in results) / max(len(results), 1),
        "first_token_THINK_START_rate": sum(x["first_token_is_THINK_START"] for x in results) / max(len(results), 1),
        "generated_THINK_mean": mean("generated_think"),
        "generated_THINK_over_gold_N_mean": mean("generated_think_over_gold_n"),
        "generated_THINK_over_target_K_mean": mean("generated_think_over_target_k"),
        "first_THINK_START_probability_mean": mean("first_THINK_START_probability"),
        "first_THINK_START_rank_mean": mean("first_THINK_START_rank"),
        "THINK_END_rate": sum(x["think_end_seen"] for x in results) / max(len(results), 1),
        "ANSWER_rate": sum(x["answer_seen"] for x in results) / max(len(results), 1),
        "final_answer_exact_match": sum(x["answer_correct"] for x in results) / max(len(results), 1),
        "examples": results[:8],
    }


def run_stage1(args):
    repo = Path(args.repo)
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    train, eval_rows, data_meta = load_split(Path(args.manifest))
    tok, model, projector = load_model_and_projector(True)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    params = [p for p in model.parameters() if p.requires_grad]
    proj_params = [p for p in projector.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params + proj_params, lr=LR)
    cfg = {
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=str(repo), text=True).strip(),
        "commit": git_sha(repo),
        "manifest": str(args.manifest),
        "data_meta": data_meta,
        "seed": SEED,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], "modules_to_save": ["embed_tokens", "lm_head"]},
        "loss": {
            "stage1": "L_answer + L_start + L_sync + L_end + L_marker",
            "stage2": "L_answer + L_start + L_sync + L_end + L_marker + L_decode",
            "decode": "causal CE only on gold textual CoT/EOS from expand_prompt + projected z prefix",
            "pooling_cosine_default": False,
            "component_normalization": "each component is independently mean-normalized before summation",
        },
        "batch": trainable_report(tok, model, projector),
        "stage1_target": args.stage1_steps,
        "stage1_extension_ceiling": 30000,
        "samples_seen_stage1": args.stage1_steps * PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM,
        "effective_epochs_stage1": args.stage1_steps * PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM / max(len(train), 1),
    }
    write_json(run / "config.json", cfg)
    log = run / "stage1_train.jsonl"
    status = run / "status.json"
    t0 = now()
    checkpoint_steps = {250, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000}
    stop_after_30k = False
    for step in range(1, args.stage1_steps + 1):
        r = train[(step - 1) % len(train)]
        torch.cuda.reset_peak_memory_stats()
        s0 = now()
        model.zero_grad(set_to_none=True); projector.zero_grad(set_to_none=True)
        c = forward_components(tok, model, projector, r, need_hidden=False)
        loss = stage1_total_loss(c)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite stage1 loss at {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params + proj_params, 1.0)
        opt.step()
        rec = {
            "phase": "stage1",
            "step": step,
            "sample_id": r.get("sample_id"),
            "loss_total": float(loss.detach().cpu()),
            **component_record(c),
            "L_boundary_diagnostic": float(c["boundary_loss"].detach().cpu()),
            "boundary_active_token_count": c["boundary_active_token_count"],
            "step_runtime": now() - s0,
            "elapsed": now() - t0,
            "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
        append_jsonl(log, rec)
        if step % 100 == 0 or step == 1:
            write_json(status, {"status": "running", "phase": "stage1", "step": step, "last": rec, "elapsed": now() - t0})
        if step in checkpoint_steps:
            ckpt = save_checkpoint(run, model, projector, tok, f"stage1_step{step}", step)
            val = validation_nll(tok, model, projector, eval_rows, n=args.eval_n, stage2=False)
            ga = generation_protocol_audit(tok, model, eval_rows, n=args.gen_audit_n)
            evt = {"event": "stage1_checkpoint_eval", "step": step, "checkpoint": ckpt, "validation": val, "generation_protocol": ga, "elapsed": now() - t0}
            append_jsonl(log, evt)
            write_json(run / "eval" / f"stage1_step{step}.json", evt)
            start_ok = (ga.get("THINK_START_rate") or 0.0) > 0.95
            end_ok = (ga.get("THINK_END_rate") or 0.0) > 0.95
            answer_ok = (ga.get("ANSWER_rate") or 0.0) > 0.95
            k_ratio = ga.get("generated_THINK_over_target_K_mean") or 0.0
            k_ok = 0.7 <= k_ratio <= 1.4
            if step >= 5000 and start_ok and end_ok and answer_ok and k_ok:
                write_json(status, {"status": "stage1_gate_passed", "step": step, "generation_protocol": ga, "checkpoint": ckpt})
                break
            if step == 30000 and not (start_ok and end_ok and answer_ok and k_ok):
                stop_after_30k = True
                write_json(status, {"status": "stopped_at_30k_protocol_fail", "reason": "START/END/ANSWER/K gate failed", "step": step, "generation_protocol": ga})
                break
    final = save_checkpoint(run, model, projector, tok, "stage1_final", step)
    write_json(status, {"status": "stage1_complete" if not stop_after_30k else "stage1_stopped_after_gate", "step": step, "final_checkpoint": final, "elapsed": now() - t0})


def run_stage2(args):
    repo = Path(args.repo)
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    train, eval_rows, data_meta = load_split(Path(args.manifest))
    stage1_ckpt = Path(args.stage1_ckpt) if args.stage1_ckpt else run / "checkpoints" / "stage1_final"
    adapter_path = stage1_ckpt / "adapter"
    projector_path = stage1_ckpt / "projector.pt"
    tokenizer_path = run / "checkpoints" / "tokenizer"
    if not adapter_path.exists() or not projector_path.exists():
        raise FileNotFoundError(f"Stage2 requires Stage1 adapter/projector: {stage1_ckpt}")
    tok, model, projector = load_model_and_projector(True, adapter_path=adapter_path, tokenizer_path=tokenizer_path, projector_path=projector_path)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    params = [p for p in model.parameters() if p.requires_grad]
    proj_params = [p for p in projector.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params + proj_params, lr=LR)
    cfg_path = run / "stage2_config.json"
    cfg = {
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=str(repo), text=True).strip(),
        "commit": git_sha(repo),
        "manifest": str(args.manifest),
        "stage1_checkpoint": str(stage1_ckpt),
        "data_meta": data_meta,
        "seed": SEED,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "loss": {
            "stage2": "L_answer + L_start + L_sync + L_end + L_marker + L_decode",
            "decode": "same LLM second causal forward; prompt + K projected z placeholders + gold textual CoT; CE only on gold CoT/EOS",
            "z_detach": False,
            "pooling_cosine": False,
        },
        "batch": trainable_report(tok, model, projector),
        "stage2_target": args.stage2_steps,
        "stage2_extension_ceiling": 45000,
        "samples_seen_stage2": args.stage2_steps * PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM,
        "effective_epochs_stage2": args.stage2_steps * PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM / max(len(train), 1),
    }
    write_json(cfg_path, cfg)
    log = run / "stage2_train.jsonl"
    status = run / "status.json"
    t0 = now()
    checkpoint_steps = {1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000}
    for step in range(1, args.stage2_steps + 1):
        r = train[(step - 1) % len(train)]
        torch.cuda.reset_peak_memory_stats()
        s0 = now()
        model.zero_grad(set_to_none=True); projector.zero_grad(set_to_none=True)
        c = forward_components(tok, model, projector, r, need_hidden=True)
        loss = stage2_total_loss(c)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite stage2 loss at {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params + proj_params, 1.0)
        opt.step()
        rec = {
            "phase": "stage2",
            "step": step,
            "sample_id": r.get("sample_id"),
            "loss_total": float(loss.detach().cpu()),
            **component_record(c),
            "loss_decode": float(c["decode"]["loss"].detach().cpu()),
            "decode_token_acc": c["decode"]["token_acc"],
            "decode_active_token_count": c["decode"]["active_count"],
            "z_grad_norm_after_backward": float(c["z"].grad.detach().float().norm().cpu()) if c["z"].grad is not None else 0.0,
            "step_runtime": now() - s0,
            "elapsed": now() - t0,
            "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
        append_jsonl(log, rec)
        if step % 100 == 0 or step == 1:
            write_json(status, {"status": "running", "phase": "stage2", "step": step, "last": rec, "elapsed": now() - t0})
        if step in checkpoint_steps:
            ckpt = save_checkpoint(run, model, projector, tok, f"stage2_step{step}", step)
            val = validation_nll(tok, model, projector, eval_rows, n=args.eval_n, stage2=True)
            ga = generation_protocol_audit(tok, model, eval_rows, n=args.gen_audit_n)
            evt = {"event": "stage2_checkpoint_eval", "step": step, "checkpoint": ckpt, "validation": val, "generation_protocol": ga, "elapsed": now() - t0}
            append_jsonl(log, evt)
            write_json(run / "eval" / f"stage2_step{step}.json", evt)
    final = save_checkpoint(run, model, projector, tok, "stage2_final", step)
    write_json(status, {"status": "stage2_complete", "step": step, "final_checkpoint": final, "elapsed": now() - t0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["audit-smoke", "stage1", "stage2"], required=True)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--run-dir", default=DEFAULT_RUN)
    ap.add_argument("--repo", default=str(Path.cwd()))
    ap.add_argument("--smoke-steps", type=int, default=50)
    ap.add_argument("--stage1-steps", type=int, default=25000)
    ap.add_argument("--stage2-steps", type=int, default=35000)
    ap.add_argument("--stage1-ckpt", default="")
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--gen-audit-n", type=int, default=32)
    args = ap.parse_args()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.mode == "audit-smoke":
        run_audit_smoke(args)
    elif args.mode == "stage1":
        run_stage1(args)
    elif args.mode == "stage2":
        run_stage2(args)


if __name__ == "__main__":
    main()
