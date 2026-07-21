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
from PIL import Image
from transformers import Adafactor, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.htext.formal_eval import hidden_geometry
from src.htext.heima_reuse import (
    HeimaOfficialAbstractProjection,
    backend_resolution_snapshot,
    extract_thinking_state,
    heima_ce_loss,
    official_embedding_replacement,
    prepare_latent_for_decoder,
)

SECTIONS = ("summary", "caption", "reasoning")
THINKING_TOKENS = {
    "summary": "<THINKING_OF_SUMMARY>",
    "caption": "<THINKING_OF_CAPTION>",
    "reasoning": "<THINKING_OF_REASONING>",
}
PREFIX_SECTIONS = {
    "summary": ("summary",),
    "caption": ("summary", "caption"),
    "reasoning": ("summary", "caption", "reasoning"),
}


def prefix_sections(section: str, context_mode: str) -> tuple[str, ...]:
    if context_mode == "local":
        return (section,)
    if context_mode == "causal_cumulative":
        return PREFIX_SECTIONS[section]
    raise ValueError(f"unsupported loss1_latent_context_mode: {context_mode}")


def prepare_stage_latents(z: dict[str, torch.Tensor], section: str, args, *, detach_encoder_latent: bool) -> dict[str, torch.Tensor]:
    out = {}
    prefix = prefix_sections(section, args.loss1_latent_context_mode)
    for s in prefix:
        detach = detach_encoder_latent
        if not detach and args.loss1_latent_context_mode == "causal_cumulative" and args.cumulative_grad_mode == "current_only":
            detach = s != section
        out[s] = prepare_latent_for_decoder(z[s], detach)
    return out


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


def tok(tokenizer, text: str, max_len: int | None = None) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids[:max_len] if max_len else ids


def batch_rows(rows: list[dict], batch_size: int, step: int) -> list[dict]:
    start = (step * batch_size) % len(rows)
    return (rows + rows)[start : start + batch_size]


def model_dim(model) -> int:
    cfg = model.config
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return int(cfg.text_config.hidden_size)
    if hasattr(cfg, "n_embd"):
        return int(cfg.n_embd)
    raise AttributeError("cannot infer hidden size")


def model_dtype(model) -> torch.dtype:
    return next(model.parameters()).dtype


def make_optimizer(param_groups, args):
    if args.optimizer == "adafactor":
        groups = []
        for group in param_groups:
            copied = dict(group)
            copied["weight_decay"] = args.weight_decay
            groups.append(copied)
        return Adafactor(groups, scale_parameter=False, relative_step=False, warmup_init=False)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    raise ValueError(args.optimizer)


def finite_norm(t: torch.Tensor | None) -> tuple[float, bool]:
    if t is None:
        return 0.0, True
    td = t.detach().float()
    return float(td.norm().item()), bool(torch.isfinite(td).all().item())


def image_path(args, rec: dict) -> Path:
    p = Path(rec["image"])
    return p if p.is_absolute() else args.image_root / p


def load_image(path: Path, max_side: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def load_vlm_a(args, device: torch.device):
    dtype = getattr(torch, args.torch_dtype) if args.torch_dtype else None
    processor = AutoProcessor.from_pretrained(args.model_a_path, local_files_only=True, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": list(THINKING_TOKENS.values())})
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


def load_decoder_b(args, device: torch.device):
    dtype = getattr(torch, args.torch_dtype) if args.torch_dtype else None
    kwargs = {"local_files_only": True, "use_safetensors": True, "trust_remote_code": True}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    tokenizer = AutoTokenizer.from_pretrained(args.model_b_path, local_files_only=True, use_safetensors=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": list(THINKING_TOKENS.values())})
    model = AutoModelForCausalLM.from_pretrained(args.model_b_path, **kwargs)
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model.to(device)
    return tokenizer, model


def encoder_text(rec: dict, include_answer: bool) -> str:
    text = "Question:\n" + rec["question"] + "\n"
    for section in SECTIONS:
        text += THINKING_TOKENS[section] + "\n"
    text += "Answer:"
    if include_answer:
        text += " " + rec["answer"]
    return text


def vlm_inputs(processor, args, records: list[dict], include_answer: bool):
    texts, images = [], []
    for rec in records:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": encoder_text(rec, include_answer)},
            ],
        }]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        images.append(load_image(image_path(args, rec), args.max_image_side))
    return processor(text=texts, images=images, padding=True, return_tensors="pt")


def encoder_forward(model_a, processor, tokenizer_a, records: list[dict], args):
    device = next(model_a.parameters()).device
    full = vlm_inputs(processor, args, records, include_answer=True).to(device)
    prefix = vlm_inputs(processor, args, records, include_answer=False)
    labels = torch.full_like(full["input_ids"], -100)
    for i in range(len(records)):
        prefix_len = int(prefix["attention_mask"][i].sum().item())
        full_len = int(full["attention_mask"][i].sum().item())
        labels[i, prefix_len - len(SECTIONS) - 1 : full_len] = full["input_ids"][i, prefix_len - len(SECTIONS) - 1 : full_len]
    out = model_a(**full, output_hidden_states=True, use_cache=False)
    loss = heima_ce_loss(out.logits, labels)
    z, trace = {}, {}
    for section in SECTIONS:
        tid = tokenizer_a.convert_tokens_to_ids(THINKING_TOKENS[section])
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
    return loss, out.logits, labels, z, trace


def pad(tokenizer, ids_rows: list[list[int]], labels_rows: list[list[int]], device: torch.device):
    max_len = max(len(x) for x in ids_rows)
    input_ids = torch.full((len(ids_rows), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    labels = torch.full_like(input_ids, -100)
    attention = torch.zeros_like(input_ids)
    for i, ids in enumerate(ids_rows):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        labels[i, : len(labels_rows[i])] = torch.tensor(labels_rows[i], dtype=torch.long, device=device)
        attention[i, : len(ids)] = 1
    return input_ids, labels, attention


def decoder_prompt(rec: dict, section: str, tokenizer_b, args, q_only: bool = False, context_mode: str = "local") -> str:
    q_ids = tok(tokenizer_b, rec["question"], args.max_q)
    question = tokenizer_b.decode(q_ids, skip_special_tokens=False)
    if q_only:
        return f"Question:\n{question}\n\nInstruction:\nReconstruct the Heima {section} thought. Do not use the image.\n\nTarget:\n"
    if context_mode == "local":
        return (
            f"Question:\n{question}\n\n"
            f"Instruction:\nReconstruct the Heima {section} thought from the latent. Do not use the image.\n\n"
            f"{THINKING_TOKENS[section]}\n\nTarget:\n"
        )
    slots = "\n".join(THINKING_TOKENS[s] for s in prefix_sections(section, context_mode))
    return (
        f"Question:\n{question}\n\n"
        f"Instruction:\nReconstruct the Heima {section} thought from the causal prefix latents. Do not use the image.\n\n"
        f"{slots}\n\nTarget:\n"
    )


def decoder_forward(
    model_b,
    projector,
    tokenizer_b,
    section: str,
    records: list[dict],
    z,
    args,
    q_only: bool = False,
    context_mode: str | None = None,
):
    device = next(model_b.parameters()).device
    rows, label_rows, slots = [], [], []
    context_mode = context_mode or args.loss1_latent_context_mode
    slot_sections = () if q_only else prefix_sections(section, context_mode)
    token_ids = {s: tokenizer_b.convert_tokens_to_ids(THINKING_TOKENS[s]) for s in slot_sections}
    for rec in records:
        prompt_ids = tok(tokenizer_b, decoder_prompt(rec, section, tokenizer_b, args, q_only=q_only, context_mode=context_mode))
        target_ids = tok(tokenizer_b, rec[section] + tokenizer_b.eos_token, args.max_target)
        rows.append(prompt_ids + target_ids)
        labels = [-100] * len(prompt_ids) + target_ids
        if not q_only:
            sample_slots = []
            for slot_section in slot_sections:
                locs = [i for i, value in enumerate(prompt_ids) if value == token_ids[slot_section]]
                if len(locs) != 1:
                    raise RuntimeError(f"expected one {slot_section} slot for {section}, got {locs}")
                sample_slots.append(locs[0])
            if sample_slots != sorted(sample_slots):
                raise RuntimeError(f"latent slots are not in causal order for {section}: {sample_slots}")
            slots.append(sample_slots)
            if getattr(args, "train_latent_marker_ntp", False):
                for slot_section, pos in zip(slot_sections, sample_slots):
                    labels[pos] = token_ids[slot_section]
        label_rows.append(labels)
    input_ids, labels, attention = pad(tokenizer_b, rows, label_rows, device)
    if q_only:
        out = model_b(input_ids=input_ids, attention_mask=attention, use_cache=False)
    else:
        embeds = model_b.get_input_embeddings()(input_ids)
        if isinstance(z, torch.Tensor):
            z_by_section = {section: z}
        else:
            z_by_section = z
        if isinstance(projector, dict):
            projected = torch.stack([projector[s](z_by_section[s]) for s in slot_sections], dim=1)
        else:
            projected = torch.stack([projector(z_by_section[s]) for s in slot_sections], dim=1)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, sample_slots in enumerate(slots):
            for pos in sample_slots:
                mask[i, pos] = True
        expected = len(records) * len(slot_sections)
        if int(mask.sum().item()) != expected:
            raise RuntimeError(f"expected {expected} latent slots for {section}, got {int(mask.sum().item())}")
        embeds = official_embedding_replacement(embeds, projected, mask)
        out = model_b(inputs_embeds=embeds, attention_mask=attention, use_cache=False)
    return heima_ce_loss(out.logits, labels), out.logits, labels


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
def evaluate(model_a, processor, tokenizer_a, decoders, projectors, tokenizer_b, rows: list[dict], args):
    model_a.eval()
    for s in SECTIONS:
        decoders[s].eval()
        projectors[s].eval()
    total_main = 0.0
    totals = {
        s: {
            "qz": 0.0,
            "q": 0.0,
            "whole_shuffle": 0.0,
            "current_shuffle": 0.0,
            "history_shuffle": 0.0,
            "history_count": 0,
            "zero": 0.0,
            "random": 0.0,
            "local_qz": 0.0,
            "zs": [],
        }
        for s in SECTIONS
    }
    z_banks = {s: [] for s in SECTIONS}
    main_losses = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        main, _logits, _labels, z, _trace = encoder_forward(model_a, processor, tokenizer_a, batch, args)
        main_losses.extend([float(main.item())] * len(batch))
        for s in SECTIONS:
            z_banks[s].append(z[s].detach())
    z_all = {s: torch.cat(z_banks[s], dim=0) for s in SECTIONS}
    shuffled = {s: torch.roll(z_all[s], shifts=1, dims=0) if len(rows) > 1 else torch.zeros_like(z_all[s]) for s in SECTIONS}
    random_prefix = {}
    for s in SECTIONS:
        rand = torch.randn_like(z_all[s])
        random_prefix[s] = (rand.float() / rand.float().norm(dim=-1, keepdim=True).clamp_min(1e-6) * z_all[s].float().norm(dim=-1, keepdim=True)).to(z_all[s].dtype)
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        total_main += sum(main_losses[start : start + len(batch)])
        for s in SECTIONS:
            prefix = prefix_sections(s, args.loss1_latent_context_mode)
            z_prefix = {ps: z_all[ps][start : start + len(batch)] for ps in prefix}
            whole = {ps: shuffled[ps][start : start + len(batch)] for ps in prefix}
            current = {ps: (shuffled[ps][start : start + len(batch)] if ps == s else z_prefix[ps]) for ps in prefix}
            history = {ps: (shuffled[ps][start : start + len(batch)] if ps != s else z_prefix[ps]) for ps in prefix}
            zero = {ps: torch.zeros_like(z_prefix[ps]) for ps in prefix}
            random_z = {ps: random_prefix[ps][start : start + len(batch)] for ps in prefix}
            qz, _l, _lab = decoder_forward(decoders[s], projectors, tokenizer_b, s, batch, z_prefix, args)
            q, _ql, _qab = decoder_forward(decoders[s], projectors, tokenizer_b, s, batch, z_prefix, args, q_only=True)
            wh, _sl, _slab = decoder_forward(decoders[s], projectors, tokenizer_b, s, batch, whole, args)
            cur, _cl, _clab = decoder_forward(decoders[s], projectors, tokenizer_b, s, batch, current, args)
            ze, _zl, _zlab = decoder_forward(decoders[s], projectors, tokenizer_b, s, batch, zero, args)
            ra, _rl, _rlab = decoder_forward(decoders[s], projectors, tokenizer_b, s, batch, random_z, args)
            local_z = z_all[s][start : start + len(batch)]
            loc, _ll, _llab = decoder_forward(decoders[s], projectors[s], tokenizer_b, s, batch, local_z, args, context_mode="local")
            totals[s]["qz"] += float(qz.item()) * len(batch)
            totals[s]["q"] += float(q.item()) * len(batch)
            totals[s]["whole_shuffle"] += float(wh.item()) * len(batch)
            totals[s]["current_shuffle"] += float(cur.item()) * len(batch)
            if len(prefix) > 1:
                hi, _hl, _hlab = decoder_forward(decoders[s], projectors, tokenizer_b, s, batch, history, args)
                totals[s]["history_shuffle"] += float(hi.item()) * len(batch)
                totals[s]["history_count"] += len(batch)
            totals[s]["zero"] += float(ze.item()) * len(batch)
            totals[s]["random"] += float(ra.item()) * len(batch)
            totals[s]["local_qz"] += float(loc.item()) * len(batch)
            totals[s]["zs"].append(z_all[s][start : start + len(batch)].detach().cpu())
    n = len(rows)
    out = {"main_nll": total_main / n, "sections": {}}
    for s in SECTIONS:
        qz = totals[s]["qz"] / n
        q = totals[s]["q"] / n
        wh = totals[s]["whole_shuffle"] / n
        cur = totals[s]["current_shuffle"] / n
        hist = None if totals[s]["history_count"] == 0 else totals[s]["history_shuffle"] / totals[s]["history_count"]
        ze = totals[s]["zero"] / n
        ra = totals[s]["random"] / n
        local_qz = totals[s]["local_qz"] / n
        out["sections"][s] = {
            "context_mode": args.loss1_latent_context_mode,
            "nll_correct": qz,
            "qz_nll": qz,
            "q_only_nll": q,
            "nll_q_only": q,
            "shuffle_nll": wh,
            "nll_whole_shuffle": wh,
            "nll_current_shuffle": cur,
            "nll_history_shuffle": hist,
            "zero_nll": ze,
            "nll_zero": ze,
            "nll_random": ra,
            "qz_gain_over_q": q - qz,
            "q_plus_z_gain": q - qz,
            "normal_shuffle_margin": wh - qz,
            "whole_shuffle_margin": wh - qz,
            "current_shuffle_margin": cur - qz,
            "history_shuffle_margin": None if hist is None else hist - qz,
            "normal_zero_margin": ze - qz,
            "cumulative_vs_local_gain": local_qz - qz,
            "latent_geometry": hidden_geometry(torch.cat(totals[s]["zs"], dim=0)),
        }
    return out


@torch.no_grad()
def decoder_generate(model_b, projectors, tokenizer_b, section: str, records: list[dict], z, args, *, q_only: bool = False):
    device = next(model_b.parameters()).device
    rows, slots = [], []
    context_mode = args.loss1_latent_context_mode
    slot_sections = () if q_only else prefix_sections(section, context_mode)
    token_ids = {s: tokenizer_b.convert_tokens_to_ids(THINKING_TOKENS[s]) for s in slot_sections}
    for rec in records:
        prompt_ids = tok(tokenizer_b, decoder_prompt(rec, section, tokenizer_b, args, q_only=q_only, context_mode=context_mode))
        rows.append(prompt_ids)
        if not q_only:
            sample_slots = []
            for slot_section in slot_sections:
                locs = [i for i, value in enumerate(prompt_ids) if value == token_ids[slot_section]]
                if len(locs) != 1:
                    raise RuntimeError(f"expected one {slot_section} generation slot for {section}, got {locs}")
                sample_slots.append(locs[0])
            slots.append(sample_slots)
    input_ids, _labels, attention = pad(tokenizer_b, rows, [[-100] * len(r) for r in rows], device)
    if q_only:
        generated = model_b.generate(
            input_ids=input_ids,
            attention_mask=attention,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer_b.eos_token_id,
            pad_token_id=tokenizer_b.pad_token_id,
        )
        continuations = generated[:, input_ids.shape[1] :]
    else:
        embeds = model_b.get_input_embeddings()(input_ids)
        projected = torch.stack([projectors[s](z[s]) for s in slot_sections], dim=1)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, sample_slots in enumerate(slots):
            for pos in sample_slots:
                mask[i, pos] = True
        embeds = official_embedding_replacement(embeds, projected, mask)
        generated = model_b.generate(
            inputs_embeds=embeds,
            attention_mask=attention,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer_b.eos_token_id,
            pad_token_id=tokenizer_b.pad_token_id,
        )
        continuations = generated
    return tokenizer_b.batch_decode(continuations, skip_special_tokens=True)


@torch.no_grad()
def save_generation_eval(path: Path, model_a, processor, tokenizer_a, decoders, projectors, tokenizer_b, rows: list[dict], args, *, training_group: str, checkpoint: str) -> None:
    model_a.eval()
    for s in SECTIONS:
        decoders[s].eval()
        projectors[s].eval()
    rows = rows[: args.generation_samples]
    z_banks = {s: [] for s in SECTIONS}
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        _main, _logits, _labels, z, _trace = encoder_forward(model_a, processor, tokenizer_a, batch, args)
        for s in SECTIONS:
            z_banks[s].append(z[s].detach())
    z_all = {s: torch.cat(z_banks[s], dim=0) for s in SECTIONS}
    shuffled = {s: torch.roll(z_all[s], shifts=1, dims=0) if len(rows) > 1 else torch.zeros_like(z_all[s]) for s in SECTIONS}
    output = path
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            sample_slice = slice(start, start + len(batch))
            for section in SECTIONS:
                prefix = prefix_sections(section, args.loss1_latent_context_mode)
                correct = {ps: z_all[ps][sample_slice] for ps in prefix}
                whole = {ps: shuffled[ps][sample_slice] for ps in prefix}
                current = {ps: (shuffled[ps][sample_slice] if ps == section else correct[ps]) for ps in prefix}
                history = {ps: (shuffled[ps][sample_slice] if ps != section else correct[ps]) for ps in prefix}
                zero = {ps: torch.zeros_like(correct[ps]) for ps in prefix}
                conditions = {
                    "correct": (correct, False),
                    "whole_prefix_shuffle": (whole, False),
                    "current_only_shuffle": (current, False),
                    "zero": (zero, False),
                    "q_only": (correct, True),
                }
                if len(prefix) > 1:
                    conditions["history_only_shuffle"] = (history, False)
                for condition, (z_variant, q_only) in conditions.items():
                    preds = decoder_generate(decoders[section], projectors, tokenizer_b, section, batch, z_variant, args, q_only=q_only)
                    for offset, (rec, pred) in enumerate(zip(batch, preds)):
                        f.write(json.dumps({
                            "sample_id": start + offset,
                            "question": rec["question"],
                            "stage": section,
                            "gold_text": rec[section],
                            "prediction": pred,
                            "condition": condition,
                            "context_mode": args.loss1_latent_context_mode,
                            "training_group": training_group,
                            "checkpoint": checkpoint,
                        }, ensure_ascii=False) + "\n")


def save_ckpt(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_s0(args, run_dir: Path, train):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, tokenizer_a, model_a = load_vlm_a(args, device)
    opt = make_optimizer([{"params": model_a.parameters(), "lr": args.lr_a}], args)
    logs, first_trace = [], None
    start = time.time()
    for step in range(1, args.s0_steps + 1):
        batch = batch_rows(train, args.batch_size, step - 1)
        opt.zero_grad(set_to_none=True)
        loss, _logits, _labels, _z, trace = encoder_forward(model_a, processor, tokenizer_a, batch, args)
        if first_trace is None:
            first_trace = trace
        loss.backward()
        g, finite = grad_norm(model_a.parameters())
        if not finite:
            raise RuntimeError("non-finite S0 grad")
        torch.nn.utils.clip_grad_norm_(model_a.parameters(), args.clip_grad)
        opt.step()
        if step == 1 or step == args.s0_steps or step % args.log_every == 0:
            logs.append({"step": step, "main": float(loss.item()), "grad_A": g})
    save_ckpt(run_dir / "checkpoints" / "s0_encoder.pt", model_a=model_a.state_dict())
    write_json(run_dir / "groups" / "s0_multisection_main" / "result.json", {"runtime_sec": time.time() - start, "logs": logs, "position_trace_first_batch": first_trace})
    return processor, tokenizer_a, model_a


def train_s1(args, run_dir: Path, processor, tokenizer_a, model_a, train, val):
    set_seed(args.seed + 101)
    device = next(model_a.parameters()).device
    tokenizer_b, proto = load_decoder_b(args, device)
    decoders = {"summary": proto}
    for section in ("caption", "reasoning"):
        _tok, model = load_decoder_b(args, device)
        decoders[section] = model
    projectors = {s: HeimaOfficialAbstractProjection(model_dim(model_a), model_dim(decoders[s])).to(device=device, dtype=model_dtype(decoders[s])) for s in SECTIONS}
    for p in model_a.parameters():
        p.requires_grad_(False)
    model_a.zero_grad(set_to_none=True)
    params = []
    for s in SECTIONS:
        params += [{"params": decoders[s].parameters(), "lr": args.lr_b}, {"params": projectors[s].parameters(), "lr": args.lr_projector}]
    opt = make_optimizer(params, args)
    logs = []
    start = time.time()
    for step in range(1, args.s1_steps + 1):
        batch = batch_rows(train, args.batch_size, step - 1)
        opt.zero_grad(set_to_none=True)
        _main, _logits, _labels, z, _trace = encoder_forward(model_a, processor, tokenizer_a, batch, args)
        loss_sum = torch.zeros((), device=device)
        per = {}
        for s in SECTIONS:
            loss, _l, _lab = decoder_forward(
                decoders[s],
                projectors,
                tokenizer_b,
                s,
                batch,
                prepare_stage_latents(z, s, args, detach_encoder_latent=True),
                args,
            )
            loss_sum = loss_sum + loss
            per[s] = float(loss.item())
        loss_sum.backward()
        ga, _ = grad_norm(model_a.parameters())
        if ga != 0.0:
            raise RuntimeError("S1 sent gradient to frozen A")
        torch.nn.utils.clip_grad_norm_([p for s in SECTIONS for p in list(decoders[s].parameters()) + list(projectors[s].parameters())], args.clip_grad)
        opt.step()
        if step == 1 or step == args.s1_steps or step % args.log_every == 0:
            logs.append({"step": step, "loss1": per, "grad_A": ga})
    metrics = evaluate(model_a, processor, tokenizer_a, decoders, projectors, tokenizer_b, val[: args.eval_samples], args)
    save_ckpt(run_dir / "checkpoints" / "s1_staged_sections.pt", model_a=model_a.state_dict(), decoders={s: decoders[s].state_dict() for s in SECTIONS}, projectors={s: projectors[s].state_dict() for s in SECTIONS})
    write_json(run_dir / "groups" / "s1_staged_sections" / "result.json", {"runtime_sec": time.time() - start, "logs": logs, "validation": metrics})
    return tokenizer_b, decoders, projectors


def load_s1(args, run_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, tokenizer_a, model_a = load_vlm_a(args, device)
    tokenizer_b, proto = load_decoder_b(args, device)
    decoders = {"summary": proto}
    for section in ("caption", "reasoning"):
        _tok, model = load_decoder_b(args, device)
        decoders[section] = model
    projectors = {s: HeimaOfficialAbstractProjection(model_dim(model_a), model_dim(decoders[s])).to(device=device, dtype=model_dtype(decoders[s])) for s in SECTIONS}
    ckpt = torch.load(run_dir / "checkpoints" / "s1_staged_sections.pt", map_location=device)
    model_a.load_state_dict(ckpt["model_a"], strict=True)
    for s in SECTIONS:
        decoders[s].load_state_dict(ckpt["decoders"][s], strict=True)
        projectors[s].load_state_dict(ckpt["projectors"][s], strict=True)
    return processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors


def attribution(args, model_a, processor, tokenizer_a, tokenizer_b, decoders, projectors, batch, detach: bool):
    for m in [model_a] + [decoders[s] for s in SECTIONS] + [projectors[s] for s in SECTIONS]:
        m.zero_grad(set_to_none=True)
    _main, _logits, _labels, z, _trace = encoder_forward(model_a, processor, tokenizer_a, batch, args)
    loss_sum = torch.zeros((), device=next(model_a.parameters()).device)
    stage_latent_grads = {}
    for s in SECTIONS:
        loss, _l, _lab = decoder_forward(
            decoders[s],
            projectors,
            tokenizer_b,
            s,
            batch,
            prepare_stage_latents(z, s, args, detach_encoder_latent=detach),
            args,
        )
        grads = torch.autograd.grad(loss, [z[ps] for ps in SECTIONS], retain_graph=True, allow_unused=True)
        stage_latent_grads[s] = {
            ps: {"norm": finite_norm(g)[0], "finite": finite_norm(g)[1]}
            for ps, g in zip(SECTIONS, grads)
        }
        loss_sum = loss_sum + loss
    loss_sum.backward()
    ga, fa = grad_norm(model_a.parameters())
    if detach and ga != 0.0:
        raise RuntimeError("detach Loss1 returned to A")
    if (not detach) and not (ga > 0 and fa):
        raise RuntimeError("no-detach Loss1 did not reach A")
    return {
        "detach_encoder_latent": detach,
        "loss1_latent_context_mode": args.loss1_latent_context_mode,
        "cumulative_grad_mode": args.cumulative_grad_mode,
        "grad_A_from_loss1": ga,
        "finite": fa,
        "stage_latent_grads": stage_latent_grads,
    }


def train_joint(args, run_dir: Path, train, val, detach: bool, group_name: str | None = None):
    set_seed(args.seed + 202)
    processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors = load_s1(args, run_dir)
    params = [{"params": model_a.parameters(), "lr": args.lr_a}]
    for s in SECTIONS:
        params += [{"params": decoders[s].parameters(), "lr": args.lr_b}, {"params": projectors[s].parameters(), "lr": args.lr_projector}]
    opt = make_optimizer(params, args)
    logs = []
    start = time.time()
    attr = attribution(args, model_a, processor, tokenizer_a, tokenizer_b, decoders, projectors, batch_rows(train, args.batch_size, 0), detach)
    for step in range(1, args.joint_steps + 1):
        batch = batch_rows(train, args.batch_size, step - 1)
        opt.zero_grad(set_to_none=True)
        main, _logits, _labels, z, _trace = encoder_forward(model_a, processor, tokenizer_a, batch, args)
        loss_sum = torch.zeros((), device=main.device)
        per = {}
        for s in SECTIONS:
            loss, _l, _lab = decoder_forward(
                decoders[s],
                projectors,
                tokenizer_b,
                s,
                batch,
                prepare_stage_latents(z, s, args, detach_encoder_latent=detach),
                args,
            )
            loss_sum = loss_sum + loss
            per[s] = float(loss.item())
        total = main + args.lambda1 * loss_sum
        total.backward()
        ga, finite = grad_norm(model_a.parameters())
        if not finite:
            raise RuntimeError("non-finite joint grad")
        torch.nn.utils.clip_grad_norm_([p for group in params for p in group["params"]], args.clip_grad)
        opt.step()
        if step == 1 or step == args.joint_steps or step % args.log_every == 0:
            logs.append({"step": step, "main": float(main.item()), "loss1": per, "total": float(total.item()), "grad_A": ga})
    metrics = evaluate(model_a, processor, tokenizer_a, decoders, projectors, tokenizer_b, val[: args.eval_samples], args)
    if group_name is None:
        if args.loss1_latent_context_mode == "local":
            name = "joint_detach" if detach else "ours_l1_no_detach"
        else:
            name = "cumulative_joint_detach" if detach else f"cumulative_ours_l1_no_detach_{args.cumulative_grad_mode}"
    else:
        name = group_name
    ckpt_path = run_dir / "checkpoints" / f"{name}.pt"
    if not getattr(args, "skip_joint_checkpoints", False):
        save_ckpt(ckpt_path, model_a=model_a.state_dict(), decoders={s: decoders[s].state_dict() for s in SECTIONS}, projectors={s: projectors[s].state_dict() for s in SECTIONS})
    generation_path = None
    if getattr(args, "save_generation_eval", False):
        generation_path = run_dir / "generations" / f"{name}.jsonl"
        save_generation_eval(generation_path, model_a, processor, tokenizer_a, decoders, projectors, tokenizer_b, val, args, training_group=name, checkpoint=str(ckpt_path))
    result = {
        "runtime_sec": time.time() - start,
        "gradient_attribution": attr,
        "logs": logs,
        "validation": metrics,
        "checkpoint_saved": not getattr(args, "skip_joint_checkpoints", False),
        "generation_eval": None if generation_path is None else str(generation_path),
    }
    write_json(run_dir / "groups" / name / "result.json", result)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-a-path", default="/data/zxl/small_models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--model-b-path", default="/data/zxl/small_models/Qwen2.5-0.5B-Instruct")
    p.add_argument("--subset", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_available_images_v1"))
    p.add_argument("--image-root", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1/image_files"))
    p.add_argument("--out", type=Path, default=Path("/data/zxl/runs/heima_small_vlm_official_sections"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--s0-steps", type=int, default=20)
    p.add_argument("--s1-steps", type=int, default=20)
    p.add_argument("--joint-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-samples", type=int, default=8)
    p.add_argument("--lr-a", type=float, default=1e-5)
    p.add_argument("--lr-b", type=float, default=2e-5)
    p.add_argument("--lr-projector", type=float, default=1e-4)
    p.add_argument("--lambda1", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["adafactor", "adamw"], default="adafactor")
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--max-q", type=int, default=160)
    p.add_argument("--max-target", type=int, default=160)
    p.add_argument("--max-image-side", type=int, default=336)
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--train-latent-marker-ntp", action="store_true")
    p.add_argument("--loss1-latent-context-mode", choices=["local", "causal_cumulative"], default="local")
    p.add_argument("--cumulative-grad-mode", choices=["all_prefix", "current_only"], default="all_prefix")
    p.add_argument("--run-local-and-cumulative-comparison", action="store_true")
    p.add_argument("--save-generation-eval", action="store_true")
    p.add_argument("--skip-joint-checkpoints", action="store_true")
    p.add_argument("--generation-samples", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=96)
    args = p.parse_args()
    if args.train_latent_marker_ntp and args.loss1_latent_context_mode != "local":
        raise ValueError("First cumulative Loss1 version uses text-only labels; disable --train-latent-marker-ntp")

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
        "run_id": run_id,
        "host": shell(["hostname"]),
        "gpu": shell(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader"]),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task": "MLLM official-section Heima micro baseline: summary/caption/reasoning",
        "git_baseline_note": "Built from server snapshot d6275da, the commit containing strict Qwen2.5-VL A+B official-section scripts used by text_labels_m0_m1_m2.",
        "model_a": {"type": "VLM", "path": args.model_a_path, "sees": ["image", "question"]},
        "model_b": {"type": "LLM interpreter", "path": args.model_b_path, "sees": ["question", "latent"], "does_not_see": ["image"]},
        "data": {"source": "Xkev/LLaVA-CoT-100k micro subset", "fields_used": ["image", "question", "summary", "caption", "reasoning", "answer"], "train": len(train), "validation": len(val), "test": len(test)},
        "strict_core": {"thinking_state_mode": "predictor", "typed_tokens": THINKING_TOKENS, "projector": "HeimaOfficialAbstractProjection", "replacement": "official_embedding_replacement", "loss": "CEWithChunkedOutputLoss via heima_ce_loss"},
        "args": vars(args),
    }
    write_json(run_dir / "experiment_manifest.json", manifest)
    group_names = ["s0_multisection_main", "s1_staged_sections"]
    if args.run_local_and_cumulative_comparison:
        group_names += ["L-D_local_detach", "L-N_local_no_detach", "C-D_cumulative_detach", "C-N_cumulative_no_detach_all_prefix", "C-N-current_cumulative_no_detach_current_only"]
    else:
        group_names += ["joint_detach" if args.loss1_latent_context_mode == "local" else "cumulative_joint_detach"]
        group_names += ["ours_l1_no_detach" if args.loss1_latent_context_mode == "local" else f"cumulative_ours_l1_no_detach_{args.cumulative_grad_mode}"]
    for g in group_names:
        cfg = copy.deepcopy(manifest)
        cfg["selected_group"] = g
        write_json(run_dir / "groups" / g / "config_full.json", cfg)
    processor, tokenizer_a, model_a = train_s0(args, run_dir, train)
    tokenizer_b, decoders, projectors = train_s1(args, run_dir, processor, tokenizer_a, model_a, train, val)
    del processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if args.run_local_and_cumulative_comparison:
        joint_results = {}
        specs = [
            ("L-D_local_detach", "local", "all_prefix", True),
            ("L-N_local_no_detach", "local", "all_prefix", False),
            ("C-D_cumulative_detach", "causal_cumulative", "all_prefix", True),
            ("C-N_cumulative_no_detach_all_prefix", "causal_cumulative", "all_prefix", False),
            ("C-N-current_cumulative_no_detach_current_only", "causal_cumulative", "current_only", False),
        ]
        for group_name, context_mode, grad_mode, detach in specs:
            run_args = copy.deepcopy(args)
            run_args.loss1_latent_context_mode = context_mode
            run_args.cumulative_grad_mode = grad_mode
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            joint_results[group_name] = train_joint(run_args, run_dir, train, val, detach, group_name=group_name)
        detach_result = joint_results["L-D_local_detach"]
        ours_result = joint_results["L-N_local_no_detach"]
    else:
        detach_result = train_joint(args, run_dir, train, val, True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ours_result = train_joint(args, run_dir, train, val, False)
        joint_results = {
            "joint_detach" if args.loss1_latent_context_mode == "local" else "cumulative_joint_detach": detach_result,
            "ours_l1_no_detach" if args.loss1_latent_context_mode == "local" else f"cumulative_ours_l1_no_detach_{args.cumulative_grad_mode}": ours_result,
        }
    summary = {
        "status": "complete",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "joint_detach": detach_result,
        "ours_l1_no_detach": ours_result,
        "joint_results": joint_results,
        "backend_resolution": backend_resolution_snapshot(),
    }
    write_json(run_dir / "summary.json", summary)
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = summary["completed_at_utc"]
    write_json(run_dir / "experiment_manifest.json", manifest)
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
