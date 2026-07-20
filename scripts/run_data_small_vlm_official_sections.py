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


def decoder_prompt(rec: dict, section: str, tokenizer_b, args, q_only: bool = False) -> str:
    q_ids = tok(tokenizer_b, rec["question"], args.max_q)
    question = tokenizer_b.decode(q_ids, skip_special_tokens=False)
    if q_only:
        return f"Question:\n{question}\n\nInstruction:\nReconstruct the Heima {section} thought. Do not use the image.\n\nTarget:\n"
    return (
        f"Question:\n{question}\n\n"
        f"Instruction:\nReconstruct the Heima {section} thought from the latent. Do not use the image.\n\n"
        f"{THINKING_TOKENS[section]}\n\nTarget:\n"
    )


def decoder_forward(model_b, projector, tokenizer_b, section: str, records: list[dict], z: torch.Tensor, args, q_only: bool = False):
    device = next(model_b.parameters()).device
    rows, label_rows, slots = [], [], []
    token_id = tokenizer_b.convert_tokens_to_ids(THINKING_TOKENS[section])
    for rec in records:
        prompt_ids = tok(tokenizer_b, decoder_prompt(rec, section, tokenizer_b, args, q_only=q_only))
        target_ids = tok(tokenizer_b, rec[section] + tokenizer_b.eos_token, args.max_target)
        rows.append(prompt_ids + target_ids)
        labels = [-100] * len(prompt_ids) + target_ids
        if not q_only:
            locs = [i for i, value in enumerate(prompt_ids) if value == token_id]
            if len(locs) != 1:
                raise RuntimeError(f"expected one slot for {section}, got {locs}")
            slots.append(locs[0])
            if getattr(args, "train_latent_marker_ntp", False):
                labels[locs[0]] = token_id
        label_rows.append(labels)
    input_ids, labels, attention = pad(tokenizer_b, rows, label_rows, device)
    if q_only:
        out = model_b(input_ids=input_ids, attention_mask=attention, use_cache=False)
    else:
        embeds = model_b.get_input_embeddings()(input_ids)
        projected = projector(z).unsqueeze(1)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, pos in enumerate(slots):
            mask[i, pos] = True
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
    totals = {s: {"qz": 0.0, "q": 0.0, "shuffle": 0.0, "zero": 0.0, "zs": []} for s in SECTIONS}
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
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        total_main += sum(main_losses[start : start + len(batch)])
        for s in SECTIONS:
            z = z_all[s][start : start + len(batch)]
            qz, _l, _lab = decoder_forward(decoders[s], projectors[s], tokenizer_b, s, batch, z, args)
            q, _ql, _qab = decoder_forward(decoders[s], projectors[s], tokenizer_b, s, batch, z, args, q_only=True)
            sh, _sl, _slab = decoder_forward(decoders[s], projectors[s], tokenizer_b, s, batch, shuffled[s][start : start + len(batch)], args)
            ze, _zl, _zlab = decoder_forward(decoders[s], projectors[s], tokenizer_b, s, batch, torch.zeros_like(z), args)
            totals[s]["qz"] += float(qz.item()) * len(batch)
            totals[s]["q"] += float(q.item()) * len(batch)
            totals[s]["shuffle"] += float(sh.item()) * len(batch)
            totals[s]["zero"] += float(ze.item()) * len(batch)
            totals[s]["zs"].append(z.detach().cpu())
    n = len(rows)
    out = {"main_nll": total_main / n, "sections": {}}
    for s in SECTIONS:
        qz = totals[s]["qz"] / n
        q = totals[s]["q"] / n
        sh = totals[s]["shuffle"] / n
        ze = totals[s]["zero"] / n
        out["sections"][s] = {
            "qz_nll": qz,
            "q_only_nll": q,
            "shuffle_nll": sh,
            "zero_nll": ze,
            "qz_gain_over_q": q - qz,
            "normal_shuffle_margin": sh - qz,
            "normal_zero_margin": ze - qz,
            "latent_geometry": hidden_geometry(torch.cat(totals[s]["zs"], dim=0)),
        }
    return out


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
            loss, _l, _lab = decoder_forward(decoders[s], projectors[s], tokenizer_b, s, batch, z[s].detach(), args)
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
    for s in SECTIONS:
        loss, _l, _lab = decoder_forward(decoders[s], projectors[s], tokenizer_b, s, batch, prepare_latent_for_decoder(z[s], detach), args)
        loss_sum = loss_sum + loss
    loss_sum.backward()
    ga, fa = grad_norm(model_a.parameters())
    if detach and ga != 0.0:
        raise RuntimeError("detach Loss1 returned to A")
    if (not detach) and not (ga > 0 and fa):
        raise RuntimeError("no-detach Loss1 did not reach A")
    return {"detach_encoder_latent": detach, "grad_A_from_loss1": ga, "finite": fa}


def train_joint(args, run_dir: Path, train, val, detach: bool):
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
            loss, _l, _lab = decoder_forward(decoders[s], projectors[s], tokenizer_b, s, batch, prepare_latent_for_decoder(z[s], detach), args)
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
    name = "joint_detach" if detach else "ours_l1_no_detach"
    save_ckpt(run_dir / "checkpoints" / f"{name}.pt", model_a=model_a.state_dict(), decoders={s: decoders[s].state_dict() for s in SECTIONS}, projectors={s: projectors[s].state_dict() for s in SECTIONS})
    result = {"runtime_sec": time.time() - start, "gradient_attribution": attr, "logs": logs, "validation": metrics}
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
    args = p.parse_args()

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
        "model_a": {"type": "VLM", "path": args.model_a_path, "sees": ["image", "question"]},
        "model_b": {"type": "LLM interpreter", "path": args.model_b_path, "sees": ["question", "latent"], "does_not_see": ["image"]},
        "data": {"source": "Xkev/LLaVA-CoT-100k micro subset", "fields_used": ["image", "question", "summary", "caption", "reasoning", "answer"], "train": len(train), "validation": len(val), "test": len(test)},
        "strict_core": {"thinking_state_mode": "predictor", "typed_tokens": THINKING_TOKENS, "projector": "HeimaOfficialAbstractProjection", "replacement": "official_embedding_replacement", "loss": "CEWithChunkedOutputLoss via heima_ce_loss"},
        "args": vars(args),
    }
    write_json(run_dir / "experiment_manifest.json", manifest)
    for g in ["s0_multisection_main", "s1_staged_sections", "joint_detach", "ours_l1_no_detach"]:
        cfg = copy.deepcopy(manifest)
        cfg["selected_group"] = g
        write_json(run_dir / "groups" / g / "config_full.json", cfg)
    processor, tokenizer_a, model_a = train_s0(args, run_dir, train)
    tokenizer_b, decoders, projectors = train_s1(args, run_dir, processor, tokenizer_a, model_a, train, val)
    del processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    detach_result = train_joint(args, run_dir, train, val, True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ours_result = train_joint(args, run_dir, train, val, False)
    summary = {
        "status": "complete",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "joint_detach": detach_result,
        "ours_l1_no_detach": ours_result,
        "backend_resolution": backend_resolution_snapshot(),
    }
    write_json(run_dir / "summary.json", summary)
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = summary["completed_at_utc"]
    write_json(run_dir / "experiment_manifest.json", manifest)
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
