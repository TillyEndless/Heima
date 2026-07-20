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

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import Adafactor, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from src.htext.formal_eval import hidden_geometry
from src.htext.heima_reuse import (
    HeimaOfficialAbstractProjection,
    backend_resolution_snapshot,
    extract_thinking_state,
    heima_ce_loss,
    official_embedding_replacement,
    prepare_latent_for_decoder,
)

THINKING_TOKEN = "<THINKING_OF_REASONING>"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_dim(model) -> int:
    cfg = model.config
    if hasattr(cfg, "n_embd"):
        return int(cfg.n_embd)
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return int(cfg.text_config.hidden_size)
    raise AttributeError("cannot infer model hidden size")


def model_dtype(model) -> torch.dtype:
    return next(model.parameters()).dtype


def make_optimizer(param_groups, args):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    if args.optimizer == "adafactor":
        groups = []
        for group in param_groups:
            copied = dict(group)
            copied["weight_decay"] = args.weight_decay
            groups.append(copied)
        return Adafactor(
            groups,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    if args.optimizer == "sgd":
        return torch.optim.SGD(param_groups, weight_decay=args.weight_decay)
    raise ValueError(f"unsupported optimizer: {args.optimizer}")


def tok(tokenizer, text: str, max_len: int | None = None) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids[:max_len] if max_len else ids


def batch_rows(rows: list[dict], batch_size: int, step: int) -> list[dict]:
    start = (step * batch_size) % len(rows)
    return (rows + rows)[start : start + batch_size]


def image_path(args, rec: dict) -> Path:
    path = Path(rec["image"])
    return path if path.is_absolute() else args.image_root / path


def load_image(path: Path, max_side: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def load_vlm_a(args, device: torch.device):
    torch_dtype = getattr(torch, args.torch_dtype) if args.torch_dtype else None
    processor = AutoProcessor.from_pretrained(args.model_a_path, local_files_only=True, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [THINKING_TOKEN]})
    kwargs = {"local_files_only": True, "trust_remote_code": True}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForImageTextToText.from_pretrained(args.model_a_path, **kwargs)
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)
    return processor, tokenizer, model


def load_llm_b(args, device: torch.device):
    torch_dtype = getattr(torch, args.torch_dtype) if args.torch_dtype else None
    kwargs = {"local_files_only": True, "use_safetensors": True, "trust_remote_code": True}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    tokenizer = AutoTokenizer.from_pretrained(args.model_b_path, local_files_only=True, use_safetensors=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [THINKING_TOKEN]})
    model = AutoModelForCausalLM.from_pretrained(args.model_b_path, **kwargs)
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model.to(device)
    return tokenizer, model


def encoder_text(rec: dict, answer: bool) -> str:
    text = "Question:\n" + rec["question"] + "\n" + THINKING_TOKEN + "\nAnswer:"
    if answer:
        text += " " + rec["answer"]
    return text


def vlm_inputs(processor, args, records: list[dict], include_answer: bool):
    texts, images = [], []
    for rec in records:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": encoder_text(rec, include_answer)},
                ],
            }
        ]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        images.append(load_image(image_path(args, rec), args.max_image_side))
    return processor(text=texts, images=images, padding=True, return_tensors="pt")


def encoder_forward(model_a, processor, tokenizer, records: list[dict], args):
    device = next(model_a.parameters()).device
    thinking_id = tokenizer.convert_tokens_to_ids(THINKING_TOKEN)
    full = vlm_inputs(processor, args, records, include_answer=True).to(device)
    prefix = vlm_inputs(processor, args, records, include_answer=False)
    labels = torch.full_like(full["input_ids"], -100)
    for i in range(len(records)):
        prefix_len = int(prefix["attention_mask"][i].sum().item())
        full_len = int(full["attention_mask"][i].sum().item())
        labels[i, prefix_len - 1 : full_len] = full["input_ids"][i, prefix_len - 1 : full_len]
    out = model_a(**full, output_hidden_states=True, use_cache=False)
    loss = heima_ce_loss(out.logits, labels)
    state = extract_thinking_state(
        input_ids=full["input_ids"],
        last_hidden_state=out.hidden_states[-1],
        thinking_token_id=thinking_id,
        mode="predictor",
    )
    return loss, out.logits, labels, state.hidden, {
        "thinking_pos": state.thinking_positions.detach().cpu().tolist(),
        "selected_pos": state.selected_positions.detach().cpu().tolist(),
        "semantics": state.semantics,
        "image_paths": [str(image_path(args, rec)) for rec in records],
    }


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


def decoder_forward(model_b, projector, tokenizer_b, records: list[dict], z: torch.Tensor, args, q_only: bool = False):
    device = next(model_b.parameters()).device
    thinking_id = tokenizer_b.convert_tokens_to_ids(THINKING_TOKEN)
    rows, labels_rows, slot_positions = [], [], []
    for rec in records:
        q_ids = tok(tokenizer_b, rec["question"], args.max_q)
        question = tokenizer_b.decode(q_ids, skip_special_tokens=False)
        if q_only:
            prompt = "Question:\n" + question + "\n\nInstruction:\nReconstruct the visual reasoning thought.\n\nReasoning:\n"
        else:
            prompt = (
                "Question:\n"
                + question
                + "\n\nInstruction:\nReconstruct the visual reasoning thought from the Heima latent. Do not use the image.\n\n"
                + THINKING_TOKEN
                + "\n\nReasoning:\n"
            )
        prompt_ids = tok(tokenizer_b, prompt)
        target_ids = tok(tokenizer_b, rec["reasoning"] + tokenizer_b.eos_token, args.max_target)
        rows.append(prompt_ids + target_ids)
        labels_rows.append([-100] * len(prompt_ids) + target_ids)
        if not q_only:
            locs = [i for i, value in enumerate(prompt_ids) if value == thinking_id]
            if len(locs) != 1:
                raise RuntimeError(f"expected exactly one latent token, got {locs}")
            slot_positions.append(locs[0])
    input_ids, labels, attention = pad(tokenizer_b, rows, labels_rows, device)
    if q_only:
        out = model_b(input_ids=input_ids, attention_mask=attention, use_cache=False)
    else:
        projected = projector(z).unsqueeze(1)
        embeds = model_b.get_input_embeddings()(input_ids)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, pos in enumerate(slot_positions):
            mask[i, pos] = True
        embeds = official_embedding_replacement(embeds, projected, mask)
        out = model_b(inputs_embeds=embeds, attention_mask=attention, use_cache=False)
    loss = heima_ce_loss(out.logits, labels)
    return loss, out.logits, labels


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
def eval_all(model_a, processor, tokenizer_a, model_b, projector, tokenizer_b, rows: list[dict], args) -> dict:
    model_a.eval()
    model_b.eval()
    projector.eval()
    main_total = loss1_total = q_total = shuffle_total = zero_total = 0.0
    zs = []
    main_losses = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        main, _logits, _labels, z, _trace = encoder_forward(model_a, processor, tokenizer_a, batch, args)
        main_losses.extend([float(main.item())] * len(batch))
        zs.append(z.detach())
    z_all_device = torch.cat(zs, dim=0)
    if len(rows) > 1:
        shuffled_all = torch.roll(z_all_device, shifts=1, dims=0)
    else:
        shuffled_all = torch.zeros_like(z_all_device)
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        z = z_all_device[start : start + len(batch)]
        shuffled_z = shuffled_all[start : start + len(batch)]
        zero_z = torch.zeros_like(z)
        loss1, _l, _lab = decoder_forward(model_b, projector, tokenizer_b, batch, z, args)
        qloss, _ql, _qab = decoder_forward(model_b, projector, tokenizer_b, batch, z, args, q_only=True)
        shuffle_loss, _sl, _slab = decoder_forward(model_b, projector, tokenizer_b, batch, shuffled_z, args)
        zero_loss, _zl, _zlab = decoder_forward(model_b, projector, tokenizer_b, batch, zero_z, args)
        main_total += sum(main_losses[start : start + len(batch)])
        loss1_total += float(loss1.item()) * len(batch)
        q_total += float(qloss.item()) * len(batch)
        shuffle_total += float(shuffle_loss.item()) * len(batch)
        zero_total += float(zero_loss.item()) * len(batch)
    z_all = z_all_device.detach().cpu()
    normal = loss1_total / len(rows)
    shuffled = shuffle_total / len(rows)
    zero = zero_total / len(rows)
    q_only = q_total / len(rows)
    return {
        "main_nll": main_total / len(rows),
        "loss1_qz_nll": normal,
        "loss1_q_only_nll": q_only,
        "loss1_shuffle_nll": shuffled,
        "loss1_zero_nll": zero,
        "qz_gain_over_q": q_only - normal,
        "normal_shuffle_margin": shuffled - normal,
        "normal_zero_margin": zero - normal,
        "latent_geometry": hidden_geometry(z_all),
    }


def save_ckpt(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_s0(args, run_dir: Path, train: list[dict]):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, tokenizer_a, model_a = load_vlm_a(args, device)
    opt = make_optimizer([{"params": model_a.parameters(), "lr": args.lr_a}], args)
    logs, first_trace = [], None
    start = time.time()
    for step in range(1, args.s0_steps + 1):
        batch = batch_rows(train, args.batch_size, step - 1)
        opt.zero_grad(set_to_none=True)
        model_a.train()
        loss, _logits, _labels, _z, trace = encoder_forward(model_a, processor, tokenizer_a, batch, args)
        if first_trace is None:
            first_trace = trace
        loss.backward()
        g, finite = grad_norm(model_a.parameters())
        if not finite:
            raise RuntimeError("S0 non-finite gradient")
        torch.nn.utils.clip_grad_norm_(model_a.parameters(), args.clip_grad)
        opt.step()
        if step == 1 or step == args.s0_steps or step % args.log_every == 0:
            logs.append({"step": step, "main_loss": float(loss.item()), "grad_A": g})
    payload = {"model_a": model_a.state_dict()}
    if args.save_optimizer:
        payload["optimizer"] = opt.state_dict()
    save_ckpt(run_dir / "checkpoints" / "s0_vlm_encoder.pt", **payload)
    write_json(run_dir / "groups" / "s0_vlm_main_only" / "result.json", {
        "runtime_sec": time.time() - start,
        "logs": logs,
        "position_trace_first_batch": first_trace,
    })
    return processor, tokenizer_a, model_a


def train_s1(args, run_dir: Path, processor, tokenizer_a, model_a, train: list[dict], val: list[dict]):
    set_seed(args.seed + 101)
    device = next(model_a.parameters()).device
    tokenizer_b, model_b = load_llm_b(args, device)
    projector = HeimaOfficialAbstractProjection(model_dim(model_a), model_dim(model_b)).to(device=device, dtype=model_dtype(model_b))
    for p in model_a.parameters():
        p.requires_grad_(False)
    model_a.zero_grad(set_to_none=True)
    opt = make_optimizer(
        [{"params": model_b.parameters(), "lr": args.lr_b}, {"params": projector.parameters(), "lr": args.lr_projector}],
        args,
    )
    logs = []
    start = time.time()
    for step in range(1, args.s1_steps + 1):
        batch = batch_rows(train, args.batch_size, step - 1)
        opt.zero_grad(set_to_none=True)
        model_a.eval()
        model_b.train()
        projector.train()
        _main, _logits, _labels, z, _trace = encoder_forward(model_a, processor, tokenizer_a, batch, args)
        loss1, _l, _lab = decoder_forward(model_b, projector, tokenizer_b, batch, z.detach(), args)
        loss1.backward()
        ga, _ = grad_norm(model_a.parameters())
        gb, fb = grad_norm(model_b.parameters())
        gp, fp = grad_norm(projector.parameters())
        if ga != 0.0:
            raise RuntimeError("S1 sent gradients to frozen A")
        if not (fb and fp and gb > 0 and gp > 0):
            raise RuntimeError("S1 invalid B/projector gradients")
        torch.nn.utils.clip_grad_norm_(list(model_b.parameters()) + list(projector.parameters()), args.clip_grad)
        opt.step()
        if step == 1 or step == args.s1_steps or step % args.log_every == 0:
            logs.append({"step": step, "loss1": float(loss1.item()), "grad_A": ga, "grad_B": gb, "grad_projector": gp})
    metrics = eval_all(model_a, processor, tokenizer_a, model_b, projector, tokenizer_b, val[: args.eval_samples], args)
    save_ckpt(
        run_dir / "checkpoints" / "s1_vlm_staged.pt",
        model_a=model_a.state_dict(),
        model_b=model_b.state_dict(),
        projector=projector.state_dict(),
    )
    if args.save_optimizer:
        ckpt = torch.load(run_dir / "checkpoints" / "s1_vlm_staged.pt", map_location="cpu")
        ckpt["optimizer"] = opt.state_dict()
        torch.save(ckpt, run_dir / "checkpoints" / "s1_vlm_staged.pt")
    write_json(run_dir / "groups" / "s1_vlm_staged_detach" / "result.json", {
        "runtime_sec": time.time() - start,
        "logs": logs,
        "validation": metrics,
    })
    return tokenizer_b, model_b, projector


def load_s1(args, run_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, tokenizer_a, model_a = load_vlm_a(args, device)
    tokenizer_b, model_b = load_llm_b(args, device)
    projector = HeimaOfficialAbstractProjection(model_dim(model_a), model_dim(model_b)).to(device=device, dtype=model_dtype(model_b))
    ckpt = torch.load(run_dir / "checkpoints" / "s1_vlm_staged.pt", map_location=device)
    model_a.load_state_dict(ckpt["model_a"], strict=True)
    model_b.load_state_dict(ckpt["model_b"], strict=True)
    projector.load_state_dict(ckpt["projector"], strict=True)
    return processor, tokenizer_a, model_a, tokenizer_b, model_b, projector


def loss1_grad_attribution(args, model_a, processor, tokenizer_a, model_b, projector, tokenizer_b, batch: list[dict], detach: bool) -> dict:
    for module in (model_a, model_b, projector):
        module.zero_grad(set_to_none=True)
    _main, _logits, _labels, z, _trace = encoder_forward(model_a, processor, tokenizer_a, batch, args)
    z_for_b = prepare_latent_for_decoder(z, detach)
    loss1, _l, _lab = decoder_forward(model_b, projector, tokenizer_b, batch, z_for_b, args)
    gz = None
    if not detach:
        gz = torch.autograd.grad(loss1, z, retain_graph=True)[0]
    loss1.backward()
    ga, fa = grad_norm(model_a.parameters())
    gb, fb = grad_norm(model_b.parameters())
    gp, fp = grad_norm(projector.parameters())
    if detach and ga != 0.0:
        raise RuntimeError("detach branch Loss1 returned gradient to A")
    if (not detach) and not (ga > 0 and fa):
        raise RuntimeError("no-detach branch Loss1 did not return gradient to A")
    return {
        "detach_encoder_latent": detach,
        "grad_A_from_loss1": ga,
        "grad_B_from_loss1": gb,
        "grad_projector_from_loss1": gp,
        "g_loss1_to_z": 0.0 if gz is None else float(gz.detach().float().norm().item()),
        "finite": bool(fa and fb and fp),
    }


def train_joint(args, run_dir: Path, train: list[dict], val: list[dict], detach: bool) -> dict:
    set_seed(args.seed + 202)
    processor, tokenizer_a, model_a, tokenizer_b, model_b, projector = load_s1(args, run_dir)
    opt = make_optimizer(
        [
            {"params": model_a.parameters(), "lr": args.lr_a},
            {"params": model_b.parameters(), "lr": args.lr_b},
            {"params": projector.parameters(), "lr": args.lr_projector},
        ],
        args,
    )
    logs = []
    start = time.time()
    attribution = loss1_grad_attribution(
        args, model_a, processor, tokenizer_a, model_b, projector, tokenizer_b, batch_rows(train, args.batch_size, 0), detach
    )
    for step in range(1, args.joint_steps + 1):
        batch = batch_rows(train, args.batch_size, step - 1)
        opt.zero_grad(set_to_none=True)
        model_a.train()
        model_b.train()
        projector.train()
        main, _logits, _labels, z, _trace = encoder_forward(model_a, processor, tokenizer_a, batch, args)
        loss1, _l, _lab = decoder_forward(model_b, projector, tokenizer_b, batch, prepare_latent_for_decoder(z, detach), args)
        total = main + args.lambda1 * loss1
        total.backward()
        ga, fa = grad_norm(model_a.parameters())
        gb, fb = grad_norm(model_b.parameters())
        gp, fp = grad_norm(projector.parameters())
        if not (fa and fb and fp and torch.isfinite(total).item()):
            raise RuntimeError("joint non-finite loss/gradient")
        torch.nn.utils.clip_grad_norm_(list(model_a.parameters()) + list(model_b.parameters()) + list(projector.parameters()), args.clip_grad)
        opt.step()
        if step == 1 or step == args.joint_steps or step % args.log_every == 0:
            logs.append({"step": step, "main": float(main.item()), "loss1": float(loss1.item()), "total": float(total.item()), "grad_A": ga})
    metrics = eval_all(model_a, processor, tokenizer_a, model_b, projector, tokenizer_b, val[: args.eval_samples], args)
    name = "joint_detach" if detach else "ours_l1_no_detach"
    payload = {"model_a": model_a.state_dict(), "model_b": model_b.state_dict(), "projector": projector.state_dict()}
    if args.save_optimizer:
        payload["optimizer"] = opt.state_dict()
    save_ckpt(run_dir / "checkpoints" / f"{name}.pt", **payload)
    result = {"runtime_sec": time.time() - start, "gradient_attribution": attribution, "logs": logs, "validation": metrics}
    write_json(run_dir / "groups" / name / "result.json", result)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-a-path", default="/data/zxl/small_models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--model-b-path", default="/data/zxl/small_models/Qwen2.5-0.5B-Instruct")
    p.add_argument("--subset", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_available_images_v1"))
    p.add_argument("--image-root", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1/image_files"))
    p.add_argument("--out", type=Path, default=Path("/data/zxl/runs/heima_small_vlm_l1"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--s0-steps", type=int, default=20)
    p.add_argument("--s1-steps", type=int, default=20)
    p.add_argument("--joint-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-samples", type=int, default=12)
    p.add_argument("--lr-a", type=float, default=1e-5)
    p.add_argument("--lr-b", type=float, default=2e-5)
    p.add_argument("--lr-projector", type=float, default=1e-4)
    p.add_argument("--lambda1", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["adamw", "adafactor", "sgd"], default="adamw")
    p.add_argument("--save-optimizer", action="store_true")
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--max-q", type=int, default=160)
    p.add_argument("--max-target", type=int, default=160)
    p.add_argument("--max-image-side", type=int, default=448)
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--log-every", type=int, default=5)
    args = p.parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.out / run_id
    train = read_jsonl(args.subset / "train.jsonl")
    val = read_jsonl(args.subset / "validation.jsonl")
    test = read_jsonl(args.subset / "test.jsonl")
    missing = [str(image_path(args, rec)) for rec in train + val + test if not image_path(args, rec).exists()]
    if missing:
        raise FileNotFoundError(f"missing images: {missing[:5]} count={len(missing)}")
    manifest = {
        "status": "running",
        "run_id": run_id,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": run(["hostname"]),
        "gpu": run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader"]),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "experiment_scope": "small multimodal Model A + small LLM Model B Heima-style Main/Loss1 wiring",
        "caveat": "Resource-adapted micro experiment; not official 11B/8B Heima reproduction.",
        "model_weights": {
            "model_a": {"name": "Qwen/Qwen2.5-VL-3B-Instruct", "local_path": str(args.model_a_path), "type": "VLM"},
            "model_b": {"name": "Qwen/Qwen2.5-0.5B-Instruct", "local_path": str(args.model_b_path), "type": "LLM interpreter"},
        },
        "data": {
            "source": "Xkev/LLaVA-CoT-100k official-data micro subset",
            "subset": str(args.subset),
            "image_root": str(args.image_root),
            "train_records": len(train),
            "validation_records": len(val),
            "test_records": len(test),
            "fields_used": ["image", "question", "reasoning", "answer"],
            "model_a_sees": ["image", "question"],
            "model_b_sees": ["question", "latent"],
        },
        "strict_core": {
            "thinking_state_mode": "predictor",
            "thinking_token": THINKING_TOKEN,
            "projector": "HeimaOfficialAbstractProjection",
            "embedding_replacement": "official_embedding_replacement",
            "loss": "heima_ce_loss / CEWithChunkedOutputLoss when available",
            "detach_only_function": "prepare_latent_for_decoder",
        },
        "training_args": vars(args),
    }
    write_json(run_dir / "experiment_manifest.json", manifest)
    for name in ["s0_vlm_main_only", "s1_vlm_staged_detach", "joint_detach", "ours_l1_no_detach"]:
        cfg = copy.deepcopy(manifest)
        cfg["selected_group"] = name
        write_json(run_dir / "groups" / name / "config_full.json", cfg)

    processor, tokenizer_a, model_a = train_s0(args, run_dir, train)
    tokenizer_b, model_b, projector = train_s1(args, run_dir, processor, tokenizer_a, model_a, train, val)
    del processor, tokenizer_a, model_a, tokenizer_b, model_b, projector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    detach_result = train_joint(args, run_dir, train, val, detach=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ours_result = train_joint(args, run_dir, train, val, detach=False)
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
