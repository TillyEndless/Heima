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

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer

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
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


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
    if hasattr(model.config, "n_embd"):
        return int(model.config.n_embd)
    return int(model.config.hidden_size)


def model_dtype(model) -> torch.dtype:
    return next(model.parameters()).dtype


def load_model_pair(model_path: str, device: torch.device, dtype: str):
    torch_dtype = getattr(torch, dtype) if dtype else None
    kwargs = {"local_files_only": True, "use_safetensors": True}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_safetensors=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [THINKING_TOKEN]})
    model_a = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model_b = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    for model in (model_a, model_b):
        if not hasattr(model.config, "n_embd") and hasattr(model.config, "hidden_size"):
            model.config.n_embd = model.config.hidden_size
        model.resize_token_embeddings(len(tokenizer))
        model.config.use_cache = False
        model.to(device)
    return tokenizer, model_a, model_b


def load_model_a(model_path: str, device: torch.device, dtype: str):
    tokenizer, model_a, model_b = load_model_pair(model_path, device, dtype)
    del model_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tokenizer, model_a


def tok(tokenizer, text: str, max_len: int | None = None) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids[:max_len] if max_len else ids


def batch_rows(rows: list[dict], batch_size: int, step: int) -> list[dict]:
    start = (step * batch_size) % len(rows)
    return (rows + rows)[start : start + batch_size]


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


def encoder_forward(model_a, tokenizer, records: list[dict], args):
    device = next(model_a.parameters()).device
    thinking_id = tokenizer.convert_tokens_to_ids(THINKING_TOKEN)
    ids_rows, labels_rows = [], []
    for rec in records:
        q = "Question:\n" + rec["question"] + "\n"
        q_ids = tok(tokenizer, q, args.max_q)
        t_ids = [thinking_id]
        a_ids = tok(tokenizer, "\nAnswer: " + rec["answer"] + tokenizer.eos_token, args.max_answer)
        ids = q_ids + t_ids + a_ids
        labels = [-100] * len(q_ids) + t_ids + a_ids
        ids_rows.append(ids)
        labels_rows.append(labels)
    input_ids, labels, attention = pad(tokenizer, ids_rows, labels_rows, device)
    out = model_a(input_ids=input_ids, attention_mask=attention, output_hidden_states=True, use_cache=False)
    loss = heima_ce_loss(out.logits, labels)
    state = extract_thinking_state(
        input_ids=input_ids,
        last_hidden_state=out.hidden_states[-1],
        thinking_token_id=thinking_id,
        mode="predictor",
    )
    return loss, out.logits, labels, state.hidden, {
        "thinking_pos": state.thinking_positions.detach().cpu().tolist(),
        "selected_pos": state.selected_positions.detach().cpu().tolist(),
        "semantics": state.semantics,
    }


def decoder_forward(model_b, projector, tokenizer, records: list[dict], z: torch.Tensor, args, q_only: bool = False):
    device = next(model_b.parameters()).device
    thinking_id = tokenizer.convert_tokens_to_ids(THINKING_TOKEN)
    rows, labels_rows, slot_positions = [], [], []
    for rec in records:
        q_ids = tok(tokenizer, rec["question"], args.max_q)
        question = tokenizer.decode(q_ids, skip_special_tokens=False)
        if q_only:
            prompt = "Question:\n" + question + "\n\nInstruction:\nReconstruct the reasoning thought.\n\nReasoning:\n"
        else:
            prompt = (
                "Question:\n"
                + question
                + "\n\nInstruction:\nReconstruct the reasoning thought from the Heima latent.\n\n"
                + THINKING_TOKEN
                + "\n\nReasoning:\n"
            )
        prompt_ids = tok(tokenizer, prompt)
        target_ids = tok(tokenizer, rec["reasoning"] + tokenizer.eos_token, args.max_target)
        rows.append(prompt_ids + target_ids)
        labels_rows.append([-100] * len(prompt_ids) + target_ids)
        if not q_only:
            locs = [i for i, value in enumerate(prompt_ids) if value == thinking_id]
            if len(locs) != 1:
                raise RuntimeError(f"expected exactly one latent token, got {locs}")
            slot_positions.append(locs[0])
    input_ids, labels, attention = pad(tokenizer, rows, labels_rows, device)
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
def eval_all(model_a, model_b, projector, tokenizer, rows: list[dict], args) -> dict:
    model_a.eval()
    model_b.eval()
    projector.eval()
    main_total = loss1_total = q_total = 0.0
    shuffle_total = zero_total = 0.0
    zs = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        main, _logits, _labels, z, _trace = encoder_forward(model_a, tokenizer, batch, args)
        loss1, _l, _lab = decoder_forward(model_b, projector, tokenizer, batch, z, args)
        qloss, _ql, _qab = decoder_forward(model_b, projector, tokenizer, batch, z, args, q_only=True)
        if len(batch) > 1:
            shuffled_z = torch.roll(z, shifts=1, dims=0)
        else:
            shuffled_z = z
        zero_z = torch.zeros_like(z)
        shuffle_loss, _sl, _slab = decoder_forward(model_b, projector, tokenizer, batch, shuffled_z, args)
        zero_loss, _zl, _zlab = decoder_forward(model_b, projector, tokenizer, batch, zero_z, args)
        main_total += float(main.item()) * len(batch)
        loss1_total += float(loss1.item()) * len(batch)
        q_total += float(qloss.item()) * len(batch)
        shuffle_total += float(shuffle_loss.item()) * len(batch)
        zero_total += float(zero_loss.item()) * len(batch)
        zs.append(z.detach().cpu())
    z_all = torch.cat(zs, dim=0)
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


def train_s0(args, run_dir: Path, train: list[dict], val: list[dict]):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model_a = load_model_a(args.model_path, device, args.torch_dtype)
    opt = torch.optim.AdamW(model_a.parameters(), lr=args.lr_a, weight_decay=args.weight_decay)
    logs = []
    first_trace = None
    start = time.time()
    for step in range(1, args.s0_steps + 1):
        batch = batch_rows(train, args.batch_size, step - 1)
        opt.zero_grad(set_to_none=True)
        model_a.train()
        loss, _logits, _labels, _z, trace = encoder_forward(model_a, tokenizer, batch, args)
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
    save_ckpt(run_dir / "checkpoints" / "s0_encoder.pt", model_a=model_a.state_dict(), optimizer=opt.state_dict())
    write_json(run_dir / "groups" / "s0_main_only" / "result.json", {
        "runtime_sec": time.time() - start,
        "logs": logs,
        "position_trace_first_batch": first_trace,
    })
    return tokenizer, model_a


def train_s1(args, run_dir: Path, tokenizer, model_a, train: list[dict], val: list[dict]):
    set_seed(args.seed + 101)
    device = next(model_a.parameters()).device
    _, _unused_a, model_b = load_model_pair(args.model_path, device, args.torch_dtype)
    del _unused_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dim = model_dim(model_a)
    projector = HeimaOfficialAbstractProjection(dim, model_dim(model_b)).to(device=device, dtype=model_dtype(model_b))
    for p in model_a.parameters():
        p.requires_grad_(False)
    model_a.zero_grad(set_to_none=True)
    opt = torch.optim.AdamW(
        [{"params": model_b.parameters(), "lr": args.lr_b}, {"params": projector.parameters(), "lr": args.lr_projector}],
        weight_decay=args.weight_decay,
    )
    logs = []
    start = time.time()
    for step in range(1, args.s1_steps + 1):
        batch = batch_rows(train, args.batch_size, step - 1)
        opt.zero_grad(set_to_none=True)
        model_a.eval()
        model_b.train()
        projector.train()
        main, _logits, _labels, z, _trace = encoder_forward(model_a, tokenizer, batch, args)
        loss1, _l, _lab = decoder_forward(model_b, projector, tokenizer, batch, z.detach(), args)
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
    metrics = eval_all(model_a, model_b, projector, tokenizer, val[: args.eval_samples], args)
    save_ckpt(
        run_dir / "checkpoints" / "s1_staged.pt",
        model_a=model_a.state_dict(),
        model_b=model_b.state_dict(),
        projector=projector.state_dict(),
        optimizer=opt.state_dict(),
    )
    write_json(run_dir / "groups" / "s1_staged_detach" / "result.json", {
        "runtime_sec": time.time() - start,
        "logs": logs,
        "validation": metrics,
    })
    return model_b, projector


def load_s1(args, run_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model_a, model_b = load_model_pair(args.model_path, device, args.torch_dtype)
    projector = HeimaOfficialAbstractProjection(model_dim(model_a), model_dim(model_b)).to(
        device=device, dtype=model_dtype(model_b)
    )
    ckpt = torch.load(run_dir / "checkpoints" / "s1_staged.pt", map_location=device)
    model_a.load_state_dict(ckpt["model_a"], strict=True)
    model_b.load_state_dict(ckpt["model_b"], strict=True)
    projector.load_state_dict(ckpt["projector"], strict=True)
    return tokenizer, model_a, model_b, projector


def loss1_grad_attribution(args, model_a, model_b, projector, tokenizer, batch: list[dict], detach: bool) -> dict:
    for module in (model_a, model_b, projector):
        module.zero_grad(set_to_none=True)
    main, _logits, _labels, z, _trace = encoder_forward(model_a, tokenizer, batch, args)
    z_for_b = prepare_latent_for_decoder(z, detach)
    loss1, _l, _lab = decoder_forward(model_b, projector, tokenizer, batch, z_for_b, args)
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
    tokenizer, model_a, model_b, projector = load_s1(args, run_dir)
    opt = torch.optim.AdamW(
        [
            {"params": model_a.parameters(), "lr": args.lr_a},
            {"params": model_b.parameters(), "lr": args.lr_b},
            {"params": projector.parameters(), "lr": args.lr_projector},
        ],
        weight_decay=args.weight_decay,
    )
    logs = []
    start = time.time()
    attribution = loss1_grad_attribution(args, model_a, model_b, projector, tokenizer, batch_rows(train, args.batch_size, 0), detach)
    for step in range(1, args.joint_steps + 1):
        batch = batch_rows(train, args.batch_size, step - 1)
        opt.zero_grad(set_to_none=True)
        model_a.train()
        model_b.train()
        projector.train()
        main, _logits, _labels, z, _trace = encoder_forward(model_a, tokenizer, batch, args)
        loss1, _l, _lab = decoder_forward(model_b, projector, tokenizer, batch, prepare_latent_for_decoder(z, detach), args)
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
            logs.append({"step": step, "main": float(main.item()), "loss1": float(loss1.item()), "total": float(total.item()), "grad_A": ga, "grad_B": gb, "grad_projector": gp})
    metrics = eval_all(model_a, model_b, projector, tokenizer, val[: args.eval_samples], args)
    name = "joint_detach" if detach else "ours_l1_no_detach"
    save_ckpt(run_dir / "checkpoints" / f"{name}.pt", model_a=model_a.state_dict(), model_b=model_b.state_dict(), projector=projector.state_dict(), optimizer=opt.state_dict())
    result = {"runtime_sec": time.time() - start, "gradient_attribution": attribution, "logs": logs, "validation": metrics}
    write_json(run_dir / "groups" / name / "result.json", result)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/mnt/nas/share2/home/zxl/small_models/Qwen2.5-0.5B-Instruct")
    p.add_argument("--subset", type=Path, default=Path("/mnt/nas/share2/home/zxl/official_heima/micro_subsets/chartqa_sqa_available_images_v1"))
    p.add_argument("--out", type=Path, default=Path("/mnt/nas/share2/home/zxl/cad218_runs/heima_small_qwen_l1"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--s0-steps", type=int, default=20)
    p.add_argument("--s1-steps", type=int, default=20)
    p.add_argument("--joint-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-samples", type=int, default=16)
    p.add_argument("--lr-a", type=float, default=2e-5)
    p.add_argument("--lr-b", type=float, default=2e-5)
    p.add_argument("--lr-projector", type=float, default=1e-4)
    p.add_argument("--lambda1", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--max-q", type=int, default=160)
    p.add_argument("--max-answer", type=int, default=48)
    p.add_argument("--max-target", type=int, default=160)
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--log-every", type=int, default=5)
    args = p.parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.out / run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    train = read_jsonl(args.subset / "train.jsonl")
    val = read_jsonl(args.subset / "validation.jsonl")
    test = read_jsonl(args.subset / "test.jsonl")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {
        "status": "running",
        "run_id": run_id,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": run(["hostname"]),
        "gpu": run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader"]),
        "git_status": run(["git", "-C", str(ROOT), "status", "--short"]),
        "script_sha256": script_hash,
        "experiment_scope": "cad218 resource-adapted small-model Main/Loss1 comparison",
        "caveat": "Uses official LLaVA-CoT micro text fields and Qwen2.5-0.5B-Instruct. It is not the official 11B vision encoder + 8B decoder Heima baseline.",
        "model_weights": {
            "name": "Qwen/Qwen2.5-0.5B-Instruct",
            "local_path": args.model_path,
            "revision_observed_at_download": "7ae557604adf67be50417f59c2c2f167def9a775",
        },
        "data": {
            "subset": str(args.subset),
            "train_records": len(train),
            "validation_records": len(val),
            "test_records": len(test),
            "fields_used": ["question", "reasoning", "answer"],
            "images_used": False,
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
        "groups": {
            "s0_main_only": {"loss": "Lmain", "trainable": ["Model A"]},
            "s1_staged_detach": {"loss": "Loss1", "trainable": ["Model B", "projector"], "Model_A_frozen": True},
            "joint_detach": {"loss": "Lmain + lambda1*Loss1", "detach_encoder_latent": True},
            "ours_l1_no_detach": {"loss": "Lmain + lambda1*Loss1", "detach_encoder_latent": False},
        },
    }
    write_json(run_dir / "experiment_manifest.json", manifest)
    for name, group in manifest["groups"].items():
        cfg = copy.deepcopy(manifest)
        cfg["selected_group"] = name
        cfg["selected_group_config"] = group
        write_json(run_dir / "groups" / name / "config_full.json", cfg)

    tokenizer, model_a = train_s0(args, run_dir, train, val)
    model_b, projector = train_s1(args, run_dir, tokenizer, model_a, train, val)
    del model_a, model_b, projector, tokenizer
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
        "delta_ours_minus_detach_qz_gain": ours_result["validation"]["qz_gain_over_q"] - detach_result["validation"]["qz_gain_over_q"],
        "backend_resolution": backend_resolution_snapshot(),
    }
    write_json(run_dir / "summary.json", summary)
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = summary["completed_at_utc"]
    write_json(run_dir / "experiment_manifest.json", manifest)
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
