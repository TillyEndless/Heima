#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.htext.formal_eval import cot_nll_breakdown, hidden_geometry, logits_kl, make_latent_variants, operation_type_match
from src.htext.heima_reuse import (
    HeimaOfficialAbstractProjection,
    backend_resolution_snapshot,
    extract_thinking_state,
    heima_ce_loss,
    prepare_latent_for_decoder,
)
from src.htext.modeling import (
    THINKING_TOKEN,
    _decoder_prompt_ids,
    extract_thinking_hidden,
    h0_forward,
    h1_forward,
    setup_special_tokens,
)
from src.htext.synthetic_data import generate_synthetic_split, read_jsonl, validate_record, write_jsonl
from src.htext.trainer import _answer_em, _grad_norm, _load_tokenizer_and_model, batch_records, set_seed


OUT = Path("reports/strict_difference")
CKPT = Path("checkpoints/strict_difference")
DATA = Path("experiments/htext_gpt2/data/strict_difference")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite_number(value: float) -> bool:
    return math.isfinite(float(value))


def norm_of_params(params) -> tuple[float, bool]:
    return _grad_norm(params)


def grad_vector(params) -> torch.Tensor:
    pieces = []
    for p in params:
        if p.grad is not None:
            pieces.append(p.grad.detach().flatten().float().cpu())
    if not pieces:
        return torch.zeros(1)
    return torch.cat(pieces)


def zero_all(*modules) -> None:
    for module in modules:
        module.zero_grad(set_to_none=True)


def combo_key(record: dict) -> tuple:
    meta = record["metadata"]
    values = tuple(sorted((k, v) for k, v in meta.items() if k not in {"generator_version"}))
    return record["operation_type"], values, record["question"]


def ensure_data(seed: int, train_size: int, val_size: int, ood_size: int) -> dict:
    seed_dir = DATA / f"seed_{seed}"
    train_path = seed_dir / "train.jsonl"
    val_path = seed_dir / "validation.jsonl"
    ood_path = seed_dir / "ood.jsonl"
    if train_path.exists() and val_path.exists() and ood_path.exists():
        train, val, ood = read_jsonl(train_path), read_jsonl(val_path), read_jsonl(ood_path)
    else:
        train, val = generate_synthetic_split(train_size, val_size, seed)
        used = {combo_key(r) for r in train + val}
        pool_train, pool_val = generate_synthetic_split(train_size + val_size + ood_size * 4, ood_size * 2, seed + 10000)
        ood = []
        for record in pool_train + pool_val:
            key = combo_key(record)
            if key in used:
                continue
            item = dict(record)
            item["split"] = "ood"
            item["id"] = f"htext_ood_{len(ood):04d}_seed_{seed}"
            ood.append(item)
            used.add(key)
            if len(ood) == ood_size:
                break
        if len(ood) != ood_size:
            raise RuntimeError(f"could only build {len(ood)} OOD records for seed {seed}")
        write_jsonl(train, train_path)
        write_jsonl(val, val_path)
        write_jsonl(ood, ood_path)
    for split_name, rows in [("train", train), ("validation", val), ("ood", ood)]:
        if not all(validate_record({**r, "split": "train" if split_name == "ood" else r["split"]}) if split_name == "ood" else validate_record(r) for r in rows):
            raise RuntimeError(f"invalid generated {split_name} records for seed {seed}")
    if {combo_key(r) for r in train} & {combo_key(r) for r in val}:
        raise RuntimeError("train/validation overlap")
    if ({combo_key(r) for r in train} | {combo_key(r) for r in val}) & {combo_key(r) for r in ood}:
        raise RuntimeError("OOD overlaps train/validation")
    return {"train_path": str(train_path), "validation_path": str(val_path), "ood_path": str(ood_path)}


def base_config(seed: int, paths: dict, steps: int, batch_size: int, eval_samples: int) -> dict:
    return {
        "compatibility_mode": "strict_heima_repo",
        "heima_compatibility": "strict_repo",
        "thinking_state_mode": "predictor",
        "thinking_token": THINKING_TOKEN,
        "projector_type": "heima_official",
        "projector": "heima_official",
        "loss_backend": "torchtune_chunked_ce",
        "allow_loss_fallback": False,
        "model_name_or_path": "/mnt/nas/share2/home/zxl/models/openai-community-gpt2",
        "local_files_only": True,
        "use_safetensors": True,
        "seed": seed,
        "num_thinking_tokens": 1,
        "lambda1": 0.1,
        "h1_mode": "qz",
        "micro_batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "learning_rate_A": 1.0e-5,
        "learning_rate_B": 1.0e-5,
        "learning_rate_projector": 5.0e-5,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "max_question_tokens": 80,
        "max_answer_tokens": 16,
        "max_cot_tokens": 96,
        "max_steps": steps,
        "log_interval": max(1, steps // 4),
        "eval_samples": eval_samples,
        **paths,
    }


def assert_strict_components(config: dict) -> dict:
    if config["compatibility_mode"] != "strict_heima_repo":
        raise RuntimeError("compatibility_mode is not strict_heima_repo")
    if config["thinking_state_mode"] != "predictor":
        raise RuntimeError("thinking_state_mode must be predictor")
    if config["thinking_token"] != THINKING_TOKEN:
        raise RuntimeError("wrong thinking token")
    if config["projector_type"] != "heima_official":
        raise RuntimeError("wrong projector type")
    logits = torch.randn(2, 4, 8, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, 4, -100, 5]])
    loss = heima_ce_loss(logits, labels)
    loss.backward()
    backend = backend_resolution_snapshot()["ce_loss"]
    if backend["fallback_used"]:
        raise RuntimeError(f"CE fallback used: {backend}")
    input_ids = torch.tensor([[4, 5, 99, 6]])
    hidden = torch.randn(1, 4, 3)
    state = extract_thinking_state(input_ids=input_ids, last_hidden_state=hidden, thinking_token_id=99, mode="predictor")
    if int(state.selected_positions[0]) != int(state.thinking_positions[0]) - 1:
        raise RuntimeError("selected_pos != thinking_pos - 1")
    shifted_input_labels = input_ids[:, 1:]
    if int(shifted_input_labels[0, int(state.selected_positions[0])]) != 99:
        raise RuntimeError("selected hidden does not predict thinking token in shifted labels")
    return {"backend": backend, "selected_pos_check": True}


def assert_prompt_and_labels(tokenizer, model_a, model_b, config: dict, records: list[dict]) -> dict:
    out = h0_forward(
        model_a,
        tokenizer,
        records[:1],
        config["max_question_tokens"],
        config["max_answer_tokens"],
        config["num_thinking_tokens"],
        "predictor",
    )
    thinking_id = tokenizer.convert_tokens_to_ids(THINKING_TOKEN)
    if int(out.selected_positions[0, 0]) != int(out.thinking_positions[0, 0]) - 1:
        raise RuntimeError("actual H0 selected predictor position mismatch")
    shifted_labels = out.labels[:, 1:]
    if int(shifted_labels[0, int(out.selected_positions[0, 0])]) != thinking_id:
        raise RuntimeError("actual selected hidden does not predict thinking token")
    if (out.labels == thinking_id).sum().item() != 1:
        raise RuntimeError("Main labels do not supervise exactly one thinking token")
    prompt_ids, latent_pos = _decoder_prompt_ids(tokenizer, records[0]["question"], "qz")
    if latent_pos is None or prompt_ids[latent_pos] != thinking_id:
        raise RuntimeError("B replacement position is not typed thinking token")
    prompt_text = tokenizer.decode(prompt_ids)
    if "Question" not in prompt_text:
        raise RuntimeError("B prompt does not contain Question")
    z = out.thinking_hidden.detach()
    projector = HeimaOfficialAbstractProjection(model_a.config.n_embd, model_b.config.n_embd).to(next(model_b.parameters()).device)
    pred = h1_forward(model_b, tokenizer, records[:1], z, projector, config["max_cot_tokens"], mode="qz")
    if (pred.labels != -100).sum().item() == 0:
        raise RuntimeError("Loss1 has zero non-ignored labels")
    return {
        "selected_pos": int(out.selected_positions[0, 0]),
        "thinking_pos": int(out.thinking_positions[0, 0]),
        "b_prompt_contains_question": True,
        "replacement_token_id": int(prompt_ids[latent_pos]),
        "main_supervises_thinking_and_answer": True,
        "loss1_target": "whole_cot",
    }


def save_s0(path: Path, model_a, optimizer, rng_state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_a": model_a.state_dict(), "optimizer": optimizer.state_dict(), "rng_state": rng_state}, path)


def save_s1(path: Path, model_a, model_b, projector, optimizer, rng_state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_a": model_a.state_dict(),
            "model_b": model_b.state_dict(),
            "projector": projector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng_state": rng_state,
        },
        path,
    )


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def train_s0(config: dict, seed_dir: Path) -> tuple[dict, object, object]:
    set_seed(config["seed"])
    tokenizer, model_a, _, _ = _load_tokenizer_and_model(config, with_b=False)
    train = read_jsonl(config["train_path"])
    val = read_jsonl(config["validation_path"])
    optimizer = torch.optim.AdamW(model_a.parameters(), lr=config["learning_rate_A"], weight_decay=config["weight_decay"])
    logs = []
    start = time.time()
    for step in range(1, config["max_steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        records = batch_records(train, config["micro_batch_size"], step - 1)
        model_a.train()
        out = h0_forward(model_a, tokenizer, records, config["max_question_tokens"], config["max_answer_tokens"], 1, "predictor")
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model_a.parameters(), config["max_grad_norm"])
        grad, finite = norm_of_params(model_a.parameters())
        if not finite:
            raise RuntimeError("S0 non-finite gradient")
        optimizer.step()
        if step == 1 or step % config["log_interval"] == 0 or step == config["max_steps"]:
            logs.append({"step": step, "main_loss": float(out.loss.item()), "grad_A": grad})
    s0_path = seed_dir / "s0_encoder.pt"
    save_s0(s0_path, model_a, optimizer, rng_state())
    metrics = eval_encoder(config, tokenizer, model_a, val[: config["eval_samples"]])
    return {"checkpoint": str(s0_path), "logs": logs, "eval": metrics, "runtime_sec": time.time() - start}, tokenizer, model_a


def train_s1(config: dict, tokenizer, model_a, seed_dir: Path) -> tuple[dict, object, object]:
    set_seed(config["seed"] + 101)
    _, unused_model_a, model_b, _ = _load_tokenizer_and_model(config, with_b=True)
    del unused_model_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model_b.to(next(model_a.parameters()).device)
    for p in model_a.parameters():
        p.requires_grad_(False)
    zero_all(model_a)
    model_a.eval()
    projector = HeimaOfficialAbstractProjection(model_a.config.n_embd, model_b.config.n_embd).to(next(model_b.parameters()).device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model_b.parameters(), "lr": config["learning_rate_B"]},
            {"params": projector.parameters(), "lr": config["learning_rate_projector"]},
        ],
        weight_decay=config["weight_decay"],
    )
    train = read_jsonl(config["train_path"])
    val = read_jsonl(config["validation_path"])
    logs = []
    start = time.time()
    for step in range(1, config["max_steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        records = batch_records(train, config["micro_batch_size"], step - 1)
        zero_all(model_a)
        model_b.train()
        projector.train()
        z, _ = extract_thinking_hidden(model_a, tokenizer, records, config["max_question_tokens"], 1, "predictor")
        out = h1_forward(model_b, tokenizer, records, z.detach(), projector, config["max_cot_tokens"], mode="qz")
        out.loss.backward()
        a_grad, _ = norm_of_params(model_a.parameters())
        b_grad, b_finite = norm_of_params(model_b.parameters())
        p_grad, p_finite = norm_of_params(projector.parameters())
        if a_grad != 0.0:
            raise RuntimeError("S1 produced Model A gradients")
        if not (b_finite and p_finite and b_grad > 0 and p_grad > 0):
            raise RuntimeError("S1 B/projector gradients invalid")
        torch.nn.utils.clip_grad_norm_(list(model_b.parameters()) + list(projector.parameters()), config["max_grad_norm"])
        optimizer.step()
        if step == 1 or step % config["log_interval"] == 0 or step == config["max_steps"]:
            logs.append({"step": step, "loss1": float(out.loss.item()), "grad_A": a_grad, "grad_B": b_grad, "grad_projector": p_grad})
    s1_path = seed_dir / "s1_staged.pt"
    save_s1(s1_path, model_a, model_b, projector, optimizer, rng_state())
    interventions = eval_interventions(config, tokenizer, model_a, model_b, projector, val[: config["eval_samples"]], include_generation=False)
    q_only = eval_q_only(config, tokenizer, model_b, val[: config["eval_samples"]])
    return {"checkpoint": str(s1_path), "logs": logs, "interventions": interventions, "q_only": q_only, "runtime_sec": time.time() - start}, model_b, projector


def load_s1_for_joint(config: dict, s1_path: Path):
    tokenizer, model_a, model_b, _ = _load_tokenizer_and_model(config, with_b=True)
    ckpt = torch.load(s1_path, map_location=next(model_a.parameters()).device)
    model_a.load_state_dict(ckpt["model_a"], strict=True)
    model_b.load_state_dict(ckpt["model_b"], strict=True)
    projector = HeimaOfficialAbstractProjection(model_a.config.n_embd, model_b.config.n_embd).to(next(model_a.parameters()).device)
    projector.load_state_dict(ckpt["projector"], strict=True)
    return tokenizer, model_a, model_b, projector


def make_joint_optimizer(config: dict, model_a, model_b, projector):
    return torch.optim.AdamW(
        [
            {"params": model_a.parameters(), "lr": config["learning_rate_A"]},
            {"params": model_b.parameters(), "lr": config["learning_rate_B"]},
            {"params": projector.parameters(), "lr": config["learning_rate_projector"]},
        ],
        weight_decay=config["weight_decay"],
    )


def train_joint_branch(config: dict, s1_path: Path, seed_dir: Path, detach: bool, branch_seed_state: dict) -> tuple[dict, object, object, object]:
    restore_rng(branch_seed_state)
    tokenizer, model_a, model_b, projector = load_s1_for_joint(config, s1_path)
    optimizer = make_joint_optimizer(config, model_a, model_b, projector)
    train = read_jsonl(config["train_path"])
    val = read_jsonl(config["validation_path"])
    ood = read_jsonl(config["ood_path"])
    logs = []
    start = time.time()
    attribution = gradient_attribution(config, tokenizer, model_a, model_b, projector, batch_records(train, config["micro_batch_size"], 0), detach)
    for step in range(1, config["max_steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        records = batch_records(train, config["micro_batch_size"], step - 1)
        model_a.train()
        model_b.train()
        projector.train()
        main = h0_forward(model_a, tokenizer, records, config["max_question_tokens"], config["max_answer_tokens"], 1, "predictor")
        decoder_z = prepare_latent_for_decoder(main.thinking_hidden, detach)
        loss1 = h1_forward(model_b, tokenizer, records, decoder_z, projector, config["max_cot_tokens"], mode="qz")
        total = main.loss + config["lambda1"] * loss1.loss
        total.backward()
        a_grad, a_finite = norm_of_params(model_a.parameters())
        b_grad, b_finite = norm_of_params(model_b.parameters())
        p_grad, p_finite = norm_of_params(projector.parameters())
        if not (a_finite and b_finite and p_finite and finite_number(float(total.item()))):
            raise RuntimeError("Joint NaN/Inf")
        torch.nn.utils.clip_grad_norm_(list(model_a.parameters()) + list(model_b.parameters()) + list(projector.parameters()), config["max_grad_norm"])
        optimizer.step()
        if step == 1 or step % config["log_interval"] == 0 or step == config["max_steps"]:
            logs.append({"step": step, "main_loss": float(main.loss.item()), "loss1": float(loss1.loss.item()), "total": float(total.item()), "grad_A": a_grad, "grad_B": b_grad, "grad_projector": p_grad})
    branch = "joint_detach" if detach else "joint_no_detach"
    ckpt_path = seed_dir / f"{branch}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_a": model_a.state_dict(), "model_b": model_b.state_dict(), "projector": projector.state_dict(), "optimizer": optimizer.state_dict(), "rng_state": rng_state()}, ckpt_path)
    val_eval = eval_encoder(config, tokenizer, model_a, val[: config["eval_samples"]])
    ood_eval = eval_encoder(config, tokenizer, model_a, ood[: config["eval_samples"]])
    interventions = eval_interventions(config, tokenizer, model_a, model_b, projector, val[: config["eval_samples"]], include_generation=True)
    q_only = eval_q_only(config, tokenizer, model_b, val[: config["eval_samples"]])
    return {
        "checkpoint": str(ckpt_path),
        "logs": logs,
        "gradient_attribution": attribution,
        "validation": val_eval,
        "ood": ood_eval,
        "interventions": interventions,
        "q_only": q_only,
        "runtime_sec": time.time() - start,
    }, tokenizer, model_a, model_b


def gradient_attribution(config: dict, tokenizer, model_a, model_b, projector, records: list[dict], detach: bool) -> dict:
    lambda1 = config["lambda1"]
    zero_all(model_a, model_b, projector)
    main = h0_forward(model_a, tokenizer, records, config["max_question_tokens"], config["max_answer_tokens"], 1, "predictor")
    main.loss.backward()
    main_a = grad_vector(model_a.parameters())
    grad_a_main, _ = norm_of_params(model_a.parameters())

    zero_all(model_a, model_b, projector)
    main = h0_forward(model_a, tokenizer, records, config["max_question_tokens"], config["max_answer_tokens"], 1, "predictor")
    z = prepare_latent_for_decoder(main.thinking_hidden, detach)
    loss1 = h1_forward(model_b, tokenizer, records, z, projector, config["max_cot_tokens"], mode="qz")
    if detach:
        g_z = None
    else:
        g_z = torch.autograd.grad(loss1.loss, main.thinking_hidden, retain_graph=True)[0]
    loss1.loss.backward()
    loss1_a = grad_vector(model_a.parameters())
    grad_a_loss1, finite_a_loss1 = norm_of_params(model_a.parameters())
    grad_b_loss1, finite_b_loss1 = norm_of_params(model_b.parameters())
    grad_p_loss1, finite_p_loss1 = norm_of_params(projector.parameters())

    zero_all(model_a, model_b, projector)
    main = h0_forward(model_a, tokenizer, records, config["max_question_tokens"], config["max_answer_tokens"], 1, "predictor")
    z = prepare_latent_for_decoder(main.thinking_hidden, detach)
    loss1 = h1_forward(model_b, tokenizer, records, z, projector, config["max_cot_tokens"], mode="qz")
    (main.loss + lambda1 * loss1.loss).backward()
    grad_total_a, finite_total_a = norm_of_params(model_a.parameters())

    if detach and grad_a_loss1 != 0.0:
        raise RuntimeError("Loss1 returned to A in detach mode")
    if (not detach) and not (grad_a_loss1 > 0 and finite_a_loss1):
        raise RuntimeError("Loss1 failed to return to A in no-detach mode")
    cos = None
    if main_a.norm().item() > 0 and loss1_a.norm().item() > 0:
        cos = float(F.cosine_similarity(main_a, loss1_a, dim=0).item())
    return {
        "detach_encoder_latent": detach,
        "grad_A_from_main": grad_a_main,
        "grad_A_from_loss1": grad_a_loss1,
        "grad_B_from_loss1": grad_b_loss1,
        "grad_projector_from_loss1": grad_p_loss1,
        "g_loss1_to_z": 0.0 if g_z is None else float(g_z.detach().float().norm().item()),
        "grad_A_from_total": grad_total_a,
        "cosine_main_grad_A_loss1_grad_A": cos,
        "norm_ratio_loss1_over_main": grad_a_loss1 / grad_a_main if grad_a_main > 0 else None,
        "finite": bool(finite_a_loss1 and finite_b_loss1 and finite_p_loss1 and finite_total_a),
    }


def eval_encoder(config: dict, tokenizer, model_a, records: list[dict]) -> dict:
    model_a.eval()
    chunks = []
    zs = []
    with torch.no_grad():
        for start in range(0, len(records), config["micro_batch_size"]):
            batch = records[start : start + config["micro_batch_size"]]
            out = h0_forward(model_a, tokenizer, batch, config["max_question_tokens"], config["max_answer_tokens"], 1, "predictor")
            chunks.append(out)
            zs.append(out.thinking_hidden.detach().cpu())
    n = sum(out.labels.size(0) for out in chunks)
    answer_nll = sum(float(out.loss.item()) * out.labels.size(0) for out in chunks) / max(n, 1)
    answer_em = sum(_answer_em(out.logits, out.labels) * out.labels.size(0) for out in chunks) / max(n, 1)
    z = torch.cat(zs, dim=0)
    return {"answer_nll": answer_nll, "answer_em": answer_em, "latent_geometry": hidden_geometry(z), "retrieval": latent_retrieval(z)}


def latent_retrieval(z: torch.Tensor) -> dict:
    flat = F.normalize(z.reshape(z.size(0), -1).float(), dim=-1)
    sim = flat @ flat.T
    ranks = []
    for i in range(sim.size(0)):
        order = torch.argsort(sim[i], descending=True)
        rank = int((order == i).nonzero()[0].item()) + 1
        ranks.append(rank)
    return {"R@1": sum(r <= 1 for r in ranks) / len(ranks), "R@5": sum(r <= 5 for r in ranks) / len(ranks)}


def eval_q_only(config: dict, tokenizer, model_b, records: list[dict]) -> dict:
    model_b.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(records), config["micro_batch_size"]):
            batch = records[start : start + config["micro_batch_size"]]
            dummy_z = torch.zeros((len(batch), 1, model_b.config.n_embd), device=next(model_b.parameters()).device)
            projector = torch.nn.Identity().to(next(model_b.parameters()).device)
            out = h1_forward(model_b, tokenizer, batch, dummy_z, projector, config["max_cot_tokens"], mode="q")
            losses.append(float(out.loss.item()) * len(batch))
    return {"full_cot_nll": sum(losses) / max(len(records), 1)}


def eval_interventions(config: dict, tokenizer, model_a, model_b, projector, records: list[dict], include_generation: bool) -> dict:
    model_a.eval()
    model_b.eval()
    projector.eval()
    with torch.no_grad():
        z, _ = extract_thinking_hidden(model_a, tokenizer, records, config["max_question_tokens"], 1, "predictor")
        z0 = z[:, 0, :]
        variants = make_latent_variants(z0)
        outputs = {}
        normal_logits = None
        labels = None
        for name, value in variants.items():
            pred = h1_forward(model_b, tokenizer, records, z, projector, config["max_cot_tokens"], latent_override=value, mode="qz")
            if name == "normal":
                normal_logits = pred.logits
                labels = pred.labels
            outputs[name] = {"nll": cot_nll_breakdown(tokenizer, records, pred.logits, pred.labels)}
            if name != "normal":
                outputs[name]["kl_from_normal"] = logits_kl(normal_logits, pred.logits, labels)
    if torch.equal(variants["normal"], variants["shuffled"]):
        raise RuntimeError("normal and shuffled latents are identical")
    outputs["deltas"] = {
        "delta_shuffle_full": outputs["shuffled"]["nll"]["full"] - outputs["normal"]["nll"]["full"],
        "delta_shuffle_first8": outputs["shuffled"]["nll"]["first_8_tokens"] - outputs["normal"]["nll"]["first_8_tokens"],
        "delta_shuffle_numeric": none_sub(outputs["shuffled"]["nll"].get("numeric_tokens"), outputs["normal"]["nll"].get("numeric_tokens")),
        "delta_shuffle_intermediate": none_sub(outputs["shuffled"]["nll"].get("intermediate_tokens"), outputs["normal"]["nll"].get("intermediate_tokens")),
    }
    if include_generation:
        outputs["generation"] = free_generation_metrics(config, tokenizer, model_b, projector, records[: min(12, len(records))], variants)
    return outputs


def none_sub(a, b):
    return None if a is None or b is None else a - b


def free_generation_metrics(config: dict, tokenizer, model_b, projector, records: list[dict], variants: dict[str, torch.Tensor]) -> dict:
    metrics = {name: {"number_match": 0, "intermediate_result_match": 0, "expression_match": 0} for name in ["normal", "shuffled"]}
    lines = []
    device = next(model_b.parameters()).device
    for i, record in enumerate(records):
        prompt_ids, latent_pos = _decoder_prompt_ids(tokenizer, record["question"], "qz")
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        embeds = model_b.get_input_embeddings()(input_ids)
        for name in ["normal", "shuffled"]:
            projected = projector(variants[name][i : i + 1])
            inputs_embeds = torch.cat([embeds[:, :latent_pos, :], projected.unsqueeze(1), embeds[:, latent_pos + 1 :, :]], dim=1)
            generated = model_b.generate(inputs_embeds=inputs_embeds, attention_mask=torch.ones_like(input_ids), max_new_tokens=48, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=False)
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            nums = set(__import__("re").findall(r"-?\d+", text))
            metrics[name]["number_match"] += int(record["answer"] in nums)
            metrics[name]["intermediate_result_match"] += int(any(v in nums for v in record.get("intermediate_results", [])))
            metrics[name]["expression_match"] += int(any(op in text for op in ["+", "-", "*", "/", "="]))
            lines.append(f"id={record['id']} variant={name}\n{text}\n")
    for name in metrics:
        for key in metrics[name]:
            metrics[name][key] /= max(len(records), 1)
    return {"metrics": metrics, "decode_lines": lines}


def summarize(seed_results: dict) -> dict:
    rows = []
    for seed, result in seed_results.items():
        det = result["joint_detach"]
        nod = result["joint_no_detach"]
        rows.append(
            {
                "seed": seed,
                "detach_delta_shuffle": det["interventions"]["deltas"]["delta_shuffle_full"],
                "no_detach_delta_shuffle": nod["interventions"]["deltas"]["delta_shuffle_full"],
                "feedback_gain": nod["interventions"]["deltas"]["delta_shuffle_full"] - det["interventions"]["deltas"]["delta_shuffle_full"],
                "answer_em_diff": nod["validation"]["answer_em"] - det["validation"]["answer_em"],
                "answer_nll_diff": nod["validation"]["answer_nll"] - det["validation"]["answer_nll"],
                "grad_loss1_A_no_detach": nod["gradient_attribution"]["grad_A_from_loss1"],
                "grad_loss1_A_detach": det["gradient_attribution"]["grad_A_from_loss1"],
                "grad_cosine_no_detach": nod["gradient_attribution"]["cosine_main_grad_A_loss1_grad_A"],
            }
        )
    def agg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return {"mean": statistics.mean(vals), "std": statistics.pstdev(vals), "direction_positive": sum(v > 0 for v in vals), "n": len(vals)}
    return {"per_seed_paired": rows, "mean_std": {key: agg(key) for key in rows[0] if key != "seed"}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument("--ood-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-samples", type=int, default=48)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    component_resolution = assert_strict_components(base_config(args.seeds[0], {}, args.steps, args.batch_size, args.eval_samples))
    manifest = {"seeds": args.seeds, "args": vars(args), "strict_components": component_resolution, "status": "running"}
    write_json(OUT / "experiment_manifest.json", manifest)

    seed_results = {}
    gradient_attribution = {}
    latent_interventions = {}
    sample_lines = []
    for seed in args.seeds:
        paths = ensure_data(seed, args.train_size, args.validation_size, args.ood_size)
        config = base_config(seed, paths, args.steps, args.batch_size, args.eval_samples)
        seed_dir = CKPT / f"seed_{seed}"
        tokenizer, model_a0, model_b0, _ = _load_tokenizer_and_model(config, with_b=True)
        assert_prompt_and_labels(tokenizer, model_a0, model_b0, config, read_jsonl(paths["train_path"]))
        del model_a0, model_b0

        s0, tokenizer, model_a = train_s0(config, seed_dir)
        s1, model_b, projector = train_s1(config, tokenizer, model_a, seed_dir)
        s1_path = Path(s1["checkpoint"])
        fork_state = rng_state()
        detach_config = copy.deepcopy(config) | {"detach_encoder_latent": True}
        no_detach_config = copy.deepcopy(config) | {"detach_encoder_latent": False}
        diff = {k: (detach_config.get(k), no_detach_config.get(k)) for k in sorted(set(detach_config) | set(no_detach_config)) if detach_config.get(k) != no_detach_config.get(k)}
        if set(diff) != {"detach_encoder_latent"}:
            raise RuntimeError(f"branch configs differ beyond detach: {diff}")
        joint_detach, *_ = train_joint_branch(detach_config, s1_path, seed_dir, True, fork_state)
        joint_no_detach, *_ = train_joint_branch(no_detach_config, s1_path, seed_dir, False, fork_state)
        if Path(joint_detach["checkpoint"]).exists() is False or Path(joint_no_detach["checkpoint"]).exists() is False:
            raise RuntimeError("missing joint checkpoints")
        seed_results[str(seed)] = {"config": config, "s0": s0, "s1": s1, "joint_detach": joint_detach, "joint_no_detach": joint_no_detach}
        gradient_attribution[str(seed)] = {"joint_detach": joint_detach["gradient_attribution"], "joint_no_detach": joint_no_detach["gradient_attribution"]}
        latent_interventions[str(seed)] = {"s1": s1["interventions"], "joint_detach": joint_detach["interventions"], "joint_no_detach": joint_no_detach["interventions"]}
        for branch in ["joint_detach", "joint_no_detach"]:
            gen = seed_results[str(seed)][branch]["interventions"].get("generation", {})
            sample_lines.extend(gen.get("decode_lines", [])[:8])
        write_json(OUT / "seed_results.json", seed_results)
        write_json(OUT / "gradient_attribution.json", gradient_attribution)
        write_json(OUT / "latent_interventions.json", latent_interventions)

    summary = summarize(seed_results)
    write_json(OUT / "cross_seed_summary.json", summary)
    write_json(OUT / "config_equivalence.json", {"joint_only_diff_is_detach_encoder_latent": True, "checked_per_seed": args.seeds})
    (OUT / "sample_decodes.txt").write_text("\n".join(sample_lines), encoding="utf-8")
    manifest["status"] = "complete"
    write_json(OUT / "experiment_manifest.json", manifest)
    report = f"""# STRICT-HEIMA-DIFFERENCE-EXPERIMENT

Status: complete. Loss2 was not run.

Strict repo remained locked to `thinking_state_mode=predictor`, `{THINKING_TOKEN}`, `HeimaOfficialAbstractProjection`, and official Torchtune chunked CE with `fallback_used=false`.

S0 and S1 completed for seeds {args.seeds}. J-detach and J-no-detach were forked from each seed's same S1 checkpoint and their structured configs differed only by `detach_encoder_latent`.

Loss1-to-A check:
- detach branch grad_A_from_loss1: {[gradient_attribution[str(s)]['joint_detach']['grad_A_from_loss1'] for s in args.seeds]}
- no-detach branch grad_A_from_loss1: {[gradient_attribution[str(s)]['joint_no_detach']['grad_A_from_loss1'] for s in args.seeds]}

Cross-seed paired summary is in `cross_seed_summary.json`; raw per-seed metrics are in `seed_results.json`.
"""
    (OUT / "strict_difference_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "complete", "reports": str(OUT), "seeds": args.seeds}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
