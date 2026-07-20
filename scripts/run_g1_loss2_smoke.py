#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.g1.gradient_monitor import cosine, grad_norm, tensor_norm  # noqa: E402
from src.g1.latent_reasoner import assert_parameter_independence, main_forward  # noqa: E402
from src.g1.loss2_teacher import (  # noqa: E402
    assert_student_teacher_independent,
    assert_teacher_frozen_and_excluded,
    ensure_sem_token,
    freeze_teacher,
    loss2_forward,
    parameter_fingerprint,
    student_feature_forward,
    teacher_feature_forward,
)
from src.g1.synthetic_data import read_jsonl  # noqa: E402
from src.g1.whole_cot_decoder import ensure_latent_token, loss1_forward  # noqa: E402


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_records(records: list[dict], batch_size: int, step: int) -> list[dict]:
    start = (step * batch_size) % len(records)
    return (records + records)[start : start + batch_size]


def load_models(config: dict, *, need_b_dec: bool, need_teacher: bool):
    model_name_or_path = config.get("model_name_or_path", config.get("model_name"))
    load_kwargs = {"local_files_only": bool(config.get("local_files_only", False))}
    if "use_safetensors" in config:
        load_kwargs["use_safetensors"] = bool(config["use_safetensors"])
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **load_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    model_a = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    model_b_dec = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs) if need_b_dec else None
    model_b_teacher = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs) if need_teacher else None
    ensure_latent_token(tokenizer, *[m for m in (model_a, model_b_dec, model_b_teacher) if m is not None])
    if need_teacher:
        ensure_sem_token(tokenizer, *[m for m in (model_b_dec, model_b_teacher) if m is not None])
    if model_b_dec is not None:
        assert_parameter_independence(model_a, model_b_dec)
    if model_b_dec is not None and model_b_teacher is not None:
        assert_student_teacher_independent(model_b_dec, model_b_teacher)
        freeze_teacher(model_b_teacher)
    for model in (model_a, model_b_dec, model_b_teacher):
        if model is not None:
            model.config.use_cache = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a.to(device)
    if model_b_dec is not None:
        model_b_dec.to(device)
    if model_b_teacher is not None:
        model_b_teacher.to(device)
        freeze_teacher(model_b_teacher)
    return tokenizer, model_a, model_b_dec, model_b_teacher, device


def attribution(model_a, model_b_dec, model_b_teacher, tokenizer, records, config):
    out = {}
    model_a.zero_grad(set_to_none=True)
    if model_b_dec is not None:
        model_b_dec.zero_grad(set_to_none=True)
    main = main_forward(model_a, tokenizer, records, config["max_question_tokens"], config["max_answer_tokens"])
    main.loss.backward()
    out["grad_norm_A_from_main"], out["finite_A_main"] = grad_norm(model_a.parameters())

    if model_b_dec is not None:
        model_a.zero_grad(set_to_none=True)
        model_b_dec.zero_grad(set_to_none=True)
        main = main_forward(model_a, tokenizer, records, config["max_question_tokens"], config["max_answer_tokens"])
        l1 = loss1_forward(model_b_dec, tokenizer, records, main.z, config["max_cot_tokens"])
        l1.loss.backward()
        out["grad_norm_A_from_loss1"], out["finite_A_loss1"] = grad_norm(model_a.parameters())
        out["grad_norm_B_dec_from_loss1"], out["finite_B_dec_loss1"] = grad_norm(model_b_dec.parameters())

    if model_b_dec is not None and model_b_teacher is not None:
        model_a.zero_grad(set_to_none=True)
        model_b_dec.zero_grad(set_to_none=True)
        model_b_teacher.zero_grad(set_to_none=True)
        main = main_forward(model_a, tokenizer, records, config["max_question_tokens"], config["max_answer_tokens"])
        student, teacher, l2 = loss2_forward(
            model_b_dec,
            model_b_teacher,
            tokenizer,
            records,
            main.z,
            config["max_cot_tokens"],
            distance=config["loss2_distance"],
            aggregate=config["loss2_aggregate"],
            layer_index=config["loss2_layer_index"],
            detach_latent=config["loss2_detach_latent"],
            teacher_context_mode=config["teacher_context_mode"],
            feature_mode=config["loss2_feature_mode"],
        )
        del student, teacher
        l2.loss2.backward()
        out["grad_norm_A_from_loss2"], out["finite_A_loss2"] = grad_norm(model_a.parameters())
        out["grad_norm_B_dec_from_loss2"], out["finite_B_dec_loss2"] = grad_norm(model_b_dec.parameters())
        out["teacher_grad_count"] = sum(1 for p in model_b_teacher.parameters() if p.grad is not None)
    return out


@torch.no_grad()
def loss2_interventions(model_a, model_b_dec, model_b_teacher, tokenizer, records, config):
    if model_b_dec is None or model_b_teacher is None:
        return None
    main = main_forward(model_a, tokenizer, records, config["max_question_tokens"], config["max_answer_tokens"])
    z = main.z
    rand = torch.randn_like(z)
    rand = rand / rand.float().norm(dim=-1, keepdim=True).clamp_min(1e-6) * z.float().norm(dim=-1, keepdim=True).to(rand.dtype)
    variants = {
        "normal": z,
        "shuffle": torch.roll(z, shifts=1, dims=0) if z.shape[0] > 1 else torch.zeros_like(z),
        "zero": torch.zeros_like(z),
        "random": rand,
    }
    out = {}
    teacher = teacher_feature_forward(
        model_b_teacher,
        tokenizer,
        records,
        layer_index=config["loss2_layer_index"],
        context_mode=config["teacher_context_mode"],
    )
    for name, value in variants.items():
        student = student_feature_forward(
            model_b_dec,
            tokenizer,
            records,
            value,
            config["max_cot_tokens"],
            layer_index=config["loss2_layer_index"],
            feature_mode=config["loss2_feature_mode"],
        )
        _s, _t, l2 = loss2_forward(
            model_b_dec,
            model_b_teacher,
            tokenizer,
            records,
            value,
            config["max_cot_tokens"],
            distance=config["loss2_distance"],
            aggregate=config["loss2_aggregate"],
            layer_index=config["loss2_layer_index"],
            teacher_context_mode=config["teacher_context_mode"],
            feature_mode=config["loss2_feature_mode"],
        )
        out[f"loss2_{name}"] = float(l2.loss2.item())
        out[f"h_l_variance_{name}"] = float(student.h_l.float().var(dim=0).mean().item()) if student.h_l.shape[0] > 1 else 0.0
    out["h_t_variance"] = float(teacher.h_t.float().var(dim=0).mean().item()) if teacher.h_t.shape[0] > 1 else 0.0
    out["shuffle_margin"] = out["loss2_shuffle"] - out["loss2_normal"]
    out["zero_margin"] = out["loss2_zero"] - out["loss2_normal"]
    out["random_margin"] = out["loss2_random"] - out["loss2_normal"]
    return out


def run_group(base_config: dict, group: str, overrides: dict, out_dir: Path) -> dict:
    config = copy.deepcopy(base_config)
    config.update(overrides)
    config.setdefault("lambda_loss1", config.get("lambda1", 0.0))
    config.setdefault("lambda_loss2", 0.0)
    config.setdefault("loss2_distance", "cosine")
    config.setdefault("loss2_aggregate", "mean")
    config.setdefault("loss2_layer_index", -1)
    config.setdefault("loss2_detach_latent", False)
    config.setdefault("loss2_feature_mode", "pre_sem")
    config.setdefault("teacher_context_mode", "cumulative")
    need_b_dec = config["lambda_loss1"] > 0 or config["lambda_loss2"] > 0
    need_teacher = config["lambda_loss2"] > 0
    set_seed(config["seed"])
    tokenizer, model_a, model_b_dec, model_b_teacher, _device = load_models(config, need_b_dec=need_b_dec, need_teacher=need_teacher)
    train = read_jsonl(config["train_path"])
    params = list(model_a.parameters())
    if model_b_dec is not None:
        params += list(model_b_dec.parameters())
    opt = torch.optim.AdamW(params, lr=config["learning_rate_A"], weight_decay=config["weight_decay"])
    if model_b_teacher is not None:
        assert_teacher_frozen_and_excluded(model_b_teacher, opt)
    teacher_hash_before = parameter_fingerprint(model_b_teacher) if model_b_teacher is not None else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    logs = []
    start = time.time()
    attr = attribution(model_a, model_b_dec, model_b_teacher, tokenizer, batch_records(train, config["micro_batch_size"], 0), config)
    for step in range(1, config["max_steps"] + 1):
        records = batch_records(train, config["micro_batch_size"], step - 1)
        model_a.train()
        if model_b_dec is not None:
            model_b_dec.train()
        if model_b_teacher is not None:
            model_b_teacher.eval()
        opt.zero_grad(set_to_none=True)
        main = main_forward(model_a, tokenizer, records, config["max_question_tokens"], config["max_answer_tokens"])
        loss1 = torch.zeros((), dtype=main.loss.dtype, device=main.loss.device)
        loss2 = torch.zeros((), dtype=main.loss.dtype, device=main.loss.device)
        if model_b_dec is not None and config["lambda_loss1"] > 0:
            loss1 = loss1_forward(model_b_dec, tokenizer, records, main.z, config["max_cot_tokens"]).loss
        if model_b_dec is not None and model_b_teacher is not None and config["lambda_loss2"] > 0:
            _student, _teacher, l2 = loss2_forward(
                model_b_dec,
                model_b_teacher,
                tokenizer,
                records,
                main.z,
                config["max_cot_tokens"],
                distance=config["loss2_distance"],
                aggregate=config["loss2_aggregate"],
                layer_index=config["loss2_layer_index"],
                detach_latent=config["loss2_detach_latent"],
                teacher_context_mode=config["teacher_context_mode"],
                feature_mode=config["loss2_feature_mode"],
            )
            loss2 = l2.loss2
        total = main.loss + config["lambda_loss1"] * loss1 + config["lambda_loss2"] * loss2
        total.backward()
        a_grad, a_finite = grad_norm(model_a.parameters())
        b_grad, b_finite = grad_norm(model_b_dec.parameters()) if model_b_dec is not None else (0.0, True)
        torch.nn.utils.clip_grad_norm_(params, config["max_grad_norm"])
        opt.step()
        logs.append(
            {
                "step": step,
                "loss_main": float(main.loss.item()),
                "loss1": float(loss1.item()),
                "loss2": float(loss2.item()),
                "loss_total": float(total.item()),
                "grad_norm_A": a_grad,
                "grad_norm_B_dec": b_grad,
                "finite_A": a_finite,
                "finite_B_dec": b_finite,
            }
        )
    eval_records = read_jsonl(config["validation_path"])[: config["eval_samples"]]
    interventions = loss2_interventions(model_a, model_b_dec, model_b_teacher, tokenizer, eval_records, config)
    teacher_hash_after = parameter_fingerprint(model_b_teacher) if model_b_teacher is not None else None
    result = {
        "group": group,
        "config": config,
        "logs": logs,
        "gradient_attribution": attr,
        "loss2_interventions": interventions,
        "teacher_hash_before": teacher_hash_before,
        "teacher_hash_after": teacher_hash_after,
        "teacher_unchanged": teacher_hash_before == teacher_hash_after,
        "teacher_grad_count": sum(1 for p in model_b_teacher.parameters() if p.grad is not None) if model_b_teacher is not None else 0,
        "runtime_sec": time.time() - start,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0,
    }
    write_json(out_dir / group / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/g1_gpt2/configs/main_l1.yaml")
    parser.add_argument("--out", default="experiments/g1_gpt2/reports/loss2_smoke")
    parser.add_argument("--model-name-or-path", default=None)
    args = parser.parse_args()
    base = load_config(Path(args.config))
    if args.model_name_or_path:
        base["model_name_or_path"] = args.model_name_or_path
    base.update({"max_steps": 2, "micro_batch_size": 2, "gradient_accumulation_steps": 1, "eval_samples": 4})
    out_dir = Path(args.out)
    groups = {
        "main_only": {"lambda_loss1": 0.0, "lambda_loss2": 0.0, "lambda1": 0.0},
        "main_loss1": {"lambda_loss1": 0.1, "lambda_loss2": 0.0, "lambda1": 0.1},
        "main_loss2": {"lambda_loss1": 0.0, "lambda_loss2": 0.1, "lambda1": 0.0},
        "main_loss1_loss2": {"lambda_loss1": 0.1, "lambda_loss2": 0.1, "lambda1": 0.1},
    }
    manifest = {"started_at": time.time(), "base_config": base, "groups": groups}
    write_json(out_dir / "manifest.json", manifest)
    results = {name: run_group(base, name, overrides, out_dir) for name, overrides in groups.items()}
    write_json(out_dir / "summary.json", results)
    print(json.dumps({k: v["logs"][-1] for k, v in results.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
