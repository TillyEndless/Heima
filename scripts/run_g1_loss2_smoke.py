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
    causal_leakage_check,
    ensure_sem_token,
    exact_detach_grad_check,
    find_same_question_pairs,
    freeze_teacher,
    loss2_forward,
    loss2_intervention_diagnostics,
    parameter_fingerprint,
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


def _named_param_groups(model_a) -> dict[str, list[torch.nn.Parameter]]:
    block_ids = []
    for name, _param in model_a.named_parameters():
        parts = name.split(".")
        if "h" in parts:
            idx = parts.index("h")
            if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                block_ids.append(int(parts[idx + 1]))
    max_block = max(block_ids) if block_ids else -1
    groups = {"embedding": [], "early_blocks": [], "middle_blocks": [], "late_blocks": []}
    for name, param in model_a.named_parameters():
        parts = name.split(".")
        if any(k in name for k in ("wte", "wpe", "embed_tokens", "embeddings")):
            groups["embedding"].append(param)
            continue
        if "h" in parts:
            idx = parts.index("h")
            if idx + 1 < len(parts) and parts[idx + 1].isdigit() and max_block >= 0:
                block_id = int(parts[idx + 1])
                frac = block_id / max(1, max_block + 1)
                if frac < 1 / 3:
                    groups["early_blocks"].append(param)
                elif frac < 2 / 3:
                    groups["middle_blocks"].append(param)
                else:
                    groups["late_blocks"].append(param)
    return groups


def _grad_vector(params: list[torch.nn.Parameter]) -> torch.Tensor:
    pieces = []
    for param in params:
        if param.grad is None:
            pieces.append(torch.zeros(param.numel(), dtype=torch.float32, device=param.device))
        else:
            pieces.append(param.grad.detach().float().reshape(-1))
    if not pieces:
        return torch.empty(0)
    return torch.cat(pieces)


def _vector_cosine(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if a is None or b is None or a.numel() == 0 or b.numel() == 0:
        return None
    denom = a.norm() * b.norm()
    if float(denom.item()) == 0.0:
        return None
    return float(torch.dot(a, b).div(denom).item())


def _norm_ratio(numerator: float | None, denominator: float | None, scale: float = 1.0) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return float(scale * numerator / denominator)


def _capture_a_group_norms(model_a) -> dict:
    out = {}
    for group, params in _named_param_groups(model_a).items():
        norm, finite = grad_norm(params)
        out[group] = {"norm": norm, "finite": finite}
    return out


def attribution(model_a, model_b_dec, model_b_teacher, tokenizer, records, config):
    out = {}
    lambda1 = float(config.get("lambda_loss1", 0.0))
    lambda2 = float(config.get("lambda_loss2", 0.0))
    a_params = list(model_a.parameters())
    main_vec = None
    loss1_vec = None
    loss2_vec = None

    model_a.zero_grad(set_to_none=True)
    if model_b_dec is not None:
        model_b_dec.zero_grad(set_to_none=True)
    main = main_forward(model_a, tokenizer, records, config["max_question_tokens"], config["max_answer_tokens"])
    main.loss.backward()
    out["grad_norm_A_from_main"], out["finite_A_main"] = grad_norm(model_a.parameters())
    out["grad_groups_A_from_main"] = _capture_a_group_norms(model_a)
    main_vec = _grad_vector(a_params)

    if model_b_dec is not None:
        model_a.zero_grad(set_to_none=True)
        model_b_dec.zero_grad(set_to_none=True)
        main = main_forward(model_a, tokenizer, records, config["max_question_tokens"], config["max_answer_tokens"])
        l1 = loss1_forward(model_b_dec, tokenizer, records, main.z, config["max_cot_tokens"])
        l1.loss.backward()
        out["grad_norm_A_from_loss1"], out["finite_A_loss1"] = grad_norm(model_a.parameters())
        out["grad_norm_B_dec_from_loss1"], out["finite_B_dec_loss1"] = grad_norm(model_b_dec.parameters())
        out["grad_groups_A_from_loss1"] = _capture_a_group_norms(model_a)
        out["lambda1_weighted_grad_A_from_loss1"] = lambda1 * out["grad_norm_A_from_loss1"]
        loss1_vec = _grad_vector(a_params)

    if model_b_dec is not None and model_b_teacher is not None:
        model_a.zero_grad(set_to_none=True)
        model_b_dec.zero_grad(set_to_none=True)
        main = main_forward(model_a, tokenizer, records, config["max_question_tokens"], config["max_answer_tokens"])
        out["exact_detach_test"] = exact_detach_grad_check(
            model_b_dec,
            model_b_teacher,
            tokenizer,
            records,
            main.z,
            config["max_cot_tokens"],
            distance=config["loss2_distance"],
            aggregate=config["loss2_aggregate"],
            layer_index=config["loss2_layer_index"],
            teacher_context_mode=config["teacher_context_mode"],
            feature_mode=config["loss2_feature_mode"],
        )

        model_a.zero_grad(set_to_none=True)
        model_b_dec.zero_grad(set_to_none=True)
        model_b_teacher.zero_grad(set_to_none=True)
        main = main_forward(model_a, tokenizer, records, config["max_question_tokens"], config["max_answer_tokens"])
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
        grad_z = torch.autograd.grad(l2.loss2, main.z, retain_graph=True, allow_unused=True)[0]
        l2.loss2.backward()
        out["grad_norm_A_from_loss2"], out["finite_A_loss2"] = grad_norm(model_a.parameters())
        out["grad_norm_B_dec_from_loss2"], out["finite_B_dec_loss2"] = grad_norm(model_b_dec.parameters())
        out["grad_projector_from_loss2"] = 0.0
        out["finite_projector_loss2"] = True
        out["projector_present"] = False
        out["grad_z_from_loss2"] = 0.0 if grad_z is None else float(grad_z.float().norm().item())
        out["finite_z_loss2"] = True if grad_z is None else bool(torch.isfinite(grad_z).all().item())
        out["grad_groups_A_from_loss2"] = _capture_a_group_norms(model_a)
        out["lambda2_weighted_grad_A_from_loss2"] = lambda2 * out["grad_norm_A_from_loss2"]
        out["lambda2_weighted_grad_B_dec_from_loss2"] = lambda2 * out["grad_norm_B_dec_from_loss2"]
        out["lambda2_weighted_grad_projector_from_loss2"] = 0.0
        out["teacher_grad_count"] = sum(1 for p in model_b_teacher.parameters() if p.grad is not None)

        loss2_vec = _grad_vector(a_params)

    out["cos_main_loss1"] = _vector_cosine(main_vec, loss1_vec)
    out["cos_main_loss2"] = _vector_cosine(main_vec, loss2_vec)
    out["cos_loss1_loss2"] = _vector_cosine(loss1_vec, loss2_vec)
    out["lambda1_norm_loss1_over_main"] = _norm_ratio(out.get("grad_norm_A_from_loss1"), out.get("grad_norm_A_from_main"), lambda1)
    out["lambda2_norm_loss2_over_main"] = _norm_ratio(out.get("grad_norm_A_from_loss2"), out.get("grad_norm_A_from_main"), lambda2)
    return out


def loss2_audit_snapshot(model_a, model_b_dec, model_b_teacher, tokenizer, records, config, *, prefix: str | None = None):
    if model_b_dec is None or model_b_teacher is None:
        return None
    was_training = (model_a.training, model_b_dec.training, model_b_teacher.training)
    model_a.eval()
    model_b_dec.eval()
    model_b_teacher.eval()
    with torch.no_grad():
        main = main_forward(model_a, tokenizer, records, config["max_question_tokens"], config["max_answer_tokens"])
        diag = loss2_intervention_diagnostics(
            model_b_dec,
            model_b_teacher,
            tokenizer,
            records,
            main.z,
            config["max_cot_tokens"],
            distance=config["loss2_distance"],
            aggregate=config["loss2_aggregate"],
            layer_index=config["loss2_layer_index"],
            teacher_context_mode=config["teacher_context_mode"],
            feature_mode=config["loss2_feature_mode"],
        )
    for model, state in zip((model_a, model_b_dec, model_b_teacher), was_training):
        model.train(state)
    metrics = diag.metrics
    if prefix is None:
        return metrics
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def semantic_audits(model_a, model_b_dec, model_b_teacher, tokenizer, records, config) -> dict:
    if model_b_dec is None or model_b_teacher is None:
        return {}
    model_a.eval()
    model_b_dec.eval()
    model_b_teacher.eval()
    with torch.no_grad():
        main = main_forward(model_a, tokenizer, records, config["max_question_tokens"], config["max_answer_tokens"])
    leakage_records = [dict(r) for r in records]
    for idx, rec in enumerate(leakage_records):
        rec["cot"] = f"Future target changed for leakage audit {idx}."
    leakage = causal_leakage_check(
        model_b_dec,
        tokenizer,
        records,
        leakage_records,
        main.z.detach(),
        config["max_cot_tokens"],
        layer_index=config["loss2_layer_index"],
        feature_mode=config["loss2_feature_mode"],
    )
    pairs = find_same_question_pairs(records)
    same_question = {"available": bool(pairs), "num_pairs": len(pairs)}
    if pairs:
        i, j = pairs[0]
        pair_records = [records[i], records[j]]
        with torch.no_grad():
            pair_main = main_forward(model_a, tokenizer, pair_records, config["max_question_tokens"], config["max_answer_tokens"])
            diag = loss2_intervention_diagnostics(
                model_b_dec,
                model_b_teacher,
                tokenizer,
                pair_records,
                pair_main.z,
                config["max_cot_tokens"],
                distance=config["loss2_distance"],
                aggregate=config["loss2_aggregate"],
                layer_index=config["loss2_layer_index"],
                teacher_context_mode=config["teacher_context_mode"],
                feature_mode=config["loss2_feature_mode"],
            )
        same_question.update(
            {
                "loss2_correct": diag.metrics["loss2_normal"],
                "loss2_wrong_same_question": diag.metrics["loss2_shuffle"],
                "wrong_minus_correct": diag.metrics["shuffle_margin"],
            }
        )
    else:
        same_question["reason"] = "current smoke batch has no duplicated question with different CoT section"
    return {"causal_leakage": leakage, "same_question_paired": same_question}


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
    initial_records = batch_records(train, config["micro_batch_size"], 0)
    initial_audit = loss2_audit_snapshot(model_a, model_b_dec, model_b_teacher, tokenizer, initial_records, config)
    if initial_audit is not None:
        initial_audit = {
            **initial_audit,
            "loss2_normal_initial": initial_audit["loss2_normal"],
            "loss2_shuffle_initial": initial_audit["loss2_shuffle"],
            "loss2_zero_initial": initial_audit["loss2_zero"],
            "loss2_random_initial": initial_audit["loss2_random"],
        }
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
        step_audit = loss2_audit_snapshot(model_a, model_b_dec, model_b_teacher, tokenizer, records, config)
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
                "loss2_semantic_audit": step_audit,
            }
        )
    eval_records = read_jsonl(config["validation_path"])[: config["eval_samples"]]
    interventions = loss2_audit_snapshot(model_a, model_b_dec, model_b_teacher, tokenizer, eval_records, config)
    semantic = semantic_audits(model_a, model_b_dec, model_b_teacher, tokenizer, eval_records, config)
    teacher_hash_after = parameter_fingerprint(model_b_teacher) if model_b_teacher is not None else None
    result = {
        "group": group,
        "config": config,
        "logs": logs,
        "gradient_attribution": attr,
        "loss2_initial_audit": initial_audit,
        "loss2_interventions": interventions,
        "semantic_audits": semantic,
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
