from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from .evaluator import evaluate_interventions, evaluate_main, sample_decodes
from .gradient_monitor import cosine, finite_ratio, grad_norm, tensor_norm
from .latent_reasoner import assert_parameter_independence, main_forward
from .synthetic_data import read_jsonl
from .whole_cot_decoder import ensure_latent_token, loss1_forward


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_models(config: dict):
    model_name_or_path = config.get("model_name_or_path", config.get("model_name"))
    load_kwargs = {
        "local_files_only": bool(config.get("local_files_only", False)),
    }
    if "use_safetensors" in config:
        load_kwargs["use_safetensors"] = bool(config["use_safetensors"])

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **load_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    model_a = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    model_b = None
    if config["lambda1"] > 0:
        model_b = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        ensure_latent_token(tokenizer, model_a, model_b)
        assert_parameter_independence(model_a, model_b)
    else:
        ensure_latent_token(tokenizer, model_a)
    model_a.config.use_cache = False
    if model_b is not None:
        model_b.config.use_cache = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a.to(device)
    if model_b is not None:
        model_b.to(device)
    return tokenizer, model_a, model_b, device


def batch_records(records: list[dict], batch_size: int, step: int) -> list[dict]:
    start = (step * batch_size) % len(records)
    doubled = records + records
    return doubled[start : start + batch_size]


def train(config_path: str | Path, overrides: dict | None = None) -> dict:
    config = load_config(config_path)
    if overrides:
        config.update(overrides)
    set_seed(config["seed"])
    tokenizer, model_a, model_b, device = load_models(config)
    train_records = read_jsonl(config["train_path"])
    val_records = read_jsonl(config["validation_path"])

    params = list(model_a.parameters())
    if model_b is not None:
        params.extend(model_b.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=config["learning_rate_A"],
        weight_decay=config["weight_decay"],
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    logs = []
    warnings = []
    max_steps = config["max_steps"]
    micro = config["micro_batch_size"]
    accum = config["gradient_accumulation_steps"]
    last_z = None
    initial_l1 = None
    final_l1 = None
    start_time = time.time()

    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        audit = {}
        step_start = time.time()
        for acc in range(accum):
            records = batch_records(train_records, micro, (step - 1) * accum + acc)
            model_a.train()
            if model_b is not None:
                model_b.train()
            main = main_forward(
                model_a,
                tokenizer,
                records,
                config["max_question_tokens"],
                config["max_answer_tokens"],
            )
            loss_main = main.loss
            loss1 = torch.zeros((), dtype=loss_main.dtype, device=device)
            g_l1 = None
            if model_b is not None and config["lambda1"] > 0:
                dec = loss1_forward(
                    model_b,
                    tokenizer,
                    records,
                    main.z,
                    config["max_cot_tokens"],
                )
                loss1 = dec.loss
                if initial_l1 is None:
                    initial_l1 = loss1.item()
                final_l1 = loss1.item()
                g_l1 = torch.autograd.grad(loss1, main.z, retain_graph=True)[0]
            g_main = torch.autograd.grad(loss_main, main.z, retain_graph=True)[0]
            total = loss_main + config["lambda1"] * loss1
            (total / accum).backward()
            last_z = main.z
            step_losses.append((loss_main.item(), loss1.item(), total.item()))
            audit = {
                "g_main_norm": tensor_norm(g_main),
                "g_l1_norm": tensor_norm(g_l1),
                "g_main_l1_cosine": cosine(g_main, g_l1),
                "z_grad_norm": tensor_norm(main.z.grad),
                "finite_ratio": finite_ratio([loss_main, loss1, total, main.z]),
            }
        torch.nn.utils.clip_grad_norm_(params, config["max_grad_norm"])
        a_grad, a_finite = grad_norm(model_a.parameters())
        b_grad, b_finite = grad_norm(model_b.parameters()) if model_b is not None else (0.0, True)
        optimizer.step()

        if step % config["log_interval"] == 0 or step == 1:
            avg_main = sum(x[0] for x in step_losses) / len(step_losses)
            avg_l1 = sum(x[1] for x in step_losses) / len(step_losses)
            avg_total = sum(x[2] for x in step_losses) / len(step_losses)
            if audit["g_l1_norm"] and audit["g_main_norm"] and audit["g_l1_norm"] > 10 * audit["g_main_norm"]:
                warnings.append({"step": step, "warning": "l1_grad_gt_10x_main_grad"})
            logs.append(
                {
                    "step": step,
                    "loss_main": avg_main,
                    "loss1": avg_l1,
                    "total_loss": avg_total,
                    "model_a_grad_norm": a_grad,
                    "model_b_grad_norm": b_grad,
                    "model_a_grad_finite": a_finite,
                    "model_b_grad_finite": b_finite,
                    "all_finite": bool(a_finite and b_finite and audit["finite_ratio"] == 1.0),
                    "peak_gpu_memory_mb": (
                        torch.cuda.max_memory_allocated() / 1024 / 1024
                        if torch.cuda.is_available()
                        else 0.0
                    ),
                    "step_time_sec": time.time() - step_start,
                    **audit,
                    "norm_g_main": audit["g_main_norm"],
                    "norm_g_l1": audit["g_l1_norm"],
                    "cosine_g_main_g_l1": audit["g_main_l1_cosine"],
                    "norm_g_l1_over_g_main": (
                        audit["g_l1_norm"] / audit["g_main_norm"]
                        if audit["g_main_norm"]
                        else None
                    ),
                }
            )

    eval_records = val_records[: config["eval_samples"]]
    main_eval = evaluate_main(model_a, tokenizer, eval_records, config)
    interventions = evaluate_interventions(model_a, model_b, tokenizer, eval_records, config)
    sample_text = sample_decodes(model_a, model_b, tokenizer, eval_records, config) if model_b is not None else None

    report = {
        "status": "pass" if not warnings else "warning",
        "experiment": config["experiment_name"],
        "config": config,
        "initial_l1": initial_l1,
        "final_l1": final_l1,
        "logs": logs,
        "eval": main_eval,
        "interventions": interventions,
        "sample_cot_decode": sample_text,
        "warnings": warnings,
        "stop_gates_triggered": [],
        "runtime_sec": time.time() - start_time,
    }
    out_path = Path(config["report_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if config.get("sample_decode_path") and sample_text:
        Path(config["sample_decode_path"]).write_text(sample_text, encoding="utf-8")
    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_a": model_a.state_dict()}, ckpt_dir / "model_a.pt")
    if model_b is not None:
        torch.save({"model_b": model_b.state_dict()}, ckpt_dir / "model_b.pt")
    return report
