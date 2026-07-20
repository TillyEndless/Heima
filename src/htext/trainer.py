from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from .heima_reuse import HeimaOfficialAbstractProjection, prepare_latent_for_decoder
from .modeling import (
    LatentProjector,
    extract_thinking_hidden,
    h0_forward,
    h1_forward,
    setup_special_tokens,
)
from .synthetic_data import read_jsonl


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_records(records: list[dict], batch_size: int, step: int) -> list[dict]:
    start = (step * batch_size) % len(records)
    doubled = records + records
    return doubled[start : start + batch_size]


def _load_tokenizer_and_model(config: dict, with_b: bool = False):
    model_name = config["model_name_or_path"]
    load_kwargs = {
        "local_files_only": bool(config.get("local_files_only", False)),
    }
    if "use_safetensors" in config:
        load_kwargs["use_safetensors"] = bool(config["use_safetensors"])
    if config.get("torch_dtype"):
        dtype_name = str(config["torch_dtype"])
        if not hasattr(torch, dtype_name):
            raise ValueError(f"unsupported torch_dtype: {dtype_name}")
        load_kwargs["torch_dtype"] = getattr(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    model_a = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model_b = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs) if with_b else None
    for model in [m for m in [model_a, model_b] if m is not None]:
        if not hasattr(model.config, "n_embd") and hasattr(model.config, "hidden_size"):
            model.config.n_embd = model.config.hidden_size
    setup_special_tokens(tokenizer, *([model_a, model_b] if model_b is not None else [model_a]))
    model_a.config.use_cache = False
    if model_b is not None:
        model_b.config.use_cache = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a.to(device)
    if model_b is not None:
        model_b.to(device)
    return tokenizer, model_a, model_b, device


def _grad_norm(params) -> tuple[float, bool]:
    total = 0.0
    finite = True
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        finite = finite and torch.isfinite(g).all().item()
        total += float(g.float().pow(2).sum().item())
    return total**0.5, bool(finite)


def _answer_em(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    shifted_pred = pred[:, :-1]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    rows = []
    for i in range(labels.size(0)):
        row_mask = mask[i]
        if row_mask.any():
            rows.append(torch.equal(shifted_pred[i][row_mask], shifted_labels[i][row_mask]))
    return sum(bool(x) for x in rows) / max(len(rows), 1)


def _save_model_a(path: Path, model_a) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"model_a": model_a.state_dict()}, path / "model_a.pt")


def _load_model_a_checkpoint(path: str | Path, model_a) -> None:
    ckpt = torch.load(Path(path) / "model_a.pt", map_location="cpu")
    missing, unexpected = model_a.load_state_dict(ckpt["model_a"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected model_a checkpoint keys: {unexpected[:8]}")
    if missing:
        raise RuntimeError(f"missing model_a checkpoint keys: {missing[:8]}")


def _thinking_state_mode(config: dict) -> str:
    if "heima_shifted_hidden" in config:
        raise ValueError("heima_shifted_hidden is deprecated; use thinking_state_mode: predictor|token")
    mode = config.get("thinking_state_mode", "predictor")
    if mode not in {"predictor", "token"}:
        raise ValueError(f"unsupported thinking_state_mode: {mode}")
    return mode


def _build_projector(config: dict, hidden_size: int, *, strict_repo: bool = False) -> torch.nn.Module:
    compatibility = config.get("heima_compatibility")
    if strict_repo or compatibility == "strict_repo":
        projector_name = config.get("projector", "heima_official")
        if projector_name != "heima_official":
            raise ValueError("strict_repo only allows projector: heima_official")
        return HeimaOfficialAbstractProjection(hidden_size, hidden_size)
    return LatentProjector(hidden_size)


def train_h0(config_path: str | Path, overrides: dict | None = None) -> dict:
    config = load_config(config_path)
    if overrides:
        config.update(overrides)
    set_seed(config["seed"])
    tokenizer, model_a, _, device = _load_tokenizer_and_model(config, with_b=False)
    train_records = read_jsonl(config["train_path"])
    val_records = read_jsonl(config["validation_path"])
    optimizer = torch.optim.AdamW(
        model_a.parameters(),
        lr=config["learning_rate_A"],
        weight_decay=config["weight_decay"],
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    logs = []
    start = time.time()
    for step in range(1, config["max_steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for acc in range(config["gradient_accumulation_steps"]):
            records = batch_records(
                train_records,
                config["micro_batch_size"],
                (step - 1) * config["gradient_accumulation_steps"] + acc,
            )
            model_a.train()
            out = h0_forward(
                model_a,
                tokenizer,
                records,
                config["max_question_tokens"],
                config["max_answer_tokens"],
                config["num_thinking_tokens"],
                _thinking_state_mode(config),
            )
            (out.loss / config["gradient_accumulation_steps"]).backward()
            losses.append(float(out.loss.item()))
        torch.nn.utils.clip_grad_norm_(model_a.parameters(), config["max_grad_norm"])
        grad, finite = _grad_norm(model_a.parameters())
        optimizer.step()
        if step == 1 or step % config["log_interval"] == 0:
            logs.append(
                {
                    "step": step,
                    "loss_main": sum(losses) / len(losses),
                    "model_a_grad_norm": grad,
                    "model_a_grad_finite": finite,
                    "peak_gpu_memory_mb": (
                        torch.cuda.max_memory_allocated() / 1024 / 1024
                        if torch.cuda.is_available()
                        else 0.0
                    ),
                }
            )
    eval_records = val_records[: config["eval_samples"]]
    model_a.eval()
    with torch.no_grad():
        out = h0_forward(
            model_a,
            tokenizer,
            eval_records,
            config["max_question_tokens"],
            config["max_answer_tokens"],
            config["num_thinking_tokens"],
            _thinking_state_mode(config),
        )
    report = {
        "status": "pass",
        "stage": "H0",
        "experiment": config["experiment_name"],
        "config": config,
        "logs": logs,
        "eval": {"main_nll": float(out.loss.item()), "token_em": _answer_em(out.logits, out.labels)},
        "runtime_sec": time.time() - start,
        "stop_gates_triggered": [],
    }
    Path(config["report_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(config["report_path"]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _save_model_a(Path(config["checkpoint_dir"]), model_a)
    return report


def _evaluate_h1(model_a, model_b, projector, tokenizer, records: list[dict], config: dict) -> dict:
    model_a.eval()
    model_b.eval()
    projector.eval()
    with torch.no_grad():
        z, _ = extract_thinking_hidden(
            model_a,
            tokenizer,
            records,
            config["max_question_tokens"],
            config["num_thinking_tokens"],
            _thinking_state_mode(config),
        )
        mode = config.get("h1_mode", "qz")
        normal = h1_forward(model_b, tokenizer, records, z, projector, config["max_cot_tokens"], mode=mode)
        zero = h1_forward(
            model_b,
            tokenizer,
            records,
            z,
            projector,
            config["max_cot_tokens"],
            latent_override=torch.zeros_like(z[:, 0, :]),
            mode=mode,
        )
        shuffled = h1_forward(
            model_b,
            tokenizer,
            records,
            z,
            projector,
            config["max_cot_tokens"],
            latent_override=torch.roll(z[:, 0, :], shifts=-1, dims=0),
            mode=mode,
        )
    return {
        "normal_nll": float(normal.loss.item()),
        "zero_nll": float(zero.loss.item()),
        "shuffled_nll": float(shuffled.loss.item()),
        "normal_minus_shuffled": float(normal.loss.item() - shuffled.loss.item()),
        "normal_minus_zero": float(normal.loss.item() - zero.loss.item()),
    }


def train_h1(config_path: str | Path, overrides: dict | None = None) -> dict:
    config = load_config(config_path)
    if overrides:
        config.update(overrides)
    set_seed(config["seed"])
    tokenizer, model_a, model_b, device = _load_tokenizer_and_model(config, with_b=True)
    _load_model_a_checkpoint(config["h0_checkpoint_dir"], model_a)
    for p in model_a.parameters():
        p.requires_grad_(False)
    model_a.eval()
    projector = _build_projector(config, model_a.config.n_embd).to(device)
    mode = config.get("h1_mode", "qz")
    params = list(model_b.parameters()) + ([] if mode == "q" else list(projector.parameters()))
    param_groups = [{"params": model_b.parameters(), "lr": config["learning_rate_B"]}]
    if mode != "q":
        param_groups.append({"params": projector.parameters(), "lr": config["learning_rate_projector"]})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_records = read_jsonl(config["train_path"])
    val_records = read_jsonl(config["validation_path"])
    logs = []
    start = time.time()
    for step in range(1, config["max_steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        z_requires_grad = None
        for acc in range(config["gradient_accumulation_steps"]):
            records = batch_records(
                train_records,
                config["micro_batch_size"],
                (step - 1) * config["gradient_accumulation_steps"] + acc,
            )
            model_b.train()
            projector.train()
            z, _ = extract_thinking_hidden(
                model_a,
                tokenizer,
                records,
                config["max_question_tokens"],
                config["num_thinking_tokens"],
                _thinking_state_mode(config),
            )
            z = z.detach()
            z_requires_grad = bool(z.requires_grad)
            out = h1_forward(model_b, tokenizer, records, z, projector, config["max_cot_tokens"], mode=mode)
            (out.loss / config["gradient_accumulation_steps"]).backward()
            losses.append(float(out.loss.item()))
        torch.nn.utils.clip_grad_norm_(params, config["max_grad_norm"])
        b_grad, b_finite = _grad_norm(model_b.parameters())
        p_grad, p_finite = _grad_norm(projector.parameters())
        a_grad, _ = _grad_norm(model_a.parameters())
        optimizer.step()
        if step == 1 or step % config["log_interval"] == 0:
            logs.append(
                {
                    "step": step,
                    "loss1": sum(losses) / len(losses),
                    "model_a_frozen_grad_norm": a_grad,
                    "model_b_grad_norm": b_grad,
                    "projector_grad_norm": p_grad,
                    "model_b_grad_finite": b_finite,
                    "projector_grad_finite": p_finite,
                    "latent_requires_grad": z_requires_grad,
                    "peak_gpu_memory_mb": (
                        torch.cuda.max_memory_allocated() / 1024 / 1024
                        if torch.cuda.is_available()
                        else 0.0
                    ),
                }
            )
    eval_records = val_records[: config["eval_samples"]]
    eval_report = _evaluate_h1(model_a, model_b, projector, tokenizer, eval_records, config)
    report = {
        "status": "pass",
        "stage": "H1",
        "experiment": config["experiment_name"],
        "config": config,
        "logs": logs,
        "eval": eval_report,
        "runtime_sec": time.time() - start,
        "stop_gates_triggered": [],
    }
    Path(config["report_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(config["report_path"]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_b": model_b.state_dict(), "projector": projector.state_dict()}, ckpt_dir / "h1.pt")
    return report


def train_joint_main_l1(config_path: str | Path, overrides: dict | None = None) -> dict:
    config = load_config(config_path)
    if overrides:
        config.update(overrides)
    set_seed(config["seed"])
    tokenizer, model_a, model_b, device = _load_tokenizer_and_model(config, with_b=True)
    if config.get("heima_compatibility") != "strict_repo":
        raise ValueError("train_joint_main_l1 strict path requires heima_compatibility: strict_repo")
    if config.get("h1_mode", "qz") != "qz":
        raise ValueError("strict_repo requires h1_mode='qz'")
    if config.get("load_legacy_checkpoint", False):
        raise ValueError("strict_repo formal experiments must not load legacy checkpoints")
    projector = _build_projector(config, model_a.config.n_embd, strict_repo=True).to(device)
    mode = config.get("h1_mode", "qz")
    detach_encoder_latent = bool(config.get("detach_encoder_latent", False))
    if mode != "qz":
        raise ValueError("Heima-lite Main+Loss1 alignment requires h1_mode='qz'")
    train_records = read_jsonl(config["train_path"])
    val_records = read_jsonl(config["validation_path"])
    params = list(model_a.parameters()) + list(model_b.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": model_a.parameters(), "lr": config["learning_rate_A"]},
            {"params": model_b.parameters(), "lr": config["learning_rate_B"]},
            {"params": projector.parameters(), "lr": config["learning_rate_projector"]},
        ],
        weight_decay=config["weight_decay"],
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    logs = []
    start = time.time()
    for step in range(1, config["max_steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        audit = {}
        for acc in range(config["gradient_accumulation_steps"]):
            records = batch_records(
                train_records,
                config["micro_batch_size"],
                (step - 1) * config["gradient_accumulation_steps"] + acc,
            )
            model_a.train()
            model_b.train()
            projector.train()
            main = h0_forward(
                model_a,
                tokenizer,
                records,
                config["max_question_tokens"],
                config["max_answer_tokens"],
                config["num_thinking_tokens"],
                _thinking_state_mode(config),
            )
            if not main.thinking_hidden.requires_grad:
                raise RuntimeError("thinking hidden must require grad in joint Main+Loss1 training")
            decoder_z = prepare_latent_for_decoder(main.thinking_hidden, detach_encoder_latent)
            loss1 = h1_forward(
                model_b,
                tokenizer,
                records,
                decoder_z,
                projector,
                config["max_cot_tokens"],
                mode="qz",
            )
            g_l1 = (
                None
                if detach_encoder_latent
                else torch.autograd.grad(loss1.loss, main.thinking_hidden, retain_graph=True)[0]
            )
            g_main = torch.autograd.grad(main.loss, main.thinking_hidden, retain_graph=True, allow_unused=True)[0]
            total = main.loss + config["lambda1"] * loss1.loss
            (total / config["gradient_accumulation_steps"]).backward()
            step_losses.append((float(main.loss.item()), float(loss1.loss.item()), float(total.item())))
            audit = {
                "latent_requires_grad": bool(main.thinking_hidden.requires_grad),
                "detach_encoder_latent": detach_encoder_latent,
                "g_l1_to_z_norm": float(g_l1.detach().float().norm().item()) if g_l1 is not None else 0.0,
                "g_main_to_z_norm": float(g_main.detach().float().norm().item()) if g_main is not None else 0.0,
            }
        torch.nn.utils.clip_grad_norm_(params, config["max_grad_norm"])
        a_grad, a_finite = _grad_norm(model_a.parameters())
        b_grad, b_finite = _grad_norm(model_b.parameters())
        p_grad, p_finite = _grad_norm(projector.parameters())
        optimizer.step()
        if step == 1 or step % config["log_interval"] == 0 or step == config["max_steps"]:
            avg_main = sum(x[0] for x in step_losses) / len(step_losses)
            avg_l1 = sum(x[1] for x in step_losses) / len(step_losses)
            avg_total = sum(x[2] for x in step_losses) / len(step_losses)
            logs.append(
                {
                    "step": step,
                    "loss_main": avg_main,
                    "loss1": avg_l1,
                    "total_loss": avg_total,
                    "model_a_grad_norm": a_grad,
                    "model_b_grad_norm": b_grad,
                    "projector_grad_norm": p_grad,
                    "model_a_grad_finite": a_finite,
                    "model_b_grad_finite": b_finite,
                    "projector_grad_finite": p_finite,
                    "all_finite": bool(a_finite and b_finite and p_finite),
                    "peak_gpu_memory_mb": (
                        torch.cuda.max_memory_allocated() / 1024 / 1024
                        if torch.cuda.is_available()
                        else 0.0
                    ),
                    **audit,
                }
            )
    eval_records = val_records[: config["eval_samples"]]
    model_a.eval()
    model_b.eval()
    projector.eval()
    with torch.no_grad():
        main_eval = h0_forward(
            model_a,
            tokenizer,
            eval_records,
            config["max_question_tokens"],
            config["max_answer_tokens"],
            config["num_thinking_tokens"],
            _thinking_state_mode(config),
        )
        loss1_eval = h1_forward(
            model_b,
            tokenizer,
            eval_records,
            main_eval.thinking_hidden,
            projector,
            config["max_cot_tokens"],
            mode="qz",
        )
    report = {
        "status": "pass",
        "stage": "HEIMA_LITE_MAIN_L1",
        "experiment": config["experiment_name"],
        "config": config,
        "logs": logs,
        "eval": {
            "main_nll": float(main_eval.loss.item()),
            "main_token_em": _answer_em(main_eval.logits, main_eval.labels),
            "loss1_nll": float(loss1_eval.loss.item()),
        },
        "runtime_sec": time.time() - start,
        "stop_gates_triggered": [],
    }
    if any(not log["all_finite"] for log in logs):
        report["status"] = "fail"
        report["stop_gates_triggered"].append("non_finite_gradient")
    if not any(log["model_a_grad_norm"] > 0 for log in logs):
        report["status"] = "fail"
        report["stop_gates_triggered"].append("loss1_or_main_did_not_update_model_a")
    out_path = Path(config["report_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_a": model_a.state_dict(),
            "model_b": model_b.state_dict(),
            "projector": projector.state_dict(),
        },
        ckpt_dir / "joint.pt",
    )
    return report
