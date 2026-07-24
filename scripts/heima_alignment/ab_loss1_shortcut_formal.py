#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_data_small_vlm_official_sections as base

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

GROUPS = ("h0_heima_b_probe", "h1_joint_ab_loss1", "h2_frozen_b_loss1_to_a")
STAGE0_CKPT = Path("/data/zxl/runs/heima_stage2_interp_supervision_small_v1/seed42/20260722_135515/checkpoints/s0_encoder.pt")
SOURCE_SPLIT = Path("/data/zxl/runs/model_a_only_loss1_formal/data_split.json")
DATASET = Path("/data/zxl/runs/model_a_only_loss1_formal/formal_split")
IMAGE_ROOT = Path("/data/zxl/runs/model_a_only_loss1_formal/image_files")
OUT = Path("/data/zxl/runs/ab_loss1_shortcut_formal")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_stats(rows: list[dict], key: str) -> dict:
    lengths = [len(str(r.get(key, "")).split()) for r in rows]
    if not lengths:
        return {"min": 0, "mean": 0, "max": 0}
    return {"min": min(lengths), "mean": sum(lengths) / len(lengths), "max": max(lengths)}


def audit_data(args) -> dict:
    train_path = args.dataset_path / "train.jsonl"
    eval_path = args.dataset_path / "validation.jsonl"
    source = json.loads(args.source_split.read_text(encoding="utf-8")) if args.source_split.exists() else {}
    train = read_jsonl(train_path) if train_path.exists() else []
    val = read_jsonl(eval_path) if eval_path.exists() else []
    required = ("image", "question", "summary", "caption", "reasoning", "answer")
    all_rows = train + val
    missing_sections = {k: sum(1 for r in all_rows if not str(r.get(k, "")).strip()) for k in required}
    missing_images = [str(base.image_path(args, r)) for r in all_rows if not base.image_path(args, r).exists()]
    audit = {
        "status": "ready" if len(train) >= args.train_samples and len(val) >= args.eval_samples and not missing_images and all(v == 0 for v in missing_sections.values()) else "blocked",
        "source_split": str(args.source_split),
        "source_split_hash": source.get("split_hash") or (sha256_path(args.source_split) if args.source_split.exists() else None),
        "dataset_path": str(args.dataset_path),
        "image_root": str(args.image_root),
        "selected_train_count": len(train),
        "selected_eval_count": len(val),
        "target_train_count": args.train_samples,
        "target_eval_count": args.eval_samples,
        "image_access_rate": 0.0 if not all_rows else (len(all_rows) - len(missing_images)) / len(all_rows),
        "missing_images_count": len(missing_images),
        "missing_images_preview": missing_images[:20],
        "section_completeness": {k: {"missing": v, "complete": len(all_rows) - v} for k, v in missing_sections.items()},
        "token_statistics_whitespace": {k: text_stats(all_rows, k) for k in ("question", "summary", "caption", "reasoning", "answer")},
        "split_hash": sha256_path(train_path) + ":" + sha256_path(eval_path) if train_path.exists() and eval_path.exists() else None,
    }
    write_json(ROOT / "reports" / "ab_loss1_shortcut_data_audit.json", audit)
    write_json(args.output_dir / "reports" / "ab_loss1_shortcut_data_audit.json", audit)
    if audit["status"] != "ready":
        write_json(ROOT / "reports" / "ab_loss1_shortcut_data_failed.json", audit)
    return audit


def bridge_state_for_qwen25_vl(state: dict, target: dict) -> tuple[str, dict, list[str]]:
    candidates = [("raw", state)]
    candidates.append(("strip_model_prefix_per_key", {(k[len("model."):] if isinstance(k, str) and k.startswith("model.") else k): v for k, v in state.items()}))
    qwen25_bridge = {}
    for k, v in state.items():
        if isinstance(k, str) and k.startswith("model.visual."):
            qwen25_bridge[k[len("model."):]] = v
        elif isinstance(k, str) and k.startswith("model.language_model."):
            qwen25_bridge["model." + k[len("model.language_model."):]] = v
        else:
            qwen25_bridge[k] = v
    candidates.append(("qwen25_vl_model_bridge", qwen25_bridge))
    best_name, best_state, best_matches = "", {}, []
    for name, candidate in candidates:
        matches = [k for k, v in candidate.items() if k in target and tuple(v.shape) == tuple(target[k].shape)]
        if len(matches) > len(best_matches):
            best_name, best_state, best_matches = name, candidate, matches
    return best_name, best_state, best_matches


def load_stage0(model_a, checkpoint: Path, device: torch.device) -> dict:
    payload = torch.load(checkpoint, map_location=device)
    state = payload.get("model_a", payload)
    target = model_a.state_dict()
    name, bridged, matches = bridge_state_for_qwen25_vl(state, target)
    if not matches:
        raise RuntimeError(f"No compatible tensors in Stage0 checkpoint: {checkpoint}")
    missing, unexpected = model_a.load_state_dict({k: bridged[k] for k in matches}, strict=False)
    return {
        "stage0_checkpoint": str(checkpoint),
        "loaded": True,
        "key_transform": name,
        "checkpoint_tensors": len(state),
        "target_tensors": len(target),
        "loaded_tensors": len(matches),
        "missing_after_partial_load": len(missing),
        "unexpected_after_partial_load": len(unexpected),
        "first_missing": list(missing)[:20],
        "first_unexpected": list(unexpected)[:20],
    }


def audit_checkpoint(args) -> dict:
    payload = torch.load(args.stage0_checkpoint, map_location="cpu")
    state = payload.get("model_a", payload)
    audit = {
        "status": "usable_stage0_found" if args.stage0_checkpoint.exists() else "missing_stage0",
        "stage0_checkpoint": str(args.stage0_checkpoint),
        "checkpoint_source": "strict Heima Stage0 explicit-CoT warmup reused as A initialization",
        "model_a_name": args.model_a_path,
        "model_b_name": args.model_b_path,
        "git_commit": os.popen(f"cd {ROOT} && git rev-parse HEAD").read().strip(),
        "git_branch": os.popen(f"cd {ROOT} && git branch --show-current").read().strip(),
        "checkpoint_tensor_count": len(state),
        "config_path": "/data/zxl/runs/heima_stage2_interp_supervision_small_v1/seed42/20260722_135515/experiment_manifest.json",
        "dataset_path": str(args.dataset_path),
        "h0_b_checkpoint": str(args.output_dir / "h0_heima_b_probe" / "checkpoints" / "b_final.pt"),
        "canonical_b_init": str(args.output_dir / "checkpoints" / f"b_init_seed{args.seed}.pt"),
        "h2_dependency": "H2 must load h0_heima_b_probe/checkpoints/b_final.pt after H0 completes",
    }
    write_json(ROOT / "reports" / "ab_loss1_shortcut_checkpoint_audit.json", audit)
    write_json(args.output_dir / "reports" / "ab_loss1_shortcut_checkpoint_audit.json", audit)
    return audit


def load_models(args, *, load_b_from: Path | None, save_b_init: bool, dry_run: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, tokenizer_a, model_a = base.load_vlm_a(args, device)
    stage0_report = load_stage0(model_a, args.stage0_checkpoint, device)
    tokenizer_b, proto = base.load_decoder_b(args, device)
    decoders = {"summary": proto}
    for section in ("caption", "reasoning"):
        _tok, model = base.load_decoder_b(args, device)
        decoders[section] = model
    projectors = {s: base.HeimaOfficialAbstractProjection(base.model_dim(model_a), base.model_dim(decoders[s])).to(device=device, dtype=base.model_dtype(decoders[s])) for s in base.SECTIONS}
    init_path = args.output_dir / "checkpoints" / f"b_init_seed{args.seed}.pt"
    if load_b_from is not None:
        if not load_b_from.exists():
            raise FileNotFoundError(f"Missing required B checkpoint for {args.group}: {load_b_from}")
        payload = torch.load(load_b_from, map_location=device)
        for s in base.SECTIONS:
            decoders[s].load_state_dict(payload["decoders"][s])
            projectors[s].load_state_dict(payload["projectors"][s])
    elif init_path.exists():
        payload = torch.load(init_path, map_location=device)
        for s in base.SECTIONS:
            decoders[s].load_state_dict(payload["decoders"][s])
            projectors[s].load_state_dict(payload["projectors"][s])
    elif save_b_init and not dry_run:
        save_b_checkpoint(init_path, decoders, projectors, {"checkpoint_type": "canonical_b_init", "seed": args.seed})
    return device, processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors, stage0_report


def save_b_checkpoint(path: Path, decoders, projectors, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "meta": meta,
        "decoders": {s: decoders[s].state_dict() for s in base.SECTIONS},
        "projectors": {s: projectors[s].state_dict() for s in base.SECTIONS},
    }, path)


def save_a_checkpoint(path: Path, model_a, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "model_a": model_a.state_dict()}, path)


def set_group_trainability(args, model_a, decoders, projectors) -> None:
    train_a = args.group in {"h1_joint_ab_loss1", "h2_frozen_b_loss1_to_a"}
    train_b = args.group in {"h0_heima_b_probe", "h1_joint_ab_loss1"}
    for p in model_a.parameters():
        p.requires_grad_(train_a)
    for s in base.SECTIONS:
        decoders[s].train(train_b)
        projectors[s].train(train_b)
        for p in decoders[s].parameters():
            p.requires_grad_(train_b)
        for p in projectors[s].parameters():
            p.requires_grad_(train_b)


def trainable_params(model) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def optimizer_for_group(args, model_a, decoders, projectors):
    groups = []
    if args.group in {"h1_joint_ab_loss1", "h2_frozen_b_loss1_to_a"}:
        groups.append({"params": trainable_params(model_a), "lr": args.lr_a})
    if args.group in {"h0_heima_b_probe", "h1_joint_ab_loss1"}:
        for s in base.SECTIONS:
            groups.append({"params": trainable_params(decoders[s]), "lr": args.lr_b})
            groups.append({"params": trainable_params(projectors[s]), "lr": args.lr_projector})
    groups = [g for g in groups if g["params"]]
    if not groups:
        raise RuntimeError(f"No trainable parameters for group {args.group}")
    return base.make_optimizer(groups, args)


def loss1_forward(args, decoders, projectors, tokenizer_b, batch, z, *, detach_encoder_latent: bool):
    losses = {}
    loss_sum = torch.zeros((), device=next(decoders["summary"].parameters()).device)
    for s in base.SECTIONS:
        loss, _logits, _labels = base.decoder_forward(
            decoders[s], projectors, tokenizer_b, s, batch,
            base.prepare_stage_latents(z, s, args, detach_encoder_latent=detach_encoder_latent), args,
        )
        losses[s] = loss
        loss_sum = loss_sum + loss
    return loss_sum / len(base.SECTIONS), losses


def grad_audit(args, model_a, processor, tokenizer_a, tokenizer_b, decoders, projectors, batch) -> dict:
    for module in [model_a] + [decoders[s] for s in base.SECTIONS] + [projectors[s] for s in base.SECTIONS]:
        module.zero_grad(set_to_none=True)
    detach = args.group == "h0_heima_b_probe"
    main, _logits, _labels, z, _trace = base.encoder_forward(model_a, processor, tokenizer_a, batch, args)
    for s in base.SECTIONS:
        if z[s].requires_grad:
            z[s].retain_grad()
    loss1_mean, per = loss1_forward(args, decoders, projectors, tokenizer_b, batch, z, detach_encoder_latent=detach)
    if loss1_mean.requires_grad:
        loss1_mean.backward(retain_graph=True)
    grad_a, finite_a = base.grad_norm(model_a.parameters())
    b_params = [p for s in base.SECTIONS for p in list(decoders[s].parameters()) + list(projectors[s].parameters())]
    grad_b, finite_b = base.grad_norm(b_params)
    grad_z = {s: base.finite_norm(z[s].grad)[0] if z[s].grad is not None else 0.0 for s in base.SECTIONS}
    out = {
        "group": args.group,
        "main_loss_probe": float(main.detach().cpu().item()),
        "loss1_mean_probe": float(loss1_mean.detach().cpu().item()),
        "loss1_by_section": {s: float(per[s].detach().cpu().item()) for s in base.SECTIONS},
        "detach_encoder_latent_for_loss1": detach,
        "grad_z_summary": grad_z["summary"],
        "grad_z_caption": grad_z["caption"],
        "grad_z_reasoning": grad_z["reasoning"],
        "grad_A_from_loss1": grad_a,
        "grad_B_from_loss1": grad_b,
        "finite_A": finite_a,
        "finite_B": finite_b,
    }
    for module in [model_a] + [decoders[s] for s in base.SECTIONS] + [projectors[s] for s in base.SECTIONS]:
        module.zero_grad(set_to_none=True)
    return out


def evaluate_group(args, model_a, processor, tokenizer_a, tokenizer_b, decoders, projectors, val, step: int, group_dir: Path, ckpt: str) -> dict:
    sample = val[: args.eval_samples]
    metrics = base.evaluate(model_a, processor, tokenizer_a, decoders, projectors, tokenizer_b, sample, args)
    sections = metrics.get("sections", {})
    simplified = {
        "step": step,
        "checkpoint": ckpt,
        "validation_main_nll": metrics.get("main_nll"),
        "answer_generation_accuracy": None,
        "answer_generation_note": "Formal Heima benchmark evaluation disabled by request; answer generation accuracy is not computed in this control runner.",
        "loss1_reconstruction": {s: {"ce_loss": sections.get(s, {}).get("qz_nll"), "token_accuracy": None} for s in base.SECTIONS},
        "latent_intervention": {
            s: {
                "correct_nll": sections.get(s, {}).get("qz_nll"),
                "shuffle_nll": sections.get(s, {}).get("shuffle_nll"),
                "zero_nll": sections.get(s, {}).get("zero_nll"),
                "q_only_nll": sections.get(s, {}).get("q_only_nll"),
                "shuffle_margin": sections.get(s, {}).get("whole_shuffle_margin"),
                "zero_margin": sections.get(s, {}).get("normal_zero_margin"),
                "q_gain": sections.get(s, {}).get("qz_gain_over_q"),
            }
            for s in base.SECTIONS
        },
        "raw_metrics": metrics,
    }
    eval_dir = group_dir / f"eval_step{step}"
    write_json(eval_dir / "metrics.json", simplified)
    if args.generation_samples > 0:
        base.save_generation_eval(eval_dir / "generation.jsonl", model_a, processor, tokenizer_a, decoders, projectors, tokenizer_b, sample, args, training_group=args.group, checkpoint=ckpt)
    return simplified


def assert_group_contract(args, optimizer, model_a, decoders, projectors) -> dict:
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    a_ids = {id(p) for p in model_a.parameters()}
    b_ids = {id(p) for s in base.SECTIONS for p in list(decoders[s].parameters()) + list(projectors[s].parameters())}
    contract = {
        "group": args.group,
        "optimizer_contains_A": bool(opt_ids & a_ids),
        "optimizer_contains_B": bool(opt_ids & b_ids),
        "trainable_A_params": sum(p.numel() for p in model_a.parameters() if p.requires_grad),
        "trainable_B_params": sum(p.numel() for s in base.SECTIONS for p in list(decoders[s].parameters()) + list(projectors[s].parameters()) if p.requires_grad),
        "loss2_enabled": False,
        "cumulative_latent": False,
        "model_a_only_training": False,
    }
    expected = {
        "h0_heima_b_probe": (False, True),
        "h1_joint_ab_loss1": (True, True),
        "h2_frozen_b_loss1_to_a": (True, False),
    }[args.group]
    if (contract["optimizer_contains_A"], contract["optimizer_contains_B"]) != expected:
        raise RuntimeError(f"Optimizer contract failed: {contract}")
    return contract


def run_dry_run(args) -> dict:
    data = audit_data(args)
    ckpt = audit_checkpoint(args)
    if data["status"] != "ready":
        raise SystemExit("STOP: data audit is blocked; training not started")
    group_summaries = {}
    groups = GROUPS if args.group == "all" else (args.group,)
    for group in groups:
        args.group = group
        load_b = args.output_dir / "h0_heima_b_probe" / "checkpoints" / "b_final.pt" if group == "h2_frozen_b_loss1_to_a" else None
        if group == "h2_frozen_b_loss1_to_a" and not load_b.exists():
            group_summaries[group] = {"dry_run_status": "deferred_until_h0_final_b_exists", "required_b_checkpoint": str(load_b)}
            continue
        base.set_seed(args.seed + 707)
        device, processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors, stage0_report = load_models(args, load_b_from=load_b, save_b_init=False, dry_run=True)
        set_group_trainability(args, model_a, decoders, projectors)
        optimizer = optimizer_for_group(args, model_a, decoders, projectors)
        contract = assert_group_contract(args, optimizer, model_a, decoders, projectors)
        train = read_jsonl(args.dataset_path / "train.jsonl")[: args.train_samples]
        audit = grad_audit(args, model_a, processor, tokenizer_a, tokenizer_b, decoders, projectors, train[: args.batch_size])
        group_summaries[group] = {"dry_run_status": "pass", "stage0_load": stage0_report, "optimizer_contract": contract, "gradient_probe": audit}
        del model_a, decoders, projectors, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out = {"status": "pass", "data": data, "checkpoint": ckpt, "groups": group_summaries}
    write_json(ROOT / "reports" / "ab_loss1_shortcut_preflight.json", out)
    write_json(args.output_dir / "reports" / "ab_loss1_shortcut_preflight.json", out)
    print(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True))
    return out


def run_train(args) -> dict:
    if args.group not in GROUPS:
        raise SystemExit("--group must be one concrete group for training")
    data = audit_data(args)
    if data["status"] != "ready":
        raise SystemExit("STOP: data audit is blocked; training not started")
    audit_checkpoint(args)
    group_dir = args.output_dir / args.group
    if group_dir.exists() and not args.resume:
        raise SystemExit(f"STOP: group output already exists; pass --resume to continue: {group_dir}")
    group_dir.mkdir(parents=True, exist_ok=True)
    train = read_jsonl(args.dataset_path / "train.jsonl")[: args.train_samples]
    val = read_jsonl(args.dataset_path / "validation.jsonl")[: args.eval_samples]
    load_b = args.output_dir / "h0_heima_b_probe" / "checkpoints" / "b_final.pt" if args.group == "h2_frozen_b_loss1_to_a" else None
    base.set_seed(args.seed + 707)
    device, processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors, stage0_report = load_models(args, load_b_from=load_b, save_b_init=True, dry_run=False)
    set_group_trainability(args, model_a, decoders, projectors)
    optimizer = optimizer_for_group(args, model_a, decoders, projectors)
    contract = assert_group_contract(args, optimizer, model_a, decoders, projectors)
    manifest = {
        "status": "running",
        "group": args.group,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage0_load": stage0_report,
        "optimizer_contract": contract,
        "loss_contract": "H0: Loss1 only; H1/H2: main + lambda_loss1 * mean(section Loss1)",
        "lambda_loss1": args.lambda_loss1,
        "save_optimizer": False,
        "loss2_enabled": False,
        "cumulative_latent": False,
        "args": vars(args),
    }
    write_json(group_dir / "manifest.json", manifest)
    grad0 = grad_audit(args, model_a, processor, tokenizer_a, tokenizer_b, decoders, projectors, train[: args.batch_size])
    write_json(group_dir / "gradient_step0.json", grad0)
    logs, evals = [], []
    save_steps = set(args.save_steps)
    for step in range(1, args.max_steps + 1):
        batch = base.batch_rows(train, args.batch_size, step - 1)
        optimizer.zero_grad(set_to_none=True)
        detach = args.group == "h0_heima_b_probe"
        if detach:
            with torch.no_grad():
                main, _logits, _labels, z, _trace = base.encoder_forward(model_a, processor, tokenizer_a, batch, args)
        else:
            main, _logits, _labels, z, _trace = base.encoder_forward(model_a, processor, tokenizer_a, batch, args)
        loss1_mean, per = loss1_forward(args, decoders, projectors, tokenizer_b, batch, z, detach_encoder_latent=detach)
        total = loss1_mean if args.group == "h0_heima_b_probe" else main + args.lambda_loss1 * loss1_mean
        total.backward()
        grad_a, finite_a = base.grad_norm(model_a.parameters())
        b_params = [p for s in base.SECTIONS for p in list(decoders[s].parameters()) + list(projectors[s].parameters())]
        grad_b, finite_b = base.grad_norm(b_params)
        if not finite_a or not finite_b:
            raise RuntimeError("non-finite gradients")
        torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], args.clip_grad)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step in save_steps:
            row = {"step": step, "main_loss": float(main.detach().cpu().item()), "loss1_mean": float(loss1_mean.detach().cpu().item()), "loss1_by_section": {s: float(per[s].detach().cpu().item()) for s in base.SECTIONS}, "total": float(total.detach().cpu().item()), "grad_A_total": grad_a, "grad_B_total": grad_b}
            logs.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            write_json(group_dir / "train_progress.json", logs)
        if step in save_steps:
            ckpt_dir = group_dir / "checkpoints"
            ckpt_paths = {}
            if args.group in {"h1_joint_ab_loss1", "h2_frozen_b_loss1_to_a"}:
                a_path = ckpt_dir / f"a_step{step}.pt"
                save_a_checkpoint(a_path, model_a, {"group": args.group, "step": step})
                ckpt_paths["A"] = str(a_path)
            if args.group in {"h0_heima_b_probe", "h1_joint_ab_loss1"}:
                b_path = ckpt_dir / f"b_step{step}.pt"
                save_b_checkpoint(b_path, decoders, projectors, {"group": args.group, "step": step})
                ckpt_paths["B"] = str(b_path)
            grad = grad_audit(args, model_a, processor, tokenizer_a, tokenizer_b, decoders, projectors, train[: args.batch_size])
            write_json(group_dir / f"gradient_step{step}.json", grad)
            metrics = evaluate_group(args, model_a, processor, tokenizer_a, tokenizer_b, decoders, projectors, val, step, group_dir, json.dumps(ckpt_paths, sort_keys=True))
            evals.append(metrics)
            write_json(group_dir / "eval_progress.json", evals)
    final_paths = {}
    if args.group in {"h1_joint_ab_loss1", "h2_frozen_b_loss1_to_a"}:
        a_final = group_dir / "checkpoints" / "a_final.pt"
        save_a_checkpoint(a_final, model_a, {"group": args.group, "step": args.max_steps, "checkpoint_type": "final"})
        final_paths["A"] = str(a_final)
    if args.group in {"h0_heima_b_probe", "h1_joint_ab_loss1"}:
        b_final = group_dir / "checkpoints" / "b_final.pt"
        save_b_checkpoint(b_final, decoders, projectors, {"group": args.group, "step": args.max_steps, "checkpoint_type": "final"})
        final_paths["B"] = str(b_final)
    manifest["status"] = "completed"
    manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["final_checkpoints"] = final_paths
    write_json(group_dir / "manifest.json", manifest)
    return manifest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=("all",) + GROUPS, default="h0_heima_b_probe")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--prepare-data", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--model-a-path", default="/data/zxl/small_models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--model-b-path", default="/data/zxl/small_models/Qwen2.5-0.5B-Instruct")
    p.add_argument("--stage0-checkpoint", type=Path, default=STAGE0_CKPT)
    p.add_argument("--source-split", type=Path, default=SOURCE_SPLIT)
    p.add_argument("--dataset-path", type=Path, default=DATASET)
    p.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--train-samples", type=int, default=5000)
    p.add_argument("--eval-samples", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--save-steps", type=lambda s: [int(x) for x in s.split(",") if x], default=[1000, 2500, 5000])
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--generation-samples", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--lr-a", type=float, default=1e-5)
    p.add_argument("--lr-b", type=float, default=2e-5)
    p.add_argument("--lr-projector", type=float, default=1e-4)
    p.add_argument("--lambda-loss1", type=float, default=0.1)
    p.add_argument("--lambda-interp", type=float, default=0.1)
    p.add_argument("--lambda1", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optimizer", choices=("adafactor", "adamw"), default="adafactor")
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--max-q", type=int, default=160)
    p.add_argument("--max-target", type=int, default=160)
    p.add_argument("--max-image-side", type=int, default=336)
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--loss1-latent-context-mode", choices=("local",), default="local")
    p.add_argument("--cumulative-grad-mode", choices=("all_prefix", "current_only"), default="all_prefix")
    p.add_argument("--train-latent-marker-ntp", action="store_true", default=False)
    args = p.parse_args()
    args.save_generation_eval = True
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.prepare_data:
        audit = audit_data(args)
        print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))
        if not args.dry_run and not args.train:
            return
    if args.dry_run:
        run_dry_run(args)
        return
    if args.train:
        run_train(args)
        return
    audit_data(args)
    audit_checkpoint(args)


if __name__ == "__main__":
    main()
