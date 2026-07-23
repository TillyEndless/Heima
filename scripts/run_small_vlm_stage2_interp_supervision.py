#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts import run_data_small_vlm_official_sections as base
from src.heima_stage2.loss2_alignment import loss2_forward as stage2_loss2_forward


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def freeze_interpreters(decoders, projectors) -> None:
    for section in base.SECTIONS:
        decoders[section].eval()
        projectors[section].eval()
        for param in decoders[section].parameters():
            param.requires_grad_(False)
        for param in projectors[section].parameters():
            param.requires_grad_(False)


def assert_interpreters_frozen(decoders, projectors) -> None:
    bad = []
    for section in base.SECTIONS:
        bad += [f"decoder.{section}.{name}" for name, param in decoders[section].named_parameters() if param.requires_grad]
        bad += [f"projector.{section}.{name}" for name, param in projectors[section].named_parameters() if param.requires_grad]
    if bad:
        raise RuntimeError("Stage2 teacher interpreter must be frozen: " + ", ".join(bad[:8]))


def grad_a_from_interp(args, model_a, processor, tokenizer_a, tokenizer_b, decoders, projectors, batch, detach: bool) -> dict:
    for model in [model_a] + [decoders[s] for s in base.SECTIONS] + [projectors[s] for s in base.SECTIONS]:
        model.zero_grad(set_to_none=True)
    _main, _logits, _labels, z, _trace = base.encoder_forward(model_a, processor, tokenizer_a, batch, args)
    loss_sum = torch.zeros((), device=next(model_a.parameters()).device)
    per = {}
    for section in base.SECTIONS:
        loss, _logits_b, _labels_b = base.decoder_forward(
            decoders[section],
            projectors,
            tokenizer_b,
            section,
            batch,
            base.prepare_stage_latents(z, section, args, detach_encoder_latent=detach),
            args,
        )
        loss_sum = loss_sum + loss
        per[section] = float(loss.item())
    if loss_sum.requires_grad:
        loss_sum.backward()
        grad_a, finite = base.grad_norm(model_a.parameters())
    else:
        grad_a, finite = 0.0, True
    teacher_grad = {}
    for section in base.SECTIONS:
        grad_decoder, finite_decoder = base.grad_norm(decoders[section].parameters())
        grad_projector, finite_projector = base.grad_norm(projectors[section].parameters())
        teacher_grad[section] = {
            "decoder": grad_decoder,
            "decoder_finite": finite_decoder,
            "projector": grad_projector,
            "projector_finite": finite_projector,
        }
    return {
        "detach_encoder_latent": detach,
        "grad_A_from_interp": grad_a,
        "finite": finite,
        "loss_interp": per,
        "teacher_grad": teacher_grad,
    }




def stage2_mode_flags(mode: str) -> dict:
    if mode == "heima_baseline":
        return {"loss1_to_total": False, "loss2_to_total": False, "detach_loss1": True, "detach_loss2": True}
    if mode in {"ours_interp_supervision", "ours_loss1"}:
        return {"loss1_to_total": True, "loss2_to_total": False, "detach_loss1": False, "detach_loss2": True}
    if mode == "ours_loss1_loss2":
        return {"loss1_to_total": True, "loss2_to_total": True, "detach_loss1": False, "detach_loss2": False}
    raise ValueError(mode)


def teacher_grad_summary(decoders, projectors) -> dict:
    out = {}
    for section in base.SECTIONS:
        grad_decoder, finite_decoder = base.grad_norm(decoders[section].parameters())
        grad_projector, finite_projector = base.grad_norm(projectors[section].parameters())
        out[section] = {
            "decoder": grad_decoder,
            "decoder_finite": finite_decoder,
            "projector": grad_projector,
            "projector_finite": finite_projector,
        }
    return out


def compute_loss1(args, tokenizer_b, decoders, projectors, batch, z, *, detach: bool):
    loss_sum = torch.zeros((), device=next(decoders[base.SECTIONS[0]].parameters()).device)
    per = {}
    for section in base.SECTIONS:
        loss, _logits_b, _labels_b = base.decoder_forward(
            decoders[section],
            projectors,
            tokenizer_b,
            section,
            batch,
            base.prepare_stage_latents(z, section, args, detach_encoder_latent=detach),
            args,
        )
        loss_sum = loss_sum + loss
        per[section] = float(loss.item())
    return loss_sum, per


def compute_loss2(args, tokenizer_b, decoders, projectors, batch, z, *, detach: bool):
    loss_sum = torch.zeros((), device=next(decoders[base.SECTIONS[0]].parameters()).device)
    per = {}
    shapes = {}
    for section in base.SECTIONS:
        out = stage2_loss2_forward(
            model_b=decoders[section],
            projector=projectors[section],
            tokenizer=tokenizer_b,
            records=batch,
            section=section,
            sections=base.SECTIONS,
            z=z[section],
            max_q=args.max_q,
            max_target=args.max_target,
            thinking_token=base.THINKING_TOKENS[section],
            pool=args.loss2_pool,
            distance=args.loss2_distance,
            text_context_mode=args.loss2_text_context_mode,
            detach_latent=detach,
        )
        loss_sum = loss_sum + out.loss
        per[section] = float(out.loss.item())
        shapes[section] = {
            "H_latent_i": list(out.latent_shape),
            "H_text_i": list(out.text_shape),
            "h_text_detached": out.h_text_detached,
        }
    return loss_sum / len(base.SECTIONS), per, shapes


def grad_attribution_loss1_loss2(args, model_a, processor, tokenizer_a, tokenizer_b, decoders, projectors, batch, mode: str) -> dict:
    flags = stage2_mode_flags(mode)
    out = {
        "mode": mode,
        "lambda_loss1": args.lambda_loss1,
        "lambda_loss2": args.lambda_loss2,
        "loss2_pool": args.loss2_pool,
        "loss2_distance": args.loss2_distance,
        "loss2_text_context_mode": args.loss2_text_context_mode,
    }
    for model in [model_a] + [decoders[s] for s in base.SECTIONS] + [projectors[s] for s in base.SECTIONS]:
        model.zero_grad(set_to_none=True)
    _main, _logits, _labels, z, _trace = base.encoder_forward(model_a, processor, tokenizer_a, batch, args)
    loss1, per1 = compute_loss1(args, tokenizer_b, decoders, projectors, batch, z, detach=flags["detach_loss1"])
    if loss1.requires_grad:
        loss1.backward(retain_graph=True)
        g_a1, f_a1 = base.grad_norm(model_a.parameters())
    else:
        g_a1, f_a1 = 0.0, True
    teacher1 = teacher_grad_summary(decoders, projectors)
    for model in [model_a] + [decoders[s] for s in base.SECTIONS] + [projectors[s] for s in base.SECTIONS]:
        model.zero_grad(set_to_none=True)
    _main, _logits, _labels, z, _trace = base.encoder_forward(model_a, processor, tokenizer_a, batch, args)
    loss2, per2, shapes = compute_loss2(args, tokenizer_b, decoders, projectors, batch, z, detach=flags["detach_loss2"])
    if loss2.requires_grad:
        loss2.backward()
        g_a2, f_a2 = base.grad_norm(model_a.parameters())
    else:
        g_a2, f_a2 = 0.0, True
    teacher2 = teacher_grad_summary(decoders, projectors)
    out.update({
        "grad_A_from_loss1": g_a1,
        "finite_A_loss1": f_a1,
        "grad_A_from_loss2": g_a2,
        "finite_A_loss2": f_a2,
        "grad_B_from_loss1": teacher1,
        "grad_B_from_loss2": teacher2,
        "per_section_loss1": per1,
        "per_section_loss2": per2,
        "loss2_hidden_shapes": shapes,
        "teacher_B_frozen": True,
        "optimizer_contains_B": False,
    })
    return out

def train_stage2(args, run_dir: Path, train, val, *, mode: str) -> dict:
    flags = stage2_mode_flags(mode)
    detach = flags["detach_loss1"]
    base.set_seed(args.seed + 303)
    processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors = base.load_s1(args, run_dir)
    for param in model_a.parameters():
        param.requires_grad_(True)
    freeze_interpreters(decoders, projectors)
    assert_interpreters_frozen(decoders, projectors)
    optimizer = base.make_optimizer([{"params": model_a.parameters(), "lr": args.lr_a}], args)
    optimizer_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    teacher_param_ids = {id(param) for section in base.SECTIONS for param in list(decoders[section].parameters()) + list(projectors[section].parameters())}
    if optimizer_param_ids & teacher_param_ids:
        raise RuntimeError("Stage2 optimizer contains frozen interpreter parameters")
    attr = grad_attribution_loss1_loss2(args, model_a, processor, tokenizer_a, tokenizer_b, decoders, projectors, base.batch_rows(train, args.batch_size, 0), mode)
    if mode == "heima_baseline" and (attr["grad_A_from_loss1"] != 0.0 or attr["grad_A_from_loss2"] != 0.0):
        raise RuntimeError("Heima baseline Loss1/Loss2 returned gradient to A")
    if mode in {"ours_interp_supervision", "ours_loss1"} and not (attr["grad_A_from_loss1"] > 0 and attr["grad_A_from_loss2"] == 0.0):
        raise RuntimeError("Ours Loss1 mode failed gradient attribution")
    if mode == "ours_loss1_loss2" and not (attr["grad_A_from_loss1"] > 0 and attr["grad_A_from_loss2"] > 0):
        raise RuntimeError("Ours Loss1+Loss2 mode failed gradient attribution")
    logs = []
    start = time.time()
    for step in range(1, args.stage2_steps + 1):
        batch = base.batch_rows(train, args.batch_size, step - 1)
        optimizer.zero_grad(set_to_none=True)
        main, _logits, _labels, z, _trace = base.encoder_forward(model_a, processor, tokenizer_a, batch, args)
        interp, per, = compute_loss1(args, tokenizer_b, decoders, projectors, batch, z, detach=flags["detach_loss1"])
        loss2, per2, shapes2 = compute_loss2(args, tokenizer_b, decoders, projectors, batch, z, detach=flags["detach_loss2"])
        if mode == "heima_baseline":
            total = main
        elif mode == "ours_interp_supervision":
            total = main + args.lambda_interp * interp
        elif mode == "ours_loss1":
            total = main + args.lambda_loss1 * interp
        elif mode == "ours_loss1_loss2":
            total = main + args.lambda_loss1 * interp + args.lambda_loss2 * loss2
        else:
            raise ValueError(mode)
        total.backward()
        grad_a, finite = base.grad_norm(model_a.parameters())
        if not finite:
            raise RuntimeError("non-finite Stage2 A gradient")
        for section in base.SECTIONS:
            grad_decoder, _ = base.grad_norm(decoders[section].parameters())
            grad_projector, _ = base.grad_norm(projectors[section].parameters())
            if grad_decoder != 0.0 or grad_projector != 0.0:
                raise RuntimeError("Stage2 produced teacher B/projector gradient")
        torch.nn.utils.clip_grad_norm_(model_a.parameters(), args.clip_grad)
        optimizer.step()
        if step == 1 or step == args.stage2_steps or step % args.log_every == 0:
            logs.append({
                "step": step,
                "ntp_loss": float(main.item()),
                "loss1": per,
                "loss2": per2,
                "loss2_hidden_shapes": shapes2,
                "total": float(total.item()),
                "grad_A": grad_a,
                "lambda_loss1": args.lambda_loss1,
                "lambda_loss2": args.lambda_loss2,
            })
    metrics = base.evaluate(model_a, processor, tokenizer_a, decoders, projectors, tokenizer_b, val[: args.eval_samples], args)
    ckpt_path = run_dir / "checkpoints" / f"stage2_{mode}_A.pt"
    base.save_ckpt(ckpt_path, model_a=model_a.state_dict())
    generation_path = run_dir / "generations" / f"stage2_{mode}.jsonl"
    base.save_generation_eval(generation_path, model_a, processor, tokenizer_a, decoders, projectors, tokenizer_b, val, args, training_group=mode, checkpoint=str(ckpt_path))
    result = {
        "mode": mode,
        "runtime_sec": time.time() - start,
        "checkpoint": str(ckpt_path),
        "checkpoint_saved": True,
        "generation_eval": str(generation_path),
        "teacher_B_frozen": True,
        "optimizer_contains_B": False,
        "gradient_attribution": attr,
        "logs": logs,
        "validation": metrics,
    }
    write_json(run_dir / "groups" / f"stage2_{mode}" / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a-path", default="/data/zxl/small_models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--model-b-path", default="/data/zxl/small_models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--subset", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_available_images_v1"))
    parser.add_argument("--image-root", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1/image_files"))
    parser.add_argument("--out", type=Path, default=Path("/data/zxl/runs/heima_stage2_interp_supervision_small_v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--s0-steps", type=int, default=200)
    parser.add_argument("--s1-steps", type=int, default=800)
    parser.add_argument("--stage2-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--lr-a", type=float, default=1e-5)
    parser.add_argument("--lr-b", type=float, default=2e-5)
    parser.add_argument("--lr-projector", type=float, default=1e-4)
    parser.add_argument("--lambda-interp", type=float, default=0.1)
    parser.add_argument("--lambda-loss1", type=float, default=0.1)
    parser.add_argument("--lambda-loss2", type=float, default=0.05)
    parser.add_argument("--loss2-pool", choices=["mean", "last"], default="mean")
    parser.add_argument("--loss2-distance", choices=["normalized_mse", "mse", "cosine"], default="normalized_mse")
    parser.add_argument("--loss2-text-context-mode", choices=["cumulative", "section_only"], default="cumulative")
    parser.add_argument("--lambda1", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adafactor", "adamw"], default="adafactor")
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--max-q", type=int, default=160)
    parser.add_argument("--max-target", type=int, default=160)
    parser.add_argument("--max-image-side", type=int, default=336)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--loss1-latent-context-mode", choices=["local", "causal_cumulative"], default="local")
    parser.add_argument("--cumulative-grad-mode", choices=["all_prefix", "current_only"], default="all_prefix")
    parser.add_argument("--generation-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--dry-run", action="store_true")
    args = SimpleNamespace(**vars(parser.parse_args()))
    args.save_generation_eval = True
    args.skip_joint_checkpoints = False
    args.train_latent_marker_ntp = False
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.out / f"seed{args.seed}" / run_id
    train = read_jsonl(args.subset / "train.jsonl")
    val = read_jsonl(args.subset / "validation.jsonl")
    missing = [str(base.image_path(args, row)) for row in train + val if not base.image_path(args, row).exists()]
    if missing:
        raise FileNotFoundError(f"missing images: {missing[:5]} count={len(missing)}")
    manifest = {
        "status": "running",
        "framework": "strict Heima Stage1 + Stage2 interpreter supervision, small-model scaled run",
        "model_A": args.model_a_path,
        "model_B": args.model_b_path,
        "dataset": str(args.subset),
        "image_root": str(args.image_root),
        "train_samples": len(train),
        "validation_samples": len(val),
        "only_stage2_difference": "interpreter loss gradient to Model A",
        "stage2_heima_baseline": "B frozen; optimizer A only; total=L_NTP; L_interp computed with z.detach for logging/eval",
        "stage2_ours_interp_supervision": "B frozen; optimizer A only; total=L_NTP + lambda_interp*L_interp; z not detached",
        "stage2_ours_loss1_loss2": "B frozen; optimizer A only; total=L_NTP + lambda_loss1*L_loss1 + lambda_loss2*L_loss2; Loss2 aligns same-B latent/text hidden features",
        "args": vars(args),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_component_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
    }
    write_json(run_dir / "experiment_manifest.json", manifest)
    if args.dry_run:
        manifest["status"] = "dry_run"
        write_json(run_dir / "experiment_manifest.json", manifest)
        print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True, default=str))
        return
    processor, tokenizer_a, model_a = base.train_s0(args, run_dir, train)
    tokenizer_b, decoders, projectors = base.train_s1(args, run_dir, processor, tokenizer_a, model_a, train, val)
    del processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    results = {
        "heima_baseline": train_stage2(args, run_dir, train, val, mode="heima_baseline"),
        "ours_interp_supervision": train_stage2(args, run_dir, train, val, mode="ours_interp_supervision"),
        "ours_loss1_loss2": train_stage2(args, run_dir, train, val, mode="ours_loss1_loss2"),
    }
    manifest["status"] = "completed"
    write_json(run_dir / "experiment_manifest.json", manifest)
    write_json(run_dir / "stage2_comparison_summary.json", results)


if __name__ == "__main__":
    main()
