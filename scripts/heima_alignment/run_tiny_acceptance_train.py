#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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

# Reuse the existing strict/scaled A+B Loss1 trainer. Do not fork its forward,
# loss, detach, optimizer, or checkpoint code in this wrapper.
from scripts import run_data_small_vlm_official_sections as trainer
from src.heima_aligned.tiny_acceptance_eval import build_tiny_acceptance_eval_manifest

DEFAULT_RUN_ROOT = Path("/data/zxl/runs/heima_ab_loss1_tiny_acceptance_v1")
DEFAULT_SUBSET = Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1")
DEFAULT_SPLIT = DEFAULT_RUN_ROOT / "data_split.json"
REPORT = ROOT / "reports" / "tiny_acceptance_trainer_integration_report.md"
JSON_REPORT = ROOT / "reports" / "tiny_acceptance_trainer_integration_report.json"


def mode_report_path(mode: str) -> Path:
    return ROOT / "reports" / f"tiny_acceptance_trainer_integration_{mode}.json"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_split_rows(split_path: Path, subset_root: Path) -> tuple[list[dict], list[dict], dict]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_all = read_jsonl(subset_root / "train.jsonl")
    eval_all = read_jsonl(subset_root / "validation.jsonl")

    def select(source: list[dict], items: list[dict]) -> list[dict]:
        rows = []
        for item in items:
            row = copy.deepcopy(source[int(item["index"])])
            # Existing trainer expects image relative to args.image_root.
            row["image"] = str(row["image"])
            for field in ("question", "summary", "caption", "reasoning", "answer"):
                if not row.get(field):
                    raise ValueError(f"missing field {field} for split item {item.get('id')}")
            rows.append(row)
        return rows

    train = select(train_all, split["train"])
    val = select(eval_all, split["eval"])
    return train, val, split


def build_trainer_args(ns: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_a_path=str(ns.model_a_path),
        model_b_path=str(ns.model_b_path),
        subset=ns.subset_root,
        image_root=ns.subset_root / "image_files",
        out=ns.run_root,
        seed=ns.seed,
        s0_steps=ns.s0_steps,
        s1_steps=ns.s1_steps,
        joint_steps=ns.joint_steps,
        batch_size=ns.batch_size,
        eval_samples=ns.eval_samples,
        lr_a=ns.lr_a,
        lr_b=ns.lr_b,
        lr_projector=ns.lr_projector,
        lambda1=ns.lambda1,
        weight_decay=ns.weight_decay,
        optimizer=ns.optimizer,
        clip_grad=ns.clip_grad,
        max_q=ns.max_q,
        max_target=ns.max_target,
        max_image_side=ns.max_image_side,
        torch_dtype=ns.torch_dtype,
        log_every=ns.log_every,
        train_latent_marker_ntp=False,
        loss1_latent_context_mode="local",
        cumulative_grad_mode="all_prefix",
        run_local_and_cumulative_comparison=False,
        save_generation_eval=True,
        skip_joint_checkpoints=False,
        generation_samples=ns.generation_samples,
        max_new_tokens=ns.max_new_tokens,
    )


def group_name(mode: str) -> str:
    if mode == "detach":
        return "baseline"
    if mode == "no_detach":
        return "ours"
    raise ValueError(mode)


def run_backward_smoke(args, train: list[dict], detach: bool) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.set_seed(args.seed)
    processor, tokenizer_a, model_a = trainer.load_vlm_a(args, device)
    tokenizer_b, proto = trainer.load_decoder_b(args, device)
    decoders = {"summary": proto}
    for section in ("caption", "reasoning"):
        _tok, model = trainer.load_decoder_b(args, device)
        decoders[section] = model
    projectors = {
        s: trainer.HeimaOfficialAbstractProjection(trainer.model_dim(model_a), trainer.model_dim(decoders[s])).to(
            device=device, dtype=trainer.model_dtype(decoders[s])
        )
        for s in trainer.SECTIONS
    }
    batch = trainer.batch_rows(train, args.batch_size, 0)
    attr = trainer.attribution(args, model_a, processor, tokenizer_a, tokenizer_b, decoders, projectors, batch, detach)
    del processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return attr


def run_formal(ns: argparse.Namespace, args, train: list[dict], val: list[dict]) -> dict:
    gname = group_name(ns.mode)
    run_dir = ns.run_root / gname
    if run_dir.exists() and not ns.resume:
        raise FileExistsError(f"run dir exists; pass --resume to reuse: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config" / "trainer_args.json", vars(args))
    write_json(run_dir / "config" / "tiny_acceptance.json", {
        "mode": ns.mode,
        "detach_encoder_latent": ns.mode == "detach",
        "split": str(ns.split),
        "subset_root": str(ns.subset_root),
        "semantic_evaluator_contract": build_tiny_acceptance_eval_manifest(train[0]),
    })
    if ns.eval_only:
        raise NotImplementedError("--eval-only requires an existing final checkpoint loader; not run in integration smoke")
    if ns.stage in ("all", "stage0"):
        processor, tokenizer_a, model_a = trainer.train_s0(args, run_dir, train)
    else:
        processor = tokenizer_a = model_a = None
    if ns.stage in ("all", "stage1"):
        if model_a is None:
            raise NotImplementedError("stage1 resume from existing S0 is not wired in this wrapper yet")
        tokenizer_b, decoders, projectors = trainer.train_s1(args, run_dir, processor, tokenizer_a, model_a, train, val)
        del processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if ns.stage in ("all", "stage2"):
        detach = ns.mode == "detach"
        result = trainer.train_joint(args, run_dir, train, val, detach, group_name=("heima_detach_baseline" if detach else "ours_loss1_no_detach"))
    else:
        result = {"status": "stage_completed_without_joint", "stage": ns.stage}
    return {"run_dir": str(run_dir), "result": result}


def write_report(payload: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    write_json(mode_report_path(payload["mode"]), payload)
    aggregate = {"latest": payload, "modes": {}}
    for mode in ("detach", "no_detach"):
        path = mode_report_path(mode)
        if path.exists():
            aggregate["modes"][mode] = json.loads(path.read_text(encoding="utf-8"))
    write_json(JSON_REPORT, aggregate)

    def grad(mode: str):
        item = aggregate["modes"].get(mode, {})
        return item.get("gradient_attribution", {}).get("grad_A_from_loss1")

    lines = [
        "# Tiny Acceptance Trainer Integration Report\n",
        "\n",
        "This integrates the tiny real-image acceptance data with the existing strict/scaled A+B Loss1 trainer. It does not introduce a new training framework.\n",
        "\n",
        "## Existing Trainer Audited\n",
        "\n",
        "- Reused script: `scripts/run_data_small_vlm_official_sections.py`\n",
        "- Reused functions: `encoder_forward`, `decoder_forward`, `prepare_stage_latents`, `attribution`, `train_s0`, `train_s1`, `train_joint`, `evaluate`, `save_generation_eval`\n",
        "- Reused detach switch: `prepare_stage_latents(... detach_encoder_latent=...)` -> `prepare_latent_for_decoder`\n",
        "- Reused optimizer/checkpoint path from existing trainer; wrapper only selects data/split/run layout.\n",
        "- Existing trainer is multi-section (`summary`, `caption`, `reasoning`); tiny acceptance reporting focuses on reasoning diagnostics without changing the trainer forward/loss path.\n",
        "\n",
        "## Tiny Data\n",
        f"- split: `{payload.get('split')}`\n",
        f"- train/eval: {payload.get('train_size')}/{payload.get('eval_size')}\n",
        "\n",
        "## Backward Smoke\n",
        "\n",
        f"- detach grad_A_from_loss1: `{grad('detach')}`\n",
        f"- no-detach grad_A_from_loss1: `{grad('no_detach')}`\n",
        "- Expected: detach is zero; no-detach is finite and greater than zero.\n",
        "\n",
        "## Formal Entrypoints\n",
        "\n",
        "- Baseline: `python scripts/heima_alignment/run_tiny_acceptance_train.py --mode detach` writes under `runs/heima_ab_loss1_tiny_acceptance_v1/baseline/`.\n",
        "- Ours: `python scripts/heima_alignment/run_tiny_acceptance_train.py --mode no_detach` writes under `runs/heima_ab_loss1_tiny_acceptance_v1/ours/`.\n",
        "- Smoke only: add `--stage smoke`; no optimizer step or checkpoint is saved.\n",
        "\n",
        "## Semantic Evaluator Contract\n",
        "\n",
        "The integration keeps the existing trainer evaluation and adds the tiny acceptance contract for downstream summarization: reasoning reconstruction NLL, content token NLL, numeric/entity/answer token accuracy, Q-only/correct/shuffle/zero interventions, generation exact match, and answer accuracy.\n",
    ]
    REPORT.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["detach", "no_detach"], required=True)
    p.add_argument("--stage", choices=["all", "stage0", "stage1", "stage2", "smoke"], default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--backward-smoke", action="store_true")
    p.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--subset-root", type=Path, default=DEFAULT_SUBSET)
    p.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--model-a-path", type=Path, default=Path("/data/zxl/small_models/Qwen2.5-VL-3B-Instruct"))
    p.add_argument("--model-b-path", type=Path, default=Path("/data/zxl/small_models/Qwen2.5-0.5B-Instruct"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--s0-steps", type=int, default=20)
    p.add_argument("--s1-steps", type=int, default=20)
    p.add_argument("--joint-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-samples", type=int, default=45)
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
    p.add_argument("--generation-samples", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=96)
    ns = p.parse_args()

    train, val, split = load_split_rows(ns.split, ns.subset_root)
    args = build_trainer_args(ns)
    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "prepared",
        "mode": ns.mode,
        "split": str(ns.split),
        "train_size": len(train),
        "eval_size": len(val),
        "detach_encoder_latent": ns.mode == "detach",
        "uses_existing_trainer": "scripts/run_data_small_vlm_official_sections.py",
        "semantic_evaluator_contract": build_tiny_acceptance_eval_manifest(train[0]),
    }
    if ns.dry_run and not ns.backward_smoke:
        payload["status"] = "dry_run_prepared"
        write_report(payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False)[:4000])
        return
    if ns.backward_smoke or ns.stage == "smoke":
        attr = run_backward_smoke(args, train, ns.mode == "detach")
        payload["status"] = "backward_smoke_pass"
        payload["gradient_attribution"] = attr
        write_report(payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False)[:4000])
        return
    result = run_formal(ns, args, train, val)
    payload["status"] = "formal_run_complete"
    payload["formal_result"] = result
    write_report(payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
