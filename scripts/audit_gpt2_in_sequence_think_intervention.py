#!/usr/bin/env python3
"""In-sequence THINK intervention audit for GPT2 CoT latent adjacent groups.

No training. Loads final checkpoints and measures CoT/Answer NLL on the original
Question <THINK> CoT Answer sequence under THINK-position interventions.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_gpt2_cot_latent_adjacent_matrix import (  # noqa: E402
    Example,
    load_model,
    make_main_batch,
)

REPORT_DIR = ROOT / "reports"
DEFAULT_RUN_ROOT = Path("/data/zxl/runs/gpt2_cot_latent_adjacent_matrix_seed42")
DEFAULT_BASE = "/data/zxl/models/openai-community-gpt2"


def load_split(run_root: Path, eval_n: int) -> List[Example]:
    payload = json.loads((run_root / "data_split.json").read_text(encoding="utf-8"))
    return [Example(**x) for x in payload["eval"][:eval_n]]


def load_group_model(group: str, run_root: Path, model_path: str, device: torch.device):
    ckpt_dir = run_root / group / "step5000"
    model, tok = load_model(model_path, device)
    state = torch.load(ckpt_dir / "model_weights.pt", map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, tok, ckpt_dir


def token_losses(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    flat = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    return flat.view(shifted_labels.shape)


def masked_nll_per_sample(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> List[float]:
    shifted_mask = mask[:, 1:].contiguous()
    losses = token_losses(logits, labels)
    denom = shifted_mask.sum(dim=1).clamp_min(1)
    vals = (losses * shifted_mask.float()).sum(dim=1) / denom
    return vals.detach().cpu().tolist()


def mean(xs: Sequence[float]) -> float:
    vals = list(xs)
    return float(sum(vals) / max(1, len(vals)))


def bootstrap_ci(vals: Sequence[float], seed: int = 42, n: int = 1000) -> List[float | None]:
    vals = list(vals)
    if not vals:
        return [None, None]
    rng = random.Random(seed)
    outs = []
    m = len(vals)
    for _ in range(n):
        outs.append(mean(vals[rng.randrange(m)] for __ in range(m)))
    outs.sort()
    return [outs[int(0.025 * n)], outs[int(0.975 * n) - 1]]


def clean_layer_acts(model, batch) -> tuple[torch.Tensor, List[torch.Tensor]]:
    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
    # hidden_states[0] is embedding output, hidden_states[i+1] is block i output.
    layer_acts = [h.detach() for h in out.hidden_states[1:]]
    return out.logits.detach(), layer_acts


def patch_block_outputs(model, source_acts: List[torch.Tensor], positions: torch.Tensor, mode: str, perm: torch.Tensor):
    handles = []
    rows = torch.arange(positions.size(0), device=positions.device)

    def make_hook(layer_idx: int):
        source = source_acts[layer_idx]

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                rest = output[1:]
            else:
                hidden = output.clone()
                rest = None
            valid = positions >= 0
            rr = rows[valid]
            pp = positions[valid]
            if mode == "zero_think":
                hidden[rr, pp, :] = 0
            elif mode == "shuffle_think":
                src_rows = perm[rr]
                src_pos = positions[src_rows]
                hidden[rr, pp, :] = source[src_rows, src_pos, :]
            else:
                raise ValueError(mode)
            if rest is None:
                return hidden
            return (hidden,) + rest

        return hook

    for idx, block in enumerate(model.transformer.h):
        handles.append(block.register_forward_hook(make_hook(idx)))
    return handles


def forward_condition(model, batch, condition: str, source_acts=None, perm=None):
    if condition == "normal":
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        return out.logits
    if condition == "zero_think_embedding":
        emb = model.get_input_embeddings()(batch["input_ids"]).clone()
        rows = torch.arange(emb.size(0), device=emb.device)
        valid = batch["positions"] >= 0
        emb[rows[valid], batch["positions"][valid], :] = 0
        out = model(inputs_embeds=emb, attention_mask=batch["attention_mask"])
        return out.logits
    if condition == "remove_think":
        attention = batch["attention_mask"].clone()
        rows = torch.arange(attention.size(0), device=attention.device)
        valid = batch["positions"] >= 0
        attention[rows[valid], batch["positions"][valid]] = 0
        out = model(input_ids=batch["input_ids"], attention_mask=attention)
        return out.logits
    if condition in {"zero_think", "shuffle_think"}:
        handles = patch_block_outputs(model, source_acts, batch["positions"], condition, perm)
        try:
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            return out.logits
        finally:
            for h in handles:
                h.remove()
    raise ValueError(condition)


def evaluate_group(group: str, examples: List[Example], args, device: torch.device) -> Dict:
    model, tok, ckpt_dir = load_group_model(group, args.run_root, args.model_path, device)
    conditions = ["normal", "zero_think_embedding", "zero_think", "shuffle_think", "remove_think"]
    per_condition = {c: {"cot": [], "answer": []} for c in conditions}
    sample_rows = []
    rng = random.Random(args.seed)

    with torch.no_grad():
        for start in range(0, len(examples), args.batch_size):
            recs = examples[start:start + args.batch_size]
            batch = make_main_batch(tok, recs, group, args.max_length, device, train=False)
            normal_logits, source_acts = clean_layer_acts(model, batch)
            perm_idx = list(range(len(recs)))
            rng.shuffle(perm_idx)
            if len(perm_idx) > 1 and all(i == p for i, p in enumerate(perm_idx)):
                perm_idx = perm_idx[1:] + perm_idx[:1]
            perm = torch.tensor(perm_idx, dtype=torch.long, device=device)

            logits_by_condition = {"normal": normal_logits}
            for cond in conditions[1:]:
                logits_by_condition[cond] = forward_condition(model, batch, cond, source_acts, perm).detach()

            batch_metrics = {}
            for cond, logits in logits_by_condition.items():
                cot = masked_nll_per_sample(logits, batch["labels"], batch["cot_mask"])
                ans = masked_nll_per_sample(logits, batch["labels"], batch["answer_mask"])
                per_condition[cond]["cot"].extend(cot)
                per_condition[cond]["answer"].extend(ans)
                batch_metrics[cond] = (cot, ans)

            for local_i, ex in enumerate(recs):
                row = {
                    "sample_id": ex.sample_id,
                    "question": ex.question,
                    "gold_cot_prefix": ex.cot[:320],
                    "gold_answer": ex.answer,
                    "think_position": int(batch["positions"][local_i].detach().cpu()),
                    "shuffle_source_sample_id": recs[perm_idx[local_i]].sample_id,
                }
                for cond in conditions:
                    row[f"cot_nll_{cond}"] = batch_metrics[cond][0][local_i]
                    row[f"answer_nll_{cond}"] = batch_metrics[cond][1][local_i]
                sample_rows.append(row)

    summary = {
        "group": group,
        "checkpoint": str(ckpt_dir),
        "num_eval_samples": len(examples),
        "conditions": {},
        "deltas_vs_normal": {},
    }
    for cond in conditions:
        summary["conditions"][cond] = {
            "cot_nll": mean(per_condition[cond]["cot"]),
            "answer_nll": mean(per_condition[cond]["answer"]),
        }
    for cond in conditions[1:]:
        cot_delta = [x - y for x, y in zip(per_condition[cond]["cot"], per_condition["normal"]["cot"])]
        ans_delta = [x - y for x, y in zip(per_condition[cond]["answer"], per_condition["normal"]["answer"])]
        summary["deltas_vs_normal"][cond] = {
            "cot_delta": mean(cot_delta),
            "cot_delta_ci": bootstrap_ci(cot_delta, seed=args.seed),
            "answer_delta": mean(ans_delta),
            "answer_delta_ci": bootstrap_ci(ans_delta, seed=args.seed + 1),
            "cot_positive_fraction": mean([float(x > 0) for x in cot_delta]),
            "answer_positive_fraction": mean([float(x > 0) for x in ans_delta]),
        }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, sample_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--model-path", default=DEFAULT_BASE)
    ap.add_argument("--groups", default="G3,G4")
    ap.add_argument("--eval-samples", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=640)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    REPORT_DIR.mkdir(exist_ok=True)
    examples = load_split(args.run_root, args.eval_samples)
    all_summary = {
        "audit": "in_sequence_think_intervention",
        "run_root": str(args.run_root),
        "eval_samples": len(examples),
        "max_length": args.max_length,
        "intervention_note": "zero_think/shuffle_think patch each GPT2 block output at the THINK position; remove_think masks that key position while preserving sequence length.",
        "groups": {},
    }
    all_rows = []
    for group in [g.strip() for g in args.groups.split(",") if g.strip()]:
        summary, rows = evaluate_group(group, examples, args, device)
        all_summary["groups"][group] = summary
        for row in rows:
            row["group"] = group
        all_rows.extend(rows)

    out_json = REPORT_DIR / "gpt2_in_sequence_think_intervention_audit.json"
    out_jsonl = REPORT_DIR / "gpt2_in_sequence_think_intervention_examples.jsonl"
    out_md = REPORT_DIR / "gpt2_in_sequence_think_intervention_audit.md"
    out_json.write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = ["# GPT2 In-Sequence THINK Intervention Audit", ""]
    lines.append(f"Run root: `{args.run_root}`")
    lines.append("")
    lines.append("This audit loads only the final G3/G4 checkpoints. No training was run. It measures CoT and Answer NLL on the original `Question <THINK> CoT Answer` sequence.")
    lines.append("")
    lines.append("| group | condition | CoT NLL | Answer NLL | CoT delta vs normal | CoT delta CI | Answer delta vs normal | Answer delta CI |")
    lines.append("|---|---|---:|---:|---:|---|---:|---|")
    for group, gs in all_summary["groups"].items():
        normal = gs["conditions"]["normal"]
        lines.append(f"| {group} | normal | {normal['cot_nll']:.6f} | {normal['answer_nll']:.6f} | - | - | - | - |")
        for cond in ["zero_think_embedding", "zero_think", "shuffle_think", "remove_think"]:
            cm = gs["conditions"][cond]
            dm = gs["deltas_vs_normal"][cond]
            lines.append(f"| {group} | {cond} | {cm['cot_nll']:.6f} | {cm['answer_nll']:.6f} | {dm['cot_delta']:.6f} | [{dm['cot_delta_ci'][0]:.6f}, {dm['cot_delta_ci'][1]:.6f}] | {dm['answer_delta']:.6f} | [{dm['answer_delta_ci'][0]:.6f}, {dm['answer_delta_ci'][1]:.6f}] |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    for group, gs in all_summary["groups"].items():
        z = gs["deltas_vs_normal"]["zero_think"]
        sh = gs["deltas_vs_normal"]["shuffle_think"]
        rm = gs["deltas_vs_normal"]["remove_think"]
        if z["cot_delta_ci"][0] > 0 and sh["cot_delta_ci"][0] > 0:
            verdict = "PASS_IN_SEQUENCE_CAUSAL_THINK_SIGNAL"
            text = "zero/shuffle THINK significantly increases CoT NLL in sequence, so the old self-decode evaluator likely understated latent usage."
        else:
            verdict = "NO_STRONG_IN_SEQUENCE_CAUSAL_THINK_SIGNAL"
            text = "zero/shuffle THINK does not robustly increase CoT NLL, so adjacent training still does not make THINK a strong causal state."
        lines.append(f"- {group}: `{verdict}`. zero THINK CoT delta={z['cot_delta']:.6f}, shuffle THINK CoT delta={sh['cot_delta']:.6f}, remove THINK CoT delta={rm['cot_delta']:.6f}. {text}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(all_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
