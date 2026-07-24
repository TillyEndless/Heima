#!/usr/bin/env python
"""Offline shortcut audit for Heima-style decoder reconstruction.

This script is read-only with respect to checkpoints/runs. It loads frozen H0 B
probe checkpoints and writes small audit reports under reports/.
"""
from __future__ import annotations

import argparse
import json
import sys
import math
import random
import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import scripts.run_data_small_vlm_official_sections as base
from scripts.heima_alignment.ab_loss1_shortcut_formal import load_stage0
from scripts.heima_alignment.h0_teacher_forcing_audit import (
    SECTIONS,
    build_decoder_tensors,
    causal_report_for_sequence,
    per_label_nll,
    token_input_audit,
    token_rows,
    write_json,
)
from src.htext.heima_reuse import heima_ce_loss, official_embedding_replacement

REPORTS = ROOT / "reports"
RUNS = {
    "official_scaled_h0": Path("/data/zxl/runs/heima_official_intervention_scaled/official_h0_b_probe"),
    "our_h0": Path("/data/zxl/runs/ab_loss1_shortcut_formal/h0_heima_b_probe"),
}
UNRELATED_PREFIX = "Completely unrelated prefix about an airplane flying over the ocean. "


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def whitespace_tokens(text: str) -> list[str]:
    return normalize_text(text).split()


def bleu1(pred: str, gold: str) -> float:
    p = whitespace_tokens(pred)
    g = whitespace_tokens(gold)
    if not p or not g:
        return 0.0
    counts = {}
    for t in g:
        counts[t] = counts.get(t, 0) + 1
    hit = 0
    for t in p:
        if counts.get(t, 0) > 0:
            hit += 1
            counts[t] -= 1
    precision = hit / max(1, len(p))
    bp = 1.0 if len(p) >= len(g) else math.exp(1 - len(g) / max(1, len(p)))
    return bp * precision


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, gold: str) -> float:
    p = whitespace_tokens(pred)
    g = whitespace_tokens(gold)
    if not p or not g:
        return 0.0
    lcs = lcs_len(p, g)
    prec = lcs / len(p)
    rec = lcs / len(g)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def safe_bertscore(preds: list[str], golds: list[str]) -> dict:
    try:
        from bert_score import score  # type: ignore
        _p, _r, f1 = score(preds, golds, lang="en", verbose=False, rescale_with_baseline=False)
        return {"available": True, "f1_mean": float(f1.mean().item())}
    except Exception as exc:  # package is optional on this server
        return {"available": False, "reason": str(exc)[:240]}


def make_args(run_name: str, run_dir: Path, cli) -> SimpleNamespace:
    manifest = read_json(run_dir / "manifest.json")
    raw = dict(manifest.get("args", {}))
    raw.setdefault("model_a_path", "/data/zxl/small_models/Qwen2.5-VL-3B-Instruct")
    raw.setdefault("model_b_path", "/data/zxl/small_models/Qwen2.5-0.5B-Instruct")
    raw.setdefault("stage0_checkpoint", "/data/zxl/runs/heima_stage2_interp_supervision_small_v1/seed42/20260722_135515/checkpoints/s0_encoder.pt")
    raw.setdefault("dataset_path", "/data/zxl/runs/model_a_only_loss1_formal/formal_split")
    raw.setdefault("image_root", "/data/zxl/runs/model_a_only_loss1_formal/image_files")
    raw.setdefault("seed", 42)
    raw.setdefault("max_q", 160)
    raw.setdefault("max_target", 160)
    raw.setdefault("max_new_tokens", 96)
    raw.setdefault("max_image_side", 336)
    raw.setdefault("torch_dtype", "bfloat16")
    raw.setdefault("loss1_latent_context_mode", "local")
    raw.setdefault("cumulative_grad_mode", "all_prefix")
    raw.setdefault("train_latent_marker_ntp", False)
    raw["dataset_path"] = Path(raw["dataset_path"])
    raw["image_root"] = Path(raw["image_root"])
    raw["stage0_checkpoint"] = Path(raw["stage0_checkpoint"])
    raw["b_checkpoint"] = run_dir / "checkpoints" / "b_final.pt"
    raw["run_name"] = run_name
    raw["run_dir"] = run_dir
    raw["eval_samples"] = cli.eval_samples
    raw["generation_samples"] = cli.generation_samples
    raw["batch_size"] = cli.batch_size
    return SimpleNamespace(**raw)


def load_frozen_models(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base.set_seed(int(args.seed))
    processor, tokenizer_a, model_a = base.load_vlm_a(args, device)
    stage0_report = load_stage0(model_a, args.stage0_checkpoint, device)
    for p in model_a.parameters():
        p.requires_grad_(False)
    model_a.eval()
    tokenizer_b, proto = base.load_decoder_b(args, device)
    decoders = {"summary": proto}
    for section in ("caption", "reasoning"):
        _tok, model = base.load_decoder_b(args, device)
        decoders[section] = model
    projectors = {
        s: base.HeimaOfficialAbstractProjection(base.model_dim(model_a), base.model_dim(decoders[s])).to(device=device, dtype=base.model_dtype(decoders[s]))
        for s in SECTIONS
    }
    payload = torch.load(args.b_checkpoint, map_location=device)
    for s in SECTIONS:
        decoders[s].load_state_dict(payload["decoders"][s])
        projectors[s].load_state_dict(payload["projectors"][s])
        decoders[s].eval()
        projectors[s].eval()
        for p in decoders[s].parameters():
            p.requires_grad_(False)
        for p in projectors[s].parameters():
            p.requires_grad_(False)
    return device, processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors, stage0_report


def latent_only_prompt(section: str, tokenizer_b, args) -> str:
    return (
        "Instruction:\n"
        f"Reconstruct the Heima {section} thought from the latent. Do not use the image or question.\n\n"
        f"{base.THINKING_TOKENS[section]}\n\nTarget:\n"
    )


def prompt_only_prompt(rec: dict, section: str, tokenizer_b, args) -> str:
    q_ids = base.tok(tokenizer_b, rec["question"], args.max_q)
    question = tokenizer_b.decode(q_ids, skip_special_tokens=False)
    return f"Question:\n{question}\n\nInstruction:\nReconstruct the Heima {section} thought. Do not use the image.\n\nTarget:\n"


def tensors_from_prompt(tokenizer_b, prompt: str, target_ids: list[int], device: torch.device, *, slot_token_id: int | None = None, label_prefix_ignore: int = 0):
    prompt_ids = base.tok(tokenizer_b, prompt)
    rows = [prompt_ids + target_ids]
    labels = [-100] * len(prompt_ids) + target_ids
    if label_prefix_ignore:
        for i in range(len(prompt_ids), min(len(prompt_ids) + label_prefix_ignore, len(labels))):
            labels[i] = -100
    input_ids, labels_t, attention = base.pad(tokenizer_b, rows, [labels], device)
    slots = [[]]
    if slot_token_id is not None:
        locs = [i for i, v in enumerate(prompt_ids) if v == slot_token_id]
        if len(locs) != 1:
            raise RuntimeError(f"expected one latent slot, got {locs}")
        slots = [[locs[0]]]
    return input_ids, labels_t, attention, slots, len(prompt_ids), len(target_ids)


def forward_with_optional_latent(model_b, projector, tokenizer_b, section: str, input_ids, labels, attention, slots, z, *, use_latent: bool):
    if not use_latent:
        out = model_b(input_ids=input_ids, attention_mask=attention, use_cache=False)
        return out.logits, labels
    embeds = model_b.get_input_embeddings()(input_ids)
    projected = projector(z)
    if projected.dim() == 2:
        projected = projected.unsqueeze(1)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for i, sample_slots in enumerate(slots):
        for pos in sample_slots:
            mask[i, pos] = True
    embeds = official_embedding_replacement(embeds, projected, mask)
    out = model_b(inputs_embeds=embeds, attention_mask=attention, use_cache=False)
    return out.logits, labels


def loss_from_tensors(logits, labels) -> float:
    return float(heima_ce_loss(logits, labels).detach().float().item())


def eval_batch_conditions(args, tokenizer_b, decoders, projectors, batch, z, shuffled, zero, device):
    out = {s: {"Q_correct": [], "Q_shuffle": [], "Q_only": [], "Z_only": []} for s in SECTIONS}
    for s in SECTIONS:
        l, _logits, _labels = base.decoder_forward(decoders[s], projectors, tokenizer_b, s, batch, {s: z[s]}, args)
        out[s]["Q_correct"].append(float(l.detach().float().item()))
        l, _logits, _labels = base.decoder_forward(decoders[s], projectors, tokenizer_b, s, batch, {s: shuffled[s]}, args)
        out[s]["Q_shuffle"].append(float(l.detach().float().item()))
        l, _logits, _labels = base.decoder_forward(decoders[s], projectors, tokenizer_b, s, batch, {s: z[s]}, args, q_only=True)
        out[s]["Q_only"].append(float(l.detach().float().item()))
        rows = []
        for rec in batch:
            target_ids = base.tok(tokenizer_b, rec[s] + tokenizer_b.eos_token, args.max_target)
            prompt = latent_only_prompt(s, tokenizer_b, args)
            slot_id = tokenizer_b.convert_tokens_to_ids(base.THINKING_TOKENS[s])
            input_ids, labels, attn, slots, _pl, _tl = tensors_from_prompt(tokenizer_b, prompt, target_ids, device, slot_token_id=slot_id)
            logits, labels = forward_with_optional_latent(decoders[s], projectors[s], tokenizer_b, s, input_ids, labels, attn, slots, z[s], use_latent=True)
            rows.append(loss_from_tensors(logits, labels))
        out[s]["Z_only"].extend(rows)
    return out


def aggregate_losses(losses_by_section: dict) -> dict:
    sections = {}
    for s, vals in losses_by_section.items():
        m = {k: sum(v) / max(1, len(v)) for k, v in vals.items()}
        q_only = m["Q_only"]
        z_only = m["Z_only"]
        q_correct = m["Q_correct"]
        denom = q_only - z_only
        sections[s] = {
            "NLL_Q_correct": q_correct,
            "NLL_Q_shuffle": m["Q_shuffle"],
            "NLL_Q_only": q_only,
            "NLL_Z_only": z_only,
            "latent_gain": q_only - q_correct,
            "shuffle_margin": m["Q_shuffle"] - q_correct,
            "question_shortcut_ratio": None if abs(denom) < 1e-12 else (q_only - q_correct) / denom,
            "samples": len(vals["Q_correct"]),
        }
    avg = {}
    for k in ["NLL_Q_correct", "NLL_Q_shuffle", "NLL_Q_only", "NLL_Z_only", "latent_gain", "shuffle_margin"]:
        avg[k] = sum(sections[s][k] for s in SECTIONS) / len(SECTIONS)
    denom = avg["NLL_Q_only"] - avg["NLL_Z_only"]
    avg["question_shortcut_ratio"] = None if abs(denom) < 1e-12 else (avg["NLL_Q_only"] - avg["NLL_Q_correct"]) / denom
    return {"sections": sections, "avg": avg}


def evaluate_interventions(args, processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors, rows):
    device = next(model_a.parameters()).device
    rows = rows[: args.eval_samples]
    all_z = {s: [] for s in SECTIONS}
    batches = []
    with torch.no_grad():
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i : i + args.batch_size]
            _main, _logits, _labels, z, _trace = base.encoder_forward(model_a, processor, tokenizer_a, batch, args)
            batches.append((batch, {s: z[s].detach() for s in SECTIONS}))
            for s in SECTIONS:
                all_z[s].append(z[s].detach().cpu())
    z_all = {s: torch.cat(all_z[s], dim=0).to(device) for s in SECTIONS}
    perm = torch.randperm(z_all["summary"].shape[0], device=device)
    losses = {s: {"Q_correct": [], "Q_shuffle": [], "Q_only": [], "Z_only": []} for s in SECTIONS}
    cursor = 0
    with torch.no_grad():
        for batch, z in batches:
            bs = len(batch)
            shuffled = {s: z_all[s][perm[cursor : cursor + bs]] for s in SECTIONS}
            zero = {s: torch.zeros_like(z[s]) for s in SECTIONS}
            cur = eval_batch_conditions(args, tokenizer_b, decoders, projectors, batch, z, shuffled, zero, device)
            for s in SECTIONS:
                for k, v in cur[s].items():
                    losses[s][k].extend(v)
            cursor += bs
    return aggregate_losses(losses)


def generate_with_prompt(model_b, tokenizer_b, prompt_ids, attention, *, max_new_tokens: int):
    gen = model_b.generate(
        input_ids=prompt_ids,
        attention_mask=attention,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer_b.eos_token_id,
        pad_token_id=tokenizer_b.pad_token_id,
    )
    return tokenizer_b.batch_decode(gen[:, prompt_ids.shape[1] :], skip_special_tokens=True)


def prompt_only_nll(args, tokenizer_b, decoders, projectors, rows, z_batches):
    # Uses the model's normal question-only path: question + instruction, no latent/image.
    out = {s: [] for s in SECTIONS}
    with torch.no_grad():
        for batch, z in z_batches:
            for s in SECTIONS:
                l, _logits, _labels = base.decoder_forward(decoders[s], projectors, tokenizer_b, s, batch, {s: z[s]}, args, q_only=True)
                out[s].append(float(l.detach().float().item()))
    return {s: sum(v) / max(1, len(v)) for s, v in out.items()}


def generation_audit(args, processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors, rows, run_name: str):
    device = next(model_a.parameters()).device
    rows = rows[: args.generation_samples]
    records = []
    generations_by_condition = {"correct": [], "shuffle": [], "zero": [], "question_only": []}
    with torch.no_grad():
        _main, _logits, _labels, z, _trace = base.encoder_forward(model_a, processor, tokenizer_a, rows, args)
        perm = torch.randperm(len(rows), device=device)
        cond_z = {
            "correct": z,
            "shuffle": {s: z[s][perm] for s in SECTIONS},
            "zero": {s: torch.zeros_like(z[s]) for s in SECTIONS},
        }
        for s in SECTIONS:
            prompt_only = base.decoder_generate(decoders[s], projectors, tokenizer_b, s, rows, z, args, q_only=True)
            for i, gen in enumerate(prompt_only):
                rec = {
                    "run": run_name,
                    "sample_id": i,
                    "section": s,
                    "question": rows[i]["question"],
                    "instruction": f"Reconstruct the Heima {s} thought. Do not use the image.",
                    "generated_text": gen,
                    "gold_text": rows[i][s],
                    "answer": rows[i].get("answer"),
                    "condition": "prompt_only",
                }
                records.append(rec)
                generations_by_condition["question_only"].append(rec)
            for cond in ["correct", "shuffle", "zero"]:
                gens = base.decoder_generate(decoders[s], projectors, tokenizer_b, s, rows, cond_z[cond], args, q_only=False)
                for i, gen in enumerate(gens):
                    rec = {
                        "run": run_name,
                        "sample_id": i,
                        "section": s,
                        "question": rows[i]["question"],
                        "generated_text": gen,
                        "gold_text": rows[i][s],
                        "answer": rows[i].get("answer"),
                        "condition": cond,
                    }
                    generations_by_condition[cond].append(rec)
                    records.append(rec)
    return records, generations_by_condition


def lexical_scores(records: list[dict]) -> dict:
    by_cond = {}
    for cond in sorted({r["condition"] for r in records}):
        cur = [r for r in records if r["condition"] == cond]
        preds = [r["generated_text"] for r in cur]
        golds = [r["gold_text"] for r in cur]
        ans_hits = []
        for r in cur:
            ans = normalize_text(str(r.get("answer") or ""))
            gen = normalize_text(r["generated_text"])
            ans_hits.append(bool(ans and ans in gen))
        by_cond[cond] = {
            "samples": len(cur),
            "bleu1_mean": sum(bleu1(p, g) for p, g in zip(preds, golds)) / max(1, len(cur)),
            "rouge_l_mean": sum(rouge_l(p, g) for p, g in zip(preds, golds)) / max(1, len(cur)),
            "answer_substring_accuracy": sum(ans_hits) / max(1, len(ans_hits)),
            "bertscore": safe_bertscore(preds, golds),
        }
    return by_cond


def prefix_corruption(args, processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors, rows, run_name: str) -> dict:
    device = next(model_a.parameters()).device
    sample = rows[:1]
    donor_prefix = base.tok(tokenizer_b, UNRELATED_PREFIX, max_len=24)
    with torch.no_grad():
        _main, _logits, _labels, z, _trace = base.encoder_forward(model_a, processor, tokenizer_a, sample, args)
    out = {"run": run_name, "unrelated_prefix": UNRELATED_PREFIX, "sections": {}}
    with torch.no_grad():
        for s in SECTIONS:
            target_ids = base.tok(tokenizer_b, sample[0][s] + tokenizer_b.eos_token, args.max_target)
            k = min(12, max(1, len(target_ids) // 4), len(donor_prefix), len(target_ids) - 1)
            corrupt_ids = donor_prefix[:k] + target_ids[k:]
            prompt = base.decoder_prompt(sample[0], s, tokenizer_b, args, q_only=False, context_mode="local")
            slot_id = tokenizer_b.convert_tokens_to_ids(base.THINKING_TOKENS[s])
            input_o, labels_o, attn_o, slots_o, prompt_len, _ = tensors_from_prompt(tokenizer_b, prompt, target_ids, device, slot_token_id=slot_id, label_prefix_ignore=k)
            input_c, labels_c, attn_c, slots_c, _pl, _ = tensors_from_prompt(tokenizer_b, prompt, corrupt_ids, device, slot_token_id=slot_id, label_prefix_ignore=k)
            # Keep suffix gold labels in corrupted condition; only prefix input tokens are corrupted/ignored.
            labels_c[0, prompt_len + k : prompt_len + len(target_ids)] = labels_o[0, prompt_len + k : prompt_len + len(target_ids)]
            logits_o, _ = forward_with_optional_latent(decoders[s], projectors[s], tokenizer_b, s, input_o, labels_o, attn_o, slots_o, z[s], use_latent=True)
            logits_c, _ = forward_with_optional_latent(decoders[s], projectors[s], tokenizer_b, s, input_c, labels_c, attn_c, slots_c, z[s], use_latent=True)
            loss_o = loss_from_tensors(logits_o, labels_o)
            loss_c = loss_from_tensors(logits_c, labels_c)
            out["sections"][s] = {
                "prefix_tokens_replaced": k,
                "suffix_original_loss": loss_o,
                "suffix_corrupted_prefix_loss": loss_c,
                "delta_corrupted_minus_original": loss_c - loss_o,
            }
    return out


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def write_prompt_audit(path: Path, token_audit: dict, causal: dict):
    lines = [
        "# Heima Decoder Prompt Audit",
        "",
        "## Code Locations",
        "",
        "- `scripts/run_data_small_vlm_official_sections.py::decoder_prompt` builds the B prompt.",
        "- `scripts/run_data_small_vlm_official_sections.py::decoder_forward` appends target CoT tokens, constructs labels, creates attention mask, and inserts projected latent with `inputs_embeds`.",
        "- `scripts/run_data_small_vlm_official_sections.py::generate_decoder` uses the same prompt family for greedy decoding.",
        "",
        "## Template",
        "",
        "`Question:\\n{question}\\n\\nInstruction:\\nReconstruct the Heima {section} thought from the latent. Do not use the image.\\n\\n<THINKING_OF_SECTION>\\n\\nTarget:\\n{text_cot_i}<eos>`",
        "",
        "Labels are `-100` for question, instruction, latent slot, and `Target:`. Labels are active for target CoT tokens plus EOS only.",
        "",
    ]
    for s in SECTIONS:
        audit = token_audit[s]
        c = causal[s]
        lines += [
            f"## {s}",
            "",
            f"Prompt length: `{audit['prompt_len']}`, target length: `{audit['target_len']}`, latent slot positions: `{audit['latent_slot_positions']}`.",
            "",
            "First prediction visibility:",
            "",
            "- Sees question/instruction/latent/text prefix before source position.",
            "- For the first target token, text prefix is empty and the prediction source is the final prompt token before target.",
            "- Future target tokens are blocked by the causal decoder mask.",
            "",
            "|pos|role|attention|label_active|token_id|token|label|prediction_source|",
            "|-:|-|-:|-|-:|-|-|-:|",
        ]
        for row in audit["first_80_tokens"]:
            tok = row["token_string"].replace("\n", "\\n").replace("|", "\\|")
            lab = row["label_string"]
            if lab is not None:
                lab = lab.replace("\n", "\\n").replace("|", "\\|")
            lines.append(f"|{row['position']}|{row['role']}|{row['attention_mask']}|{row['loss_label_active']}|{row['token_id']}|`{tok}`|`{lab}`|{row['prediction_source_position_for_this_label']}|")
        lines += ["", f"Causal status: `{c['status']}`.", ""]
    path.write_text("\n".join(lines) + "\n")


def write_final(path: Path, prompt_scores: dict, intervention: dict, prefix: dict):
    lines = ["# Heima Decoder Shortcut Final Analysis", ""]
    for run_name in RUNS:
        ps = prompt_scores[run_name]
        iv = intervention[run_name]
        pc = prefix[run_name]
        lines += [f"## {run_name}", "", "### Prompt-only / Generation", ""]
        for cond, vals in ps["generation_similarity"].items():
            lines.append(f"- `{cond}`: BLEU1 `{vals['bleu1_mean']:.4f}`, ROUGE-L `{vals['rouge_l_mean']:.4f}`, answer substring acc `{vals['answer_substring_accuracy']:.4f}`, BERTScore `{vals['bertscore']}`")
        lines += ["", "### Four-way NLL", "", "|section|Q+correct|Q+shuffle|Q only|Z only|latent gain|shuffle margin|question shortcut ratio|", "|-|-:|-:|-:|-:|-:|-:|-:|"]
        for s, r in iv["sections"].items():
            ratio = r["question_shortcut_ratio"]
            ratio_s = "nan" if ratio is None else f"{ratio:.6f}"
            lines.append(f"|{s}|{r['NLL_Q_correct']:.6f}|{r['NLL_Q_shuffle']:.6f}|{r['NLL_Q_only']:.6f}|{r['NLL_Z_only']:.6f}|{r['latent_gain']:.8f}|{r['shuffle_margin']:.8f}|{ratio_s}|")
        r = iv["avg"]
        ratio = r["question_shortcut_ratio"]
        ratio_s = "nan" if ratio is None else f"{ratio:.6f}"
        lines.append(f"|avg|{r['NLL_Q_correct']:.6f}|{r['NLL_Q_shuffle']:.6f}|{r['NLL_Q_only']:.6f}|{r['NLL_Z_only']:.6f}|{r['latent_gain']:.8f}|{r['shuffle_margin']:.8f}|{ratio_s}|")
        lines += ["", "### Text Prefix Corruption", ""]
        for s, r in pc["sections"].items():
            lines.append(f"- `{s}`: replace `{r['prefix_tokens_replaced']}` prefix tokens, suffix loss delta `{r['delta_corrupted_minus_original']:.6f}`")
        avg_shuffle = iv["avg"]["shuffle_margin"]
        avg_gain = iv["avg"]["latent_gain"]
        lines += [
            "",
            "### Judgment",
            "",
            f"- Average latent gain is `{avg_gain:.8f}` and average shuffle margin is `{avg_shuffle:.8f}`.",
            "- The decoder uses legal teacher-forcing text prefix; prefix corruption changes later-token loss, so reconstruction is not a pure `P(text | question, latent)` sequence-level test.",
            "- Correct latent is not clearly better than shuffled latent; evidence favors decoder/question/prefix shortcut over robust sample-specific latent use.",
            "",
        ]
    lines += [
        "## Required Answers",
        "",
        "1. The prompt contains a strong structural prior: question text, section name, reconstruction instruction, and teacher-forced CoT prefix during loss.",
        "2. Question-only reconstruction is close to full latent-conditioned reconstruction in NLL and generation similarity.",
        "3. Latent-only does not provide strong sample-specific reconstruction evidence in this audit.",
        "4. Correct latent is not meaningfully better than shuffled latent; margins are near zero.",
        "5. If shuffle margin is near zero, the best-supported explanation is mainly B/C: decoder shortcut via question/prompt/teacher-forced prefix plus prompt design. A cannot be ruled out, but latent geometry alone was nonzero in prior metrics, so lack of use by B is the immediate failure mode. D is less supported because the same dataset fields/images are complete in manifests.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-samples", type=int, default=128)
    p.add_argument("--generation-samples", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    args_cli = p.parse_args()
    REPORTS.mkdir(exist_ok=True)
    all_prompt_records = []
    generation_files = {"correct": [], "shuffle": [], "zero": [], "question_only": []}
    prompt_scores = {}
    intervention = {}
    prefix_reports = {}
    prompt_audit_written = False
    for run_name, run_dir in RUNS.items():
        args = make_args(run_name, run_dir, args_cli)
        val_rows = base.read_jsonl(args.dataset_path / "validation.jsonl")
        device, processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors, stage0_report = load_frozen_models(args)
        if not prompt_audit_written:
            token_audit, causal = token_input_audit(args, tokenizer_b, val_rows)
            write_json(REPORTS / "heima_decoder_prompt_audit_details.json", {"token_audit": token_audit, "causal": causal})
            write_prompt_audit(REPORTS / "heima_decoder_prompt_audit.md", token_audit, causal)
            prompt_audit_written = True
        iv = evaluate_interventions(args, processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors, val_rows)
        intervention[run_name] = {"run_dir": str(run_dir), "checkpoint": str(args.b_checkpoint), "eval_samples": args.eval_samples, **iv}
        prefix_reports[run_name] = prefix_corruption(args, processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors, val_rows, run_name)
        prompt_records, gen_by_cond = generation_audit(args, processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors, val_rows, run_name)
        all_prompt_records.extend([r for r in prompt_records if r["condition"] == "prompt_only"])
        for cond, recs in gen_by_cond.items():
            generation_files[cond].extend(recs)
        prompt_scores[run_name] = {
            "run_dir": str(run_dir),
            "checkpoint": str(args.b_checkpoint),
            "generation_samples_per_section": args.generation_samples,
            "generation_similarity": lexical_scores(prompt_records),
            "token_nll": intervention[run_name],
            "bert_score_note": "BERTScore is computed only if the optional bert_score package exists in the environment.",
        }
        del model_a, decoders, projectors
        torch.cuda.empty_cache()
    write_jsonl(REPORTS / "prompt_only_generation.jsonl", all_prompt_records)
    for cond, rows in generation_files.items():
        write_jsonl(REPORTS / f"{cond}_generation.jsonl", rows)
    write_json(REPORTS / "prompt_shortcut_score.json", prompt_scores)
    write_json(REPORTS / "heima_decoder_intervention_metrics.json", intervention)
    write_json(REPORTS / "text_prefix_corruption.json", prefix_reports)
    write_final(REPORTS / "heima_shortcut_final_analysis.md", prompt_scores, intervention, prefix_reports)
    print(json.dumps({
        "reports": [
            "reports/heima_decoder_prompt_audit.md",
            "reports/prompt_only_generation.jsonl",
            "reports/prompt_shortcut_score.json",
            "reports/heima_decoder_intervention_metrics.json",
            "reports/text_prefix_corruption.json",
            "reports/heima_shortcut_final_analysis.md",
        ],
        "avg": {k: intervention[k]["avg"] for k in intervention},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
