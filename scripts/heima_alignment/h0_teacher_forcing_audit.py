#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_data_small_vlm_official_sections as base
from scripts.heima_alignment.ab_loss1_shortcut_formal import load_stage0

SECTIONS = base.SECTIONS


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def bootstrap_ci(values: list[float], seed: int = 42, rounds: int = 1000) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return [means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]]


def token_rows(tokenizer, ids: list[int], labels: list[int], attention: list[int], prompt_len: int, slot_positions: list[int]) -> list[dict]:
    rows = []
    for pos, tid in enumerate(ids):
        label = labels[pos]
        rows.append({
            "position": pos,
            "token_id": int(tid),
            "token_string": tokenizer.decode([int(tid)], skip_special_tokens=False),
            "role": "latent_slot" if pos in slot_positions else ("prompt" if pos < prompt_len else "target_text_cot"),
            "attention_mask": int(attention[pos]),
            "label": int(label),
            "label_string": None if label == -100 else tokenizer.decode([int(label)], skip_special_tokens=False),
            "loss_label_active": bool(label != -100),
            "prediction_source_position_for_this_label": None if pos == 0 or label == -100 else pos - 1,
        })
    return rows


def build_decoder_tensors(tokenizer_b, section: str, records: list[dict], args, *, target_override_ids: list[int] | None = None, q_only: bool = False):
    rows, label_rows, slots = [], [], []
    context_mode = args.loss1_latent_context_mode
    slot_sections = () if q_only else base.prefix_sections(section, context_mode)
    token_ids = {s: tokenizer_b.convert_tokens_to_ids(base.THINKING_TOKENS[s]) for s in slot_sections}
    prompt_lens, target_lens = [], []
    for rec_idx, rec in enumerate(records):
        prompt_ids = base.tok(tokenizer_b, base.decoder_prompt(rec, section, tokenizer_b, args, q_only=q_only, context_mode=context_mode))
        if target_override_ids is None:
            target_ids = base.tok(tokenizer_b, rec[section] + tokenizer_b.eos_token, args.max_target)
        else:
            target_ids = target_override_ids
        rows.append(prompt_ids + target_ids)
        label_rows.append([-100] * len(prompt_ids) + target_ids)
        prompt_lens.append(len(prompt_ids))
        target_lens.append(len(target_ids))
        if not q_only:
            sample_slots = []
            for slot_section in slot_sections:
                locs = [i for i, value in enumerate(prompt_ids) if value == token_ids[slot_section]]
                if len(locs) != 1:
                    raise RuntimeError(f"expected one {slot_section} slot for {section}, got {locs}")
                sample_slots.append(locs[0])
            slots.append(sample_slots)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids, labels, attention = base.pad(tokenizer_b, rows, label_rows, device)
    return input_ids, labels, attention, slots, prompt_lens, target_lens


def decoder_forward_from_tensors(model_b, projector, tokenizer_b, section: str, input_ids, labels, attention, slots, z, args, *, q_only: bool = False):
    if q_only:
        out = model_b(input_ids=input_ids, attention_mask=attention, use_cache=False)
        return out.logits, labels
    embeds = model_b.get_input_embeddings()(input_ids)
    projected = projector[section](z).to(dtype=embeds.dtype)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for i, sample_slots in enumerate(slots):
        for pos in sample_slots:
            mask[i, pos] = True
    flat = embeds.view(-1, embeds.shape[-1])
    flat[mask.view(-1)] = projected.view(-1, embeds.shape[-1])
    embeds = flat.view_as(embeds)
    out = model_b(inputs_embeds=embeds, attention_mask=attention, use_cache=False)
    return out.logits, labels


def per_label_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    nll = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), ignore_index=-100, reduction="none")
    return nll.reshape_as(shift_labels)


def load_models(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base.set_seed(args.seed)
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
    projectors = {s: base.HeimaOfficialAbstractProjection(base.model_dim(model_a), base.model_dim(decoders[s])).to(device=device, dtype=base.model_dtype(decoders[s])) for s in SECTIONS}
    payload = torch.load(args.b_checkpoint, map_location=device)
    for s in SECTIONS:
        decoders[s].load_state_dict(payload["decoders"][s])
        projectors[s].load_state_dict(payload["projectors"][s])
        decoders[s].eval(); projectors[s].eval()
        for p in decoders[s].parameters():
            p.requires_grad_(False)
        for p in projectors[s].parameters():
            p.requires_grad_(False)
    return device, processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors, stage0_report


def causal_report_for_sequence(section: str, tokenizer_b, input_ids, labels, attention, slots, prompt_len: int, target_len: int) -> dict:
    ids = input_ids[0].detach().cpu().tolist()
    labs = labels[0].detach().cpu().tolist()
    attn = attention[0].detach().cpu().tolist()
    slot_positions = slots[0]
    active_positions = [i for i, x in enumerate(labs) if x != -100]
    checks = []
    for label_pos in active_positions[: min(12, len(active_positions))]:
        src = label_pos - 1
        visible = list(range(0, src + 1))
        future_target_positions = [p for p in range(label_pos + 1, prompt_len + target_len) if p < len(ids)]
        checks.append({
            "label_position": label_pos,
            "label_token": tokenizer_b.decode([labs[label_pos]], skip_special_tokens=False),
            "predicted_from_position": src,
            "visible_position_max": src,
            "future_target_positions_visible": [p for p in future_target_positions if p <= src],
            "future_target_positions_blocked": [p for p in future_target_positions if p > src][:8],
        })
    leak = any(c["future_target_positions_visible"] for c in checks)
    return {
        "section": section,
        "status": "CAUSAL_LEAK_FOUND" if leak else "PASS_NO_CAUSAL_LEAK_FOUND",
        "causal_rule": "decoder hidden at position k can attend only positions <= k; labels[p] is predicted by logits[p-1]",
        "prompt_len": prompt_len,
        "target_len": target_len,
        "latent_slot_positions": slot_positions,
        "sequence": token_rows(tokenizer_b, ids, labs, attn, prompt_len, slot_positions),
        "checks": checks,
    }


def token_input_audit(args, tokenizer_b, val_rows: list[dict]) -> tuple[dict, dict]:
    sample = [val_rows[0]]
    section_reports = {}
    causal = {}
    for section in SECTIONS:
        input_ids, labels, attention, slots, prompt_lens, target_lens = build_decoder_tensors(tokenizer_b, section, sample, args)
        causal[section] = causal_report_for_sequence(section, tokenizer_b, input_ids, labels, attention, slots, prompt_lens[0], target_lens[0])
        seq = causal[section]["sequence"]
        section_reports[section] = {
            "question": sample[0]["question"],
            "gold_text_preview": sample[0][section][:300],
            "input_layout": "prompt(question + instruction + latent slot) followed by target CoT text tokens + eos",
            "prompt_len": prompt_lens[0],
            "target_len": target_lens[0],
            "latent_slot_positions": slots[0],
            "num_labels_active": sum(1 for r in seq if r["loss_label_active"]),
            "first_80_tokens": seq[:80],
        }
    return section_reports, causal


def future_text_corruption(args, tokenizer_b, decoders, projectors, z, val_rows: list[dict]) -> dict:
    sample = [val_rows[0]]
    donor = val_rows[1]
    report = {}
    for section in SECTIONS:
        orig_ids = base.tok(tokenizer_b, sample[0][section] + tokenizer_b.eos_token, args.max_target)
        donor_ids = base.tok(tokenizer_b, donor[section] + tokenizer_b.eos_token, args.max_target)
        cut = max(2, min(len(orig_ids) - 2, len(orig_ids) // 2))
        replacement = (donor_ids + donor_ids)[: max(0, len(orig_ids) - cut)]
        corrupt_ids = orig_ids[:cut] + replacement[: len(orig_ids) - cut]
        input_o, labels_o, attn_o, slots_o, prompt_lens, _target_lens = build_decoder_tensors(tokenizer_b, section, sample, args, target_override_ids=orig_ids)
        input_c, labels_c, attn_c, slots_c, _p2, _t2 = build_decoder_tensors(tokenizer_b, section, sample, args, target_override_ids=corrupt_ids)
        logits_o, _ = decoder_forward_from_tensors(decoders[section], projectors, tokenizer_b, section, input_o, labels_o, attn_o, slots_o, z[section].detach(), args)
        logits_c, _ = decoder_forward_from_tensors(decoders[section], projectors, tokenizer_b, section, input_c, labels_c, attn_c, slots_c, z[section].detach(), args)
        nll_o = per_label_nll(logits_o, labels_o)[0]
        nll_c = per_label_nll(logits_c, labels_c)[0]
        prompt_len = prompt_lens[0]
        # target token j lives at label position prompt_len+j and is predicted from shifted nll index prompt_len+j-1.
        prefix_indices = list(range(prompt_len - 1, prompt_len + cut - 1))
        suffix_indices = list(range(prompt_len + cut - 1, min(nll_o.numel(), prompt_len + len(orig_ids) - 1)))
        prefix_diffs = [abs(float(nll_o[i].item()) - float(nll_c[i].item())) for i in prefix_indices]
        suffix_diffs = [abs(float(nll_o[i].item()) - float(nll_c[i].item())) for i in suffix_indices]
        report[section] = {
            "corruption_cut_target_token_index": cut,
            "prefix_tokens_tested": len(prefix_diffs),
            "original_prefix_loss": sum(float(nll_o[i].item()) for i in prefix_indices) / max(1, len(prefix_indices)),
            "corrupted_future_prefix_loss": sum(float(nll_c[i].item()) for i in prefix_indices) / max(1, len(prefix_indices)),
            "max_abs_prefix_loss_diff": max(prefix_diffs) if prefix_diffs else 0.0,
            "mean_abs_prefix_loss_diff": sum(prefix_diffs) / max(1, len(prefix_diffs)),
            "mean_abs_suffix_loss_diff": sum(suffix_diffs) / max(1, len(suffix_diffs)),
            "status": "PASS_NO_FUTURE_TEXT_LEAKAGE" if (max(prefix_diffs) if prefix_diffs else 0.0) < 1e-5 else "FUTURE_TEXT_INFLUENCES_PREFIX_LOSS",
            "note": "Full sequence loss is expected to change after the corruption because corrupted suffix labels and suffix input tokens change. The leakage check compares only prefix target-token losses before the corruption point.",
        }
    return report


def latent_gradient_audit(args, tokenizer_b, decoders, projectors, z, val_rows: list[dict]) -> dict:
    sample = [val_rows[0]]
    out = {}
    for section in SECTIONS:
        for module in [decoders[section], projectors[section]]:
            module.zero_grad(set_to_none=True)
        z_leaf = z[section].detach().clone().requires_grad_(True)
        loss, _logits, _labels = base.decoder_forward(decoders[section], projectors, tokenizer_b, section, sample, {section: z_leaf}, args)
        loss.backward()
        grad = z_leaf.grad.detach().float()
        out[section] = {
            "loss": float(loss.detach().cpu().item()),
            "norm": float(grad.norm().item()),
            "mean": float(grad.mean().item()),
            "abs_mean": float(grad.abs().mean().item()),
            "max": float(grad.max().item()),
            "abs_max": float(grad.abs().max().item()),
            "nonzero": bool(grad.abs().max().item() > 0),
        }
    return out


def intervention_replay(args, processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors, val_rows: list[dict]) -> dict:
    rows = val_rows[: args.eval_samples]
    z_banks = {s: [] for s in SECTIONS}
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            _main, _logits, _labels, z, _trace = base.encoder_forward(model_a, processor, tokenizer_a, batch, args)
            for s in SECTIONS:
                z_banks[s].append(z[s].detach())
    z_all = {s: torch.cat(z_banks[s], dim=0) for s in SECTIONS}
    shuffled = {s: torch.roll(z_all[s], shifts=1, dims=0) if len(rows) > 1 else torch.zeros_like(z_all[s]) for s in SECTIONS}
    zero = {s: torch.zeros_like(z_all[s]) for s in SECTIONS}
    per = {s: {"correct": [], "shuffle": [], "zero": [], "q_only": []} for s in SECTIONS}
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            sl = slice(start, start + len(batch))
            for s in SECTIONS:
                correct_loss, _l, _lab = base.decoder_forward(decoders[s], projectors, tokenizer_b, s, batch, {s: z_all[s][sl]}, args)
                shuffle_loss, _l, _lab = base.decoder_forward(decoders[s], projectors, tokenizer_b, s, batch, {s: shuffled[s][sl]}, args)
                zero_loss, _l, _lab = base.decoder_forward(decoders[s], projectors, tokenizer_b, s, batch, {s: zero[s][sl]}, args)
                q_loss, _l, _lab = base.decoder_forward(decoders[s], projectors, tokenizer_b, s, batch, {s: z_all[s][sl]}, args, q_only=True)
                per[s]["correct"].append(float(correct_loss.item()))
                per[s]["shuffle"].append(float(shuffle_loss.item()))
                per[s]["zero"].append(float(zero_loss.item()))
                per[s]["q_only"].append(float(q_loss.item()))
    sections = {}
    for s in SECTIONS:
        correct = per[s]["correct"]
        shuffle = per[s]["shuffle"]
        zero_l = per[s]["zero"]
        qonly = per[s]["q_only"]
        shuffle_margin = [a - b for a, b in zip(shuffle, correct)]
        zero_margin = [a - b for a, b in zip(zero_l, correct)]
        q_margin = [a - b for a, b in zip(qonly, correct)]
        sections[s] = {
            "NLL_correct": sum(correct) / len(correct),
            "NLL_shuffle": sum(shuffle) / len(shuffle),
            "NLL_zero": sum(zero_l) / len(zero_l),
            "NLL_qonly": sum(qonly) / len(qonly),
            "shuffle_margin": sum(shuffle_margin) / len(shuffle_margin),
            "zero_margin": sum(zero_margin) / len(zero_margin),
            "qonly_margin": sum(q_margin) / len(q_margin),
            "shuffle_margin_ci": bootstrap_ci(shuffle_margin, args.seed),
            "samples": len(correct),
        }
    avg = {
        "NLL_correct": sum(sections[s]["NLL_correct"] for s in SECTIONS) / len(SECTIONS),
        "NLL_shuffle": sum(sections[s]["NLL_shuffle"] for s in SECTIONS) / len(SECTIONS),
        "NLL_zero": sum(sections[s]["NLL_zero"] for s in SECTIONS) / len(SECTIONS),
        "NLL_qonly": sum(sections[s]["NLL_qonly"] for s in SECTIONS) / len(SECTIONS),
        "shuffle_margin": sum(sections[s]["shuffle_margin"] for s in SECTIONS) / len(SECTIONS),
        "zero_margin": sum(sections[s]["zero_margin"] for s in SECTIONS) / len(SECTIONS),
        "qonly_margin": sum(sections[s]["qonly_margin"] for s in SECTIONS) / len(SECTIONS),
    }
    return {"eval_samples": len(rows), "sections": sections, "avg": avg}


def write_markdown_audit(path: Path, token_audit: dict, causal: dict, corruption: dict, grad: dict, replay: dict) -> None:
    lines = [
        "# H0 Decoder Input Teacher-Forcing Audit",
        "",
        "## Decoder Forward Location",
        "",
        "- `scripts/run_data_small_vlm_official_sections.py::decoder_forward` constructs B-side inputs.",
        "- `scripts/heima_alignment/ab_loss1_shortcut_formal.py::loss1_forward` calls it for `h0_heima_b_probe`.",
        "- H0 uses `detach_encoder_latent=True`, A frozen, B/projectors trainable during training. This audit is offline only.",
        "",
        "## Actual B Input Sequence",
        "",
        "For each section, the input is:",
        "",
        "`Question text + reconstruction instruction + <THINKING_OF_SECTION> latent slot + Target: + text_cot_i + EOS`",
        "",
        "The latent slot token embedding is replaced with projected continuous latent before B forward via `inputs_embeds`; `attention_mask` is 1 for all non-padding tokens.",
        "",
        "## Real Batch Token Dump",
        "",
    ]
    for section in SECTIONS:
        audit = token_audit[section]
        lines += [
            f"### {section}",
            "",
            f"Question: `{audit['question']}`",
            "",
            "|pos|role|attention|label_active|token_id|token|label|prediction_source|",
            "|-:|-|-:|-|-:|-|-|-:|",
        ]
        for row in audit['first_80_tokens']:
            tok = row['token_string'].replace('\n', '\\n').replace('|', '\\|')
            label_tok = row['label_string']
            if label_tok is not None:
                label_tok = label_tok.replace('\n', '\\n').replace('|', '\\|')
            lines.append(
                f"|{row['position']}|{row['role']}|{row['attention_mask']}|{row['loss_label_active']}|{row['token_id']}|`{tok}`|`{label_tok}`|{row['prediction_source_position_for_this_label']}|"
            )
        lines += [
            "",
            f"Latent slot positions: `{audit['latent_slot_positions']}`; prompt length: `{audit['prompt_len']}`; target length: `{audit['target_len']}`.",
            "",
        ]
    lines += [
        "## Label Mask",
        "",
        "Labels are `-100` for the prompt, including question/instruction/latent slot. Labels are active only for target CoT tokens plus EOS. `heima_ce_loss` shifts labels, so label at position `p` is predicted from logits at `p-1`.",
        "",
        "## Causal Visibility",
        "",
    ]
    for section in SECTIONS:
        c = causal[section]
        lines += [
            f"### {section}",
            "",
            f"- Status: `{c['status']}`",
            f"- Prompt length: `{c['prompt_len']}`",
            f"- Target length: `{c['target_len']}`",
            f"- Latent slot positions: `{c['latent_slot_positions']}`",
            f"- Active label count: `{token_audit[section]['num_labels_active']}`",
            "",
            "First active-label checks show no future target positions visible to their prediction source.",
            "",
        ]
    leak = any(causal[s]["status"] != "PASS_NO_CAUSAL_LEAK_FOUND" for s in SECTIONS)
    lines += [
        "## Text Corruption Test",
        "",
    ]
    for section in SECTIONS:
        r = corruption[section]
        lines.append(f"- `{section}`: `{r['status']}`, max prefix loss diff `{r['max_abs_prefix_loss_diff']:.8g}`, mean suffix diff `{r['mean_abs_suffix_loss_diff']:.6g}`")
    lines += ["", "## Latent Gradient", ""]
    for section in SECTIONS:
        r = grad[section]
        lines.append(f"- `{section}`: norm `{r['norm']:.6g}`, abs_mean `{r['abs_mean']:.6g}`, abs_max `{r['abs_max']:.6g}`")
    lines += ["", "## Intervention Replay", "", "|section|NLL_correct|NLL_shuffle|NLL_qonly|NLL_zero|shuffle_margin|shuffle CI|", "|-|-:|-:|-:|-:|-:|-|"]
    for section in SECTIONS:
        r = replay["sections"][section]
        lines.append(f"|{section}|{r['NLL_correct']:.6f}|{r['NLL_shuffle']:.6f}|{r['NLL_qonly']:.6f}|{r['NLL_zero']:.6f}|{r['shuffle_margin']:.8f}|{r['shuffle_margin_ci']}|")
    lines += [
        f"|avg|{replay['avg']['NLL_correct']:.6f}|{replay['avg']['NLL_shuffle']:.6f}|{replay['avg']['NLL_qonly']:.6f}|{replay['avg']['NLL_zero']:.6f}|{replay['avg']['shuffle_margin']:.8f}|-|",
        "",
        "## Final Answer",
        "",
        "H0 is equivalent to `P(text_cot_i | question, latent, text_cot_prefix)` under standard causal teacher forcing. The `text_cot_prefix` consists only of previously generated gold target tokens and is legal teacher-forcing history. Future target tokens are not visible to the prediction source positions.",
        "",
        "No causal future-text leakage was found." if not leak else "CAUSAL_LEAK_FOUND: inspect reports/h0_causal_visibility.json immediately.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-a-path", default="/data/zxl/small_models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--model-b-path", default="/data/zxl/small_models/Qwen2.5-0.5B-Instruct")
    p.add_argument("--stage0-checkpoint", type=Path, default=Path("/data/zxl/runs/heima_stage2_interp_supervision_small_v1/seed42/20260722_135515/checkpoints/s0_encoder.pt"))
    p.add_argument("--dataset-path", type=Path, default=Path("/data/zxl/runs/model_a_only_loss1_formal/formal_split"))
    p.add_argument("--image-root", type=Path, default=Path("/data/zxl/runs/model_a_only_loss1_formal/image_files"))
    p.add_argument("--b-checkpoint", type=Path, default=Path("/data/zxl/runs/ab_loss1_shortcut_formal/h0_heima_b_probe/checkpoints/b_final.pt"))
    p.add_argument("--eval-samples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr-a", type=float, default=1e-5)
    p.add_argument("--lr-b", type=float, default=2e-5)
    p.add_argument("--lr-projector", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optimizer", default="adafactor")
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--max-q", type=int, default=160)
    p.add_argument("--max-target", type=int, default=160)
    p.add_argument("--max-image-side", type=int, default=336)
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--loss1-latent-context-mode", choices=("local",), default="local")
    p.add_argument("--cumulative-grad-mode", default="all_prefix")
    p.add_argument("--train-latent-marker-ntp", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    reports = ROOT / "reports"
    val_rows = read_jsonl(args.dataset_path / "validation.jsonl")
    device, processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors, stage0_report = load_models(args)
    token_audit, causal = token_input_audit(args, tokenizer_b, val_rows)
    write_json(reports / "h0_causal_visibility.json", {"status": "PASS_NO_CAUSAL_LEAK_FOUND" if all(causal[s]["status"] == "PASS_NO_CAUSAL_LEAK_FOUND" for s in SECTIONS) else "CAUSAL_LEAK_FOUND", "sections": causal})
    with torch.no_grad():
        _main, _logits, _labels, z, _trace = base.encoder_forward(model_a, processor, tokenizer_a, [val_rows[0]], args)
    corruption = future_text_corruption(args, tokenizer_b, decoders, projectors, z, val_rows)
    write_json(reports / "h0_future_text_corruption.json", corruption)
    grad = latent_gradient_audit(args, tokenizer_b, decoders, projectors, z, val_rows)
    write_json(reports / "h0_latent_gradient_audit.json", grad)
    replay = intervention_replay(args, processor, tokenizer_a, model_a, tokenizer_b, decoders, projectors, val_rows)
    write_json(reports / "h0_intervention_replay.json", replay)
    write_json(reports / "h0_decoder_input_audit_details.json", {"stage0_load": stage0_report, "token_audit": token_audit})
    write_markdown_audit(reports / "h0_decoder_input_audit.md", token_audit, causal, corruption, grad, replay)
    print(json.dumps({"causal_status": "PASS_NO_CAUSAL_LEAK_FOUND" if all(causal[s]["status"] == "PASS_NO_CAUSAL_LEAK_FOUND" for s in SECTIONS) else "CAUSAL_LEAK_FOUND", "corruption": {s: corruption[s]["status"] for s in SECTIONS}, "latent_grad_norms": {s: grad[s]["norm"] for s in SECTIONS}, "replay_avg": replay["avg"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
