#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.htext.formal_eval import hidden_geometry
from src.htext.heima_reuse import (
    HeimaOfficialAbstractProjection,
    backend_resolution_snapshot,
    extract_thinking_state,
    heima_ce_loss,
    official_embedding_replacement,
)
from src.htext.trainer import _grad_norm, batch_records, set_seed


OUT = Path("reports/context_conditioned_heima")
CKPT = Path("checkpoints/context_conditioned_heima")
DATA = Path("experiments/htext_gpt2/data/context_conditioned_heima")
MODEL = "/mnt/nas/share2/home/zxl/models/openai-community-gpt2"
TOKENS = ["<THINKING_OF_REASONING_1>", "<THINKING_OF_REASONING_2>", "<THINKING_OF_REASONING_3>"]
ANSWER_PREFIX = "\nAnswer: "
NUM_RE = re.compile(r"-?\d+")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tok(tokenizer, text: str, max_len: int | None = None) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids[:max_len] if max_len else ids


def load_models(with_b: int = 0):
    kw = {"local_files_only": True, "use_safetensors": True}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, **kw)
    tokenizer.pad_token = tokenizer.eos_token
    models = [AutoModelForCausalLM.from_pretrained(MODEL, **kw)]
    for _ in range(with_b):
        models.append(AutoModelForCausalLM.from_pretrained(MODEL, **kw))
    added = tokenizer.add_special_tokens({"additional_special_tokens": TOKENS})
    if added:
        for model in models:
            model.resize_token_embeddings(len(tokenizer))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for model in models:
        model.config.use_cache = False
        model.to(device)
    return tokenizer, models


def make_context_record(kind: str, group_idx: int, variant_idx: int, split: str, seed: int) -> dict:
    rng = random.Random(seed * 100000 + group_idx * 97 + variant_idx * 13)
    gid = f"{split}_{kind}_{group_idx:04d}"
    rid = f"{gid}_{variant_idx}"
    if kind == "arithmetic_context":
        a, b, c = rng.randint(8, 80), rng.randint(5, 70), rng.randint(2, 18)
        inter = a + b
        ans = inter * c
        context = f"Context: The hidden values are alpha={a}, beta={b}, and scale={c}."
        question = "Using the hidden values, add alpha and beta, then multiply by scale. What is the final value?"
        cot1 = f"Use the context values to form (alpha + beta) * scale = ({a} + {b}) * {c}."
        cot2 = f"First alpha + beta = {a} + {b} = {inter}."
        cot3 = f"Then {inter} * {c} = {ans}, so the final value is {ans}."
        operation_type = "(a+b)*c"
        facts = [a, b, c, inter, ans]
    elif kind == "ordered_operation_context":
        x, y, z = rng.randint(6, 45), rng.randint(3, 24), rng.randint(2, 60)
        inter = x * y
        ans = inter - z
        context = f"Context: Rule card says start={x}, multiplier={y}, deduction={z}."
        question = "Apply the rule card: multiply start by multiplier, then subtract deduction. What remains?"
        cot1 = f"The ordered rule is start * multiplier - deduction = {x} * {y} - {z}."
        cot2 = f"Compute the product: {x} * {y} = {inter}."
        cot3 = f"Subtract the deduction: {inter} - {z} = {ans}."
        operation_type = "a*b-c"
        facts = [x, y, z, inter, ans]
    elif kind == "table_fact_context":
        red, blue, boxes = rng.randint(7, 55), rng.randint(6, 50), rng.randint(2, 12)
        inter = red + blue
        ans = inter * boxes
        context = f"Context: Inventory table: red={red}; blue={blue}; boxes={boxes}."
        question = "From the inventory table, combine red and blue counts, then pack the total into each box count. What total is packed?"
        cot1 = f"Read the table as (red + blue) * boxes = ({red} + {blue}) * {boxes}."
        cot2 = f"The combined count is {red} + {blue} = {inter}."
        cot3 = f"Packing across boxes gives {inter} * {boxes} = {ans}."
        operation_type = "table_(a+b)*c"
        facts = [red, blue, boxes, inter, ans]
    else:
        start, inc, days, loss = rng.randint(20, 90), rng.randint(3, 20), rng.randint(2, 10), rng.randint(5, 60)
        inter = inc * days
        ans = start + inter - loss
        context = f"Context: State log: initial={start}, daily_gain={inc}, days={days}, final_loss={loss}."
        question = "Use the state log: add total gain to the initial amount, then remove the final loss. What is the ending amount?"
        cot1 = f"The state update is initial + daily_gain * days - final_loss = {start} + {inc} * {days} - {loss}."
        cot2 = f"Total gain is {inc} * {days} = {inter}."
        cot3 = f"Ending amount is {start} + {inter} - {loss} = {ans}."
        operation_type = "a+b*c-d"
        facts = [start, inc, days, loss, inter, ans]
    return {
        "id": rid,
        "context": context,
        "question": question,
        "cot1": cot1,
        "cot2": cot2,
        "cot3": cot3,
        "answer": str(ans),
        "pair_group_id": gid,
        "operation_type": operation_type,
        "intermediate_results": [str(x) for x in facts],
        "context_facts": [str(x) for x in facts[:-1]],
        "split": split,
        "context_signature": f"{kind}:{','.join(map(str, facts))}",
    }


def generate_split(seed: int, split: str, groups_per_kind: int) -> list[dict]:
    rows = []
    kinds = ["arithmetic_context", "ordered_operation_context", "table_fact_context", "multi_step_state_update_context"]
    for kind in kinds:
        for g in range(groups_per_kind):
            seen = set()
            made = 0
            attempt = 0
            while made < 4:
                row = make_context_record(kind, g, attempt, split, seed + {"train": 0, "validation": 1000, "ood": 2000}[split])
                attempt += 1
                if row["context_signature"] in seen:
                    continue
                row["id"] = f"{row['pair_group_id']}_{made}"
                seen.add(row["context_signature"])
                rows.append(row)
                made += 1
    random.Random(seed + len(split)).shuffle(rows)
    return rows


def make_data(seed: int, train_groups: int, val_groups: int, ood_groups: int) -> dict:
    root = DATA / f"seed_{seed}"
    paths = {
        "train": root / "train.jsonl",
        "validation": root / "validation.jsonl",
        "ood": root / "ood.jsonl",
    }
    regenerate = not all(p.exists() for p in paths.values())
    if not regenerate:
        existing = {split: read_jsonl(path) for split, path in paths.items()}
        regenerate = not validate_pairs(existing)["passed"]
    if regenerate:
        write_jsonl(paths["train"], generate_split(seed, "train", train_groups))
        write_jsonl(paths["validation"], generate_split(seed, "validation", val_groups))
        write_jsonl(paths["ood"], generate_split(seed, "ood", ood_groups))
    return {k: str(v) for k, v in paths.items()}


def validate_pairs(rows_by_split: dict[str, list[dict]]) -> dict:
    out = {}
    signatures = defaultdict(set)
    ok = True
    for split, rows in rows_by_split.items():
        groups = defaultdict(list)
        for row in rows:
            groups[row["pair_group_id"]].append(row)
            signatures[split].add(row["context_signature"])
        bad = []
        for gid, members in groups.items():
            questions = {m["question"] for m in members}
            ops = {m["operation_type"] for m in members}
            answers = {m["answer"] for m in members}
            contexts = {m["context_signature"] for m in members}
            if len(members) < 4 or len(questions) != 1 or len(ops) != 1 or len(answers) < 2 or len(contexts) != len(members):
                bad.append(gid)
        ok = ok and not bad
        out[split] = {"num_samples": len(rows), "num_groups": len(groups), "bad_groups": bad[:10]}
    overlaps = {}
    for a in signatures:
        for b in signatures:
            if a < b:
                overlaps[f"{a}_vs_{b}"] = len(signatures[a] & signatures[b])
                ok = ok and overlaps[f"{a}_vs_{b}"] == 0
    out["context_signature_overlaps"] = overlaps
    out["passed"] = ok
    return out


def stage_text(record: dict, stage: int) -> str:
    return record[f"cot{stage}"]


def encoder_sequence(tokenizer, record: dict, mode) -> tuple[list[int], list[int]]:
    prefix = f"{record['context']}\nQuestion: {record['question']}\n"
    ids = tok(tokenizer, prefix)
    labels = [-100] * len(ids)
    token_ids = [tokenizer.convert_tokens_to_ids(t) for t in TOKENS]
    for i in range(3):
        if mode == "explicit" or i >= int(mode):
            part = tok(tokenizer, record[f"cot{i+1}"] + " ")
            ids += part
            labels += part
        else:
            ids.append(token_ids[i])
            labels.append(token_ids[i])
    ans = tok(tokenizer, ANSWER_PREFIX + record["answer"] + tokenizer.eos_token)
    ids += ans
    labels += ans
    return ids, labels


def pad_rows(tokenizer, rows: list[list[int]], labels: list[list[int]]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = max(len(x) for x in rows)
    input_ids = torch.full((len(rows), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    label_t = torch.full_like(input_ids, -100)
    attn = torch.zeros_like(input_ids)
    for i, row in enumerate(rows):
        input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        label_t[i, : len(labels[i])] = torch.tensor(labels[i], dtype=torch.long, device=device)
        attn[i, : len(row)] = 1
    return input_ids, label_t, attn


def encoder_forward(model, tokenizer, records: list[dict], mode):
    rows, labels = zip(*(encoder_sequence(tokenizer, r, mode) for r in records))
    input_ids, label_t, attn = pad_rows(tokenizer, list(rows), list(labels))
    out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
    loss = heima_ce_loss(out.logits, label_t)
    hiddens = []
    positions = {}
    for token in TOKENS:
        tid = tokenizer.convert_tokens_to_ids(token)
        if input_ids.eq(tid).any():
            state = extract_thinking_state(
                input_ids=input_ids,
                last_hidden_state=out.hidden_states[-1],
                thinking_token_id=tid,
                mode="predictor",
            )
            hiddens.append(state.hidden)
            positions[token] = {
                "thinking": state.thinking_positions.detach().cpu().tolist(),
                "selected": state.selected_positions.detach().cpu().tolist(),
            }
    hidden = torch.stack(hiddens, dim=1) if hiddens else torch.empty((len(records), 0, out.hidden_states[-1].size(-1)), device=out.logits.device)
    return loss, out.logits, label_t, hidden, positions


def answer_em(logits, labels) -> float:
    pred = logits[:, :-1].argmax(-1)
    lab = labels[:, 1:]
    vals = []
    for i in range(lab.size(0)):
        m = lab[i] != -100
        vals.append(bool(m.any() and torch.equal(pred[i][m], lab[i][m])))
    return sum(vals) / max(1, len(vals))


def train_a(model, tokenizer, records, mode, steps: int, batch_size: int, lr: float, out_dir: Path):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    logs = []
    for step in range(1, steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        batch = batch_records(records, batch_size, step - 1)
        loss, _, _, _, _ = encoder_forward(model, tokenizer, batch, mode)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite encoder loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in {1, steps} or step % max(1, steps // 4) == 0:
            logs.append({"step": step, "loss": float(loss.item()), "grad_A": _grad_norm(model.parameters())[0]})
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_a": model.state_dict(), "optimizer": opt.state_dict(), "rng": torch.get_rng_state(), "mode": mode}, out_dir / "checkpoint.pt")
    return logs


def eval_a(model, tokenizer, records, mode, batch_size: int) -> dict:
    model.eval()
    losses, ems, hs = [], [], []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            loss, logits, labels, hidden, _ = encoder_forward(model, tokenizer, batch, mode)
            losses.append(float(loss.item()) * len(batch))
            ems.append(answer_em(logits, labels) * len(batch))
            if hidden.numel():
                hs.append(hidden.detach().cpu())
    geom = {}
    if hs:
        all_h = torch.cat(hs, dim=0)
        for i in range(all_h.size(1)):
            geom[f"stage_{i+1}"] = hidden_geometry(all_h[:, i : i + 1, :])
    return {"nll": sum(losses) / max(1, len(records)), "answer_em": sum(ems) / max(1, len(records)), "geometry": geom}


def decoder_prompt(record: dict, stage: int, mode: str) -> str:
    if mode == "z":
        return f"Latent:\n{TOKENS[stage-1]}\n\nReasoning:\n"
    if mode == "q":
        return f"Question:\n{record['question']}\n\nStage instruction:\nReconstruct cot{stage} using only the question.\n\nReasoning:\n"
    return (
        f"Question:\n{record['question']}\n\n"
        f"Stage instruction:\nReconstruct cot{stage}. The context is not shown; use the latent slot.\n\n"
        f"Latent:\n{TOKENS[stage-1]}\n\nReasoning:\n"
    )


def decoder_forward(model_b, tokenizer, records, z, projector, stage: int, mode: str, metric_masks: bool = False):
    device = next(model_b.parameters()).device
    rows, labels, positions = [], [], []
    for record in records:
        prompt = decoder_prompt(record, stage, mode)
        pids = tok(tokenizer, prompt)
        tids = tok(tokenizer, stage_text(record, stage) + tokenizer.eos_token)
        rows.append(pids + tids)
        labels.append([-100] * len(pids) + tids)
        tid = tokenizer.convert_tokens_to_ids(TOKENS[stage - 1])
        locs = [i for i, x in enumerate(pids) if x == tid]
        positions.append(locs[0] if locs else None)
    input_ids, label_t, attn = pad_rows(tokenizer, rows, labels)
    if (label_t != -100).sum().item() == 0:
        raise RuntimeError("zero non-ignored decoder labels")
    embeds = model_b.get_input_embeddings()(input_ids)
    if mode != "q":
        projected = projector(z)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, pos in enumerate(positions):
            if pos is None:
                raise RuntimeError("missing typed thinking token in decoder prompt")
            mask[i, pos] = True
        embeds = official_embedding_replacement(embeds, projected.unsqueeze(1), mask)
    out = model_b(inputs_embeds=embeds, attention_mask=attn, use_cache=False)
    loss = heima_ce_loss(out.logits, label_t)
    if not metric_masks:
        return loss, out.logits, label_t
    return loss, out.logits, label_t


def token_metric_nll(tokenizer, logits, labels, records, stage: int) -> dict:
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    logp = F.log_softmax(shift_logits, dim=-1)
    valid = shift_labels != -100
    per = torch.zeros_like(shift_labels, dtype=torch.float)
    per[valid] = -logp[valid, shift_labels[valid]]
    masks = {k: torch.zeros_like(shift_labels, dtype=torch.bool) for k in ["numeric", "intermediate", "context_fact"]}
    for i, record in enumerate(records):
        decoded = []
        for pos in range(shift_labels.size(1)):
            if shift_labels[i, pos] != -100:
                decoded.append((pos, tokenizer.decode([int(shift_labels[i, pos])])))
        nums = set(record.get("intermediate_results", []))
        facts = set(record.get("context_facts", []))
        inter = set(record.get("intermediate_results", [])[:-1])
        for pos, text in decoded:
            if NUM_RE.search(text):
                masks["numeric"][i, pos] = True
            if any(x in text for x in inter):
                masks["intermediate"][i, pos] = True
            if any(x in text for x in facts):
                masks["context_fact"][i, pos] = True
    out = {"full": float(per[valid].mean().item())}
    for key, mask in masks.items():
        m = mask & valid
        out[key] = float(per[m].mean().item()) if m.any() else None
    return out


def make_paired_latent(z: torch.Tensor, records: list[dict]) -> torch.Tensor:
    out = z.clone()
    by_group = defaultdict(list)
    for i, row in enumerate(records):
        by_group[row["pair_group_id"]].append(i)
    for idxs in by_group.values():
        if len(idxs) > 1:
            vals = [out[i].clone() for i in idxs]
            for k, i in enumerate(idxs):
                out[i] = vals[(k + 1) % len(vals)]
    return out


def make_variants(z: torch.Tensor, records: list[dict]) -> dict[str, torch.Tensor]:
    n = z.size(0)
    perm = torch.roll(torch.arange(n, device=z.device), 1)
    flat = z.float()
    dist = torch.cdist(flat, flat)
    farthest = dist.argmax(dim=1)
    rnd = torch.randn_like(z)
    rnd = F.normalize(rnd, dim=-1) * z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return {
        "normal": z,
        "paired_shuffle": make_paired_latent(z, records),
        "random_shuffle": z[perm],
        "farthest": z[farthest],
        "zero": torch.zeros_like(z),
        "mean": z.mean(dim=0, keepdim=True).expand_as(z),
        "norm_random": rnd,
    }


def evaluate_interventions(model_b, tokenizer, projector, records, z, stage: int, batch_size: int) -> dict:
    model_b.eval()
    projector.eval()
    totals = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            zb = z[start : start + len(batch)]
            for name, zv in make_variants(zb, batch).items():
                loss, logits, labels = decoder_forward(model_b, tokenizer, batch, zv, projector, stage, "qz", True)
                metrics = token_metric_nll(tokenizer, logits, labels, batch, stage)
                for k, v in metrics.items():
                    if v is not None:
                        totals[f"{name}_{k}"].append(v)
            qloss, qlogits, qlabels = decoder_forward(model_b, tokenizer, batch, zb, projector, stage, "q", True)
            qmetrics = token_metric_nll(tokenizer, qlogits, qlabels, batch, stage)
            for k, v in qmetrics.items():
                if v is not None:
                    totals[f"q_only_{k}"].append(v)
            zloss, zlogits, zlabels = decoder_forward(model_b, tokenizer, batch, zb, projector, stage, "z", True)
            zmetrics = token_metric_nll(tokenizer, zlogits, zlabels, batch, stage)
            for k, v in zmetrics.items():
                if v is not None:
                    totals[f"z_only_{k}"].append(v)
    return {k: float(statistics.mean(v)) for k, v in sorted(totals.items()) if v}


def train_stage_interpreter(model_a, tokenizer, train, val, stage: int, steps: int, batch_size: int, lr: float, seed_dir: Path):
    for p in model_a.parameters():
        p.requires_grad_(False)
    _, models = load_models(with_b=1)
    model_b = models[1]
    projector = HeimaOfficialAbstractProjection(model_a.config.n_embd, model_b.config.n_embd).to(next(model_a.parameters()).device)
    opt = torch.optim.AdamW(list(model_b.parameters()) + list(projector.parameters()), lr=lr, weight_decay=0.0)
    logs = []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        model_a.zero_grad(set_to_none=True)
        batch = batch_records(train, batch_size, step - 1)
        with torch.no_grad():
            _, _, _, h, _ = encoder_forward(model_a, tokenizer, batch, 3)
        loss, _, _ = decoder_forward(model_b, tokenizer, batch, h[:, stage - 1, :].detach(), projector, stage, "qz")
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite interpreter loss")
        loss.backward()
        ag = _grad_norm(model_a.parameters())[0]
        if ag != 0:
            raise RuntimeError("A received gradients in staged interpreter")
        torch.nn.utils.clip_grad_norm_(list(model_b.parameters()) + list(projector.parameters()), 1.0)
        opt.step()
        if step in {1, steps} or step % max(1, steps // 4) == 0:
            logs.append({"step": step, "loss": float(loss.item()), "grad_A": ag, "grad_B": _grad_norm(model_b.parameters())[0], "grad_projector": _grad_norm(projector.parameters())[0]})
    with torch.no_grad():
        z_chunks = []
        for start in range(0, len(val), batch_size):
            batch = val[start : start + batch_size]
            _, _, _, h, _ = encoder_forward(model_a, tokenizer, batch, 3)
            z_chunks.append(h[:, stage - 1, :].detach())
        z_val = torch.cat(z_chunks, dim=0)
    metrics = evaluate_interventions(model_b, tokenizer, projector, val, z_val, stage, batch_size)
    out_dir = seed_dir / f"interpreter_stage_{stage}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_b": model_b.state_dict(), "projector": projector.state_dict(), "stage": stage, "logs": logs}, out_dir / "checkpoint.pt")
    return {"logs": logs, "validation": metrics}


def latent_geometry_report(z: torch.Tensor, rows: list[dict]) -> dict:
    x = z.float().cpu()
    raw = hidden_geometry(x.unsqueeze(1))
    xc = x - x.mean(0, keepdim=True)
    centered = hidden_geometry(xc.unsqueeze(1))
    cov = (xc.T @ xc) / max(1, x.size(0) - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0)))
    xw = xc @ eigvecs @ torch.diag(torch.rsqrt(eigvals.clamp_min(1e-5)))
    whitened = hidden_geometry(xw.unsqueeze(1))
    by_group = defaultdict(list)
    for i, row in enumerate(rows):
        by_group[row["pair_group_id"]].append(i)
    within, between = [], []
    for i in range(x.size(0)):
        for j in range(i + 1, x.size(0)):
            d = float(torch.norm(x[i] - x[j]).item())
            if rows[i]["pair_group_id"] == rows[j]["pair_group_id"]:
                within.append(d)
            else:
                between.append(d)
    fact_rows = [[float(v) for v in row["intermediate_results"]] for row in rows]
    max_facts = max(len(v) for v in fact_rows)
    facts = torch.tensor([v + [0.0] * (max_facts - len(v)) for v in fact_rows], dtype=torch.float)
    probe_r2 = {}
    if len(rows) > x.size(1) // 8:
        pass
    for col in range(facts.size(1)):
        y = facts[:, col : col + 1]
        w = torch.linalg.lstsq(torch.cat([x, torch.ones(x.size(0), 1)], dim=1), y).solution
        pred = torch.cat([x, torch.ones(x.size(0), 1)], dim=1) @ w
        ss_res = ((y - pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum().clamp_min(1e-9)
        probe_r2[f"value_{col}"] = float((1 - ss_res / ss_tot).item())
    return {
        "raw": raw,
        "mean_centered": centered,
        "whitened": whitened,
        "within_pair_distance": float(statistics.mean(within)) if within else None,
        "between_pair_distance": float(statistics.mean(between)) if between else None,
        "context_value_linear_probe_train_r2": probe_r2,
    }


def run_seed(seed: int, args) -> dict:
    set_seed(seed)
    paths = make_data(seed, args.train_groups, args.val_groups, args.ood_groups)
    train, val, ood = read_jsonl(Path(paths["train"])), read_jsonl(Path(paths["validation"])), read_jsonl(Path(paths["ood"]))
    tokenizer, models = load_models(with_b=0)
    model_a = models[0]
    seed_dir = CKPT / f"seed_{seed}"
    explicit_logs = train_a(model_a, tokenizer, train, "explicit", args.steps, args.batch_size, args.lr, seed_dir / "p0_explicit")
    explicit_eval = {"train": eval_a(model_a, tokenizer, train[: min(len(train), 128)], "explicit", args.batch_size), "validation": eval_a(model_a, tokenizer, val, "explicit", args.batch_size)}
    progressive = {}
    for mode, name in [(1, "p1"), (2, "p2"), (3, "p3")]:
        logs = train_a(model_a, tokenizer, train, mode, args.steps, args.batch_size, args.lr, seed_dir / name)
        progressive[name] = {"logs": logs, "validation": eval_a(model_a, tokenizer, val, mode, args.batch_size)}
    recovering_logs = train_a(model_a, tokenizer, train, 3, args.steps, args.batch_size, args.lr, seed_dir / "recover_encoder")
    recovering_eval = {"logs": recovering_logs, "validation": eval_a(model_a, tokenizer, val, 3, args.batch_size), "ood": eval_a(model_a, tokenizer, ood, 3, args.batch_size)}
    with torch.no_grad():
        z_chunks = []
        for start in range(0, len(val), args.batch_size):
            batch = val[start : start + args.batch_size]
            _, _, _, h, _ = encoder_forward(model_a, tokenizer, batch, 3)
            z_chunks.append(h.detach().cpu())
        z_all = torch.cat(z_chunks, dim=0)
    geometry = {f"stage_{i+1}": latent_geometry_report(z_all[:, i, :], val) for i in range(z_all.size(1))}
    staged = {f"stage_{stage}": train_stage_interpreter(model_a, tokenizer, train, val, stage, args.interpreter_steps, args.batch_size, args.lr, seed_dir) for stage in [1, 2, 3]}
    return {
        "seed": seed,
        "paths": paths,
        "pair_validation": validate_pairs({"train": train, "validation": val, "ood": ood}),
        "explicit": {"logs": explicit_logs, **explicit_eval},
        "progressive": progressive,
        "recovering": recovering_eval,
        "staged": staged,
        "geometry": geometry,
    }


def semantic_gate(seed_results: dict) -> dict:
    checks = defaultdict(list)
    for seed, res in seed_results.items():
        checks["answer_em_nonfloor"].append(res["recovering"]["validation"]["answer_em"] > 0)
        stage_ok = []
        for st, sr in res["staged"].items():
            v = sr["validation"]
            full_margin = v.get("paired_shuffle_full", 0.0) - v.get("normal_full", 0.0)
            numeric_margin = v.get("paired_shuffle_numeric", 0.0) - v.get("normal_numeric", 0.0)
            inter_margin = v.get("paired_shuffle_intermediate", 0.0) - v.get("normal_intermediate", 0.0)
            fact_margin = v.get("paired_shuffle_context_fact", 0.0) - v.get("normal_context_fact", 0.0)
            q_gain = v.get("q_only_full", 0.0) - v.get("normal_full", 0.0)
            stage_ok.append(full_margin > 0 and numeric_margin > 0 and inter_margin > 0 and fact_margin > 0 and q_gain > 0)
        checks["stage_semantics"].append(sum(stage_ok) >= 2)
    out = {k: {"passed_count": sum(v), "n": len(v), "passed": sum(v) >= max(1, math.ceil(2 * len(v) / 3))} for k, v in checks.items()}
    out["allow_ours_l1_rerun"] = all(v["passed"] for v in out.values())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--interpreter-steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--train-groups", type=int, default=16)
    parser.add_argument("--val-groups", type=int, default=4)
    parser.add_argument("--ood-groups", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "compatibility_mode": "strict_heima_repo",
        "thinking_state_mode": "predictor",
        "projector_type": "heima_official",
        "loss_backend": "torchtune_chunked_ce",
        "allow_loss_fallback": False,
        "model_name_or_path": MODEL,
        "tokens": TOKENS,
        "args": vars(args),
    }
    write_json(OUT / "dataset_spec.json", {
        "schema": ["context", "question", "cot1", "cot2", "cot3", "answer", "pair_group_id", "operation_type", "intermediate_results"],
        "a_visible": ["context", "question"],
        "b_visible": ["question", "latent"],
        "paired_group_size": 4,
        "context_types": ["arithmetic context", "ordered operation context", "table/fact context", "multi-step state update context"],
    })
    results = {}
    for seed in args.seeds:
        results[str(seed)] = run_seed(seed, args)
        write_json(OUT / "partial_seed_results.json", results)
    pair_validation = {s: r["pair_validation"] for s, r in results.items()}
    explicit = {s: r["explicit"] for s, r in results.items()}
    progressive = {s: r["progressive"] for s, r in results.items()}
    recovering = {s: r["recovering"] for s, r in results.items()}
    staged = {s: r["staged"] for s, r in results.items()}
    geometry = {s: r["geometry"] for s, r in results.items()}
    gate = semantic_gate(results)
    interventions = {
        s: {
            st: {
                "paired_shuffle_minus_normal_full": sr["validation"].get("paired_shuffle_full", 0.0) - sr["validation"].get("normal_full", 0.0),
                "paired_shuffle_minus_normal_numeric": sr["validation"].get("paired_shuffle_numeric", 0.0) - sr["validation"].get("normal_numeric", 0.0),
                "paired_shuffle_minus_normal_intermediate": sr["validation"].get("paired_shuffle_intermediate", 0.0) - sr["validation"].get("normal_intermediate", 0.0),
                "paired_shuffle_minus_normal_context_fact": sr["validation"].get("paired_shuffle_context_fact", 0.0) - sr["validation"].get("normal_context_fact", 0.0),
                "q_only_minus_normal_full": sr["validation"].get("q_only_full", 0.0) - sr["validation"].get("normal_full", 0.0),
            }
            for st, sr in r["staged"].items()
        }
        for s, r in results.items()
    }
    write_json(OUT / "experiment_manifest.json", {**manifest, "backend_resolution": backend_resolution_snapshot()})
    write_json(OUT / "pair_validation.json", pair_validation)
    write_json(OUT / "explicit_results.json", explicit)
    write_json(OUT / "progressive_results.json", progressive)
    write_json(OUT / "recovering_results.json", recovering)
    write_json(OUT / "staged_interpreter_results.json", staged)
    write_json(OUT / "paired_interventions.json", interventions)
    write_json(OUT / "latent_geometry.json", geometry)
    write_json(OUT / "semantic_gate.json", gate)
    with (OUT / "sample_decodes.txt").open("w", encoding="utf-8") as f:
        for seed, res in results.items():
            f.write(f"seed {seed}\n")
            for st, sr in res["staged"].items():
                f.write(f"{st} validation metrics: {json.dumps(sr['validation'], sort_keys=True)}\n")
    report = [
        "# Context-Conditioned Heima Baseline",
        "",
        "Strict core stayed locked: predictor hidden, official projector, official embedding replacement, torchtune chunked CE.",
        "",
        f"Semantic gate allow Ours-L1 rerun: {gate['allow_ours_l1_rerun']}",
        "",
        "This run restores Heima-style information asymmetry: Model A sees context+question; Model B sees question+latent only.",
    ]
    write_json(OUT / "all_seed_results.json", results)
    (OUT / "context_conditioned_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "out": str(OUT), "semantic_gate": gate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
