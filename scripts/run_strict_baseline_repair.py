#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.htext.formal_eval import cot_nll_breakdown, hidden_geometry, logits_kl, make_latent_variants
from src.htext.heima_reuse import HeimaOfficialAbstractProjection, backend_resolution_snapshot, heima_ce_loss, official_embedding_replacement
from src.htext.modeling import THINKING_TOKEN, ANSWER_PREFIX, _decoder_prompt_ids, h0_forward, h1_forward, setup_special_tokens, tokenize_text
from src.htext.synthetic_data import generate_synthetic_split, read_jsonl, write_jsonl
from src.htext.trainer import _answer_em, _grad_norm, _load_tokenizer_and_model, batch_records, set_seed


OUT = Path("reports/strict_baseline_repair")
CKPT = Path("checkpoints/strict_baseline_repair")
DATA = Path("experiments/htext_gpt2/data/strict_baseline_repair")
NUM_RE = re.compile(r"-?\d+")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def base_config(seed: int, batch_size: int) -> dict:
    return {
        "model_name_or_path": "/mnt/nas/share2/home/zxl/models/openai-community-gpt2",
        "local_files_only": True,
        "use_safetensors": True,
        "seed": seed,
        "num_thinking_tokens": 1,
        "max_question_tokens": 80,
        "max_answer_tokens": 16,
        "max_cot_tokens": 128,
        "micro_batch_size": batch_size,
        "max_grad_norm": 1.0,
    }


def load_s0(config: dict, ckpt_path: Path):
    tokenizer, model_a, _, _ = _load_tokenizer_and_model(config, with_b=False)
    ckpt = torch.load(ckpt_path, map_location=next(model_a.parameters()).device)
    model_a.load_state_dict(ckpt["model_a"], strict=True)
    model_a.eval()
    return tokenizer, model_a


def load_s1(config: dict, ckpt_path: Path):
    tokenizer, model_a, model_b, _ = _load_tokenizer_and_model(config, with_b=True)
    ckpt = torch.load(ckpt_path, map_location=next(model_a.parameters()).device)
    model_a.load_state_dict(ckpt["model_a"], strict=True)
    model_b.load_state_dict(ckpt["model_b"], strict=True)
    projector = HeimaOfficialAbstractProjection(model_a.config.n_embd, model_b.config.n_embd).to(next(model_a.parameters()).device)
    projector.load_state_dict(ckpt["projector"], strict=True)
    model_a.eval(); model_b.eval(); projector.eval()
    return tokenizer, model_a, model_b, projector


def eval_s0(config: dict, tokenizer, model_a, records: list[dict]) -> dict:
    chunks, zs = [], []
    with torch.no_grad():
        for start in range(0, len(records), config["micro_batch_size"]):
            batch = records[start : start + config["micro_batch_size"]]
            out = h0_forward(model_a, tokenizer, batch, config["max_question_tokens"], config["max_answer_tokens"], 1, "predictor")
            chunks.append(out)
            zs.append(out.thinking_hidden.detach().cpu())
    n = sum(x.labels.size(0) for x in chunks)
    z = torch.cat(zs, dim=0)
    return {
        "answer_nll": sum(float(x.loss.item()) * x.labels.size(0) for x in chunks) / max(n, 1),
        "answer_em": sum(_answer_em(x.logits, x.labels) * x.labels.size(0) for x in chunks) / max(n, 1),
        "thinking_token_accuracy": thinking_token_acc(tokenizer, chunks),
        "latent_geometry": hidden_geometry(z),
        "retrieval": latent_retrieval(z),
    }


def thinking_token_acc(tokenizer, chunks) -> float:
    tid = tokenizer.convert_tokens_to_ids(THINKING_TOKEN)
    correct = total = 0
    for out in chunks:
        pred = out.logits[:, :-1].argmax(-1)
        labels = out.labels[:, 1:]
        mask = labels == tid
        correct += int((pred[mask] == tid).sum().item())
        total += int(mask.sum().item())
    return correct / max(total, 1)


def latent_retrieval(z: torch.Tensor) -> dict:
    flat = F.normalize(z.reshape(z.size(0), -1).float(), dim=-1)
    sim = flat @ flat.T
    ranks = []
    for i in range(sim.size(0)):
        order = torch.argsort(sim[i], descending=True)
        ranks.append(int((order == i).nonzero()[0].item()) + 1)
    return {"R@1": sum(r <= 1 for r in ranks) / len(ranks), "R@5": sum(r <= 5 for r in ranks) / len(ranks), "random_R@1": 1 / len(ranks), "random_R@5": min(5 / len(ranks), 1)}


def none_sub(a, b):
    return None if a is None or b is None else a - b


def generate_answers(config: dict, tokenizer, model_a, records: list[dict], limit: int = 30) -> list[dict]:
    device = next(model_a.parameters()).device
    tid = tokenizer.convert_tokens_to_ids(THINKING_TOKEN)
    rows = []
    for record in records[:limit]:
        q = tokenize_text(tokenizer, record["question"], config["max_question_tokens"])
        prefix = tokenize_text(tokenizer, ANSWER_PREFIX)
        input_ids = torch.tensor([q + [tid] + prefix], dtype=torch.long, device=device)
        generated = model_a.generate(
            input_ids=input_ids,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_ids = generated[0, input_ids.size(1) :].tolist()
        raw = tokenizer.decode(new_ids, skip_special_tokens=True)
        nums = NUM_RE.findall(raw)
        parsed = nums[-1] if nums else None
        stop = "eos" if tokenizer.eos_token_id in new_ids else "max_new_tokens"
        rows.append({"id": record["id"], "question": record["question"], "raw_generated_text": raw, "parsed_answer": parsed, "gold_answer": record["answer"], "token_ids": new_ids, "stop_reason": stop, "match": parsed == record["answer"]})
    return rows


def current_failure_diagnosis(seeds: list[int]) -> tuple[dict, dict]:
    strict = Path("reports/strict_difference")
    seed_results = json.loads((strict / "seed_results.json").read_text())
    grad = json.loads((strict / "gradient_attribution.json").read_text())
    diagnosis = {}
    audit_rows = []
    for seed in seeds:
        s = str(seed)
        row = seed_results[s]
        config = base_config(seed, 4)
        train = read_jsonl(row["config"]["train_path"])
        val = read_jsonl(row["config"]["validation_path"])
        tokenizer, model_a = load_s0(config, Path(row["s0"]["checkpoint"]))
        train_eval = eval_s0(config, tokenizer, model_a, train[:64])
        val_eval = eval_s0(config, tokenizer, model_a, val[:64])
        s1 = row["s1"]["interventions"]
        diagnosis[s] = {
            "s0_train": train_eval,
            "s0_validation": val_eval,
            "s1_q_only_nll": row["s1"]["q_only"]["full_cot_nll"],
            "s1_q_normal_nll": s1["normal"]["nll"],
            "s1_q_shuffle_nll": s1["shuffled"]["nll"],
            "s1_delta_shuffle": s1["deltas"],
            "grad_A_main": grad[s]["joint_no_detach"]["grad_A_from_main"],
            "lambda1_grad_A_loss1": 0.1 * grad[s]["joint_no_detach"]["grad_A_from_loss1"],
            "gradient_norm_ratio_lambda_loss1_over_main": 0.1 * grad[s]["joint_no_detach"]["grad_A_from_loss1"] / grad[s]["joint_no_detach"]["grad_A_from_main"],
        }
        if seed == seeds[0]:
            audit_rows = generate_answers(config, tokenizer, model_a, val, 30)
    answer_audit = {
        "rows": audit_rows,
        "match_rate": sum(r["match"] for r in audit_rows) / max(len(audit_rows), 1),
        "finding": "Answer EM=0 is generation/model failure if parsed answers do not match gold; parser extracts last integer from raw generated text.",
    }
    return diagnosis, answer_audit


def tiny_overfit(seed: int, steps: int, lr: float, batch_size: int) -> dict:
    set_seed(seed)
    train, _ = generate_synthetic_split(32, 8, seed)
    DATA.mkdir(parents=True, exist_ok=True)
    tiny_path = DATA / "tiny32_train.jsonl"
    write_jsonl(train, tiny_path)
    config = base_config(seed, batch_size)
    tokenizer, model_a, _, _ = _load_tokenizer_and_model(config, with_b=False)
    opt = torch.optim.AdamW(model_a.parameters(), lr=lr, weight_decay=0.0)
    logs = []
    start = time.time()
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        batch = batch_records(train, batch_size, step - 1)
        model_a.train()
        out = h0_forward(model_a, tokenizer, batch, config["max_question_tokens"], config["max_answer_tokens"], 1, "predictor")
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0)
        grad, finite = _grad_norm(model_a.parameters())
        if not finite:
            raise RuntimeError("tiny overfit non-finite gradient")
        opt.step()
        if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
            full = eval_s0(config, tokenizer, model_a, train)
            logs.append({"step": step, "batch_loss": float(out.loss.item()), "train_answer_nll": full["answer_nll"], "train_answer_em": full["answer_em"], "grad_A_main": grad})
    gen = generate_answers(config, tokenizer, model_a, train, 32)
    em_gen = sum(r["match"] for r in gen) / len(gen)
    ckpt = CKPT / "tiny_overfit_s0.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_a": model_a.state_dict(), "optimizer": opt.state_dict()}, ckpt)
    final = eval_s0(config, tokenizer, model_a, train)
    passed = final["answer_nll"] < 1.0 or em_gen > 0.5
    return {"status": "pass" if passed else "fail", "checkpoint": str(ckpt), "steps": steps, "learning_rate": lr, "logs": logs, "final_train_eval": final, "generation_em": em_gen, "generations": gen, "runtime_sec": time.time() - start, "failure_hypothesis": None if passed else "S0 did not visibly overfit 32 examples; current Answer EM floor is likely training/objective/generation rather than parser only."}


def split_cot(record: dict) -> list[str]:
    steps = list(record.get("cot_steps_text") or record.get("cot_steps_raw") or [record["cot"]])
    if len(steps) >= 3:
        return [steps[0], steps[1], " ".join(steps[2:])]
    if len(steps) == 2:
        return [steps[0], steps[1], f"The answer is {record['answer']}."]
    return [record["question"], record["cot"], f"The answer is {record['answer']}."]


def progressive_rows(records: list[dict], stage: int) -> list[dict]:
    out = []
    for r in records:
        parts = split_cot(r)
        item = dict(r)
        item["cot_parts"] = {"cot1": parts[0], "cot2": parts[1], "cot3": parts[2]}
        item["progressive_stage"] = stage
        out.append(item)
    return out


def progressive_forward(model, tokenizer, records: list[dict], stage: int, max_question: int, max_answer: int):
    device = next(model.parameters()).device
    tid = tokenizer.convert_tokens_to_ids(THINKING_TOKEN)
    prefix_ids = tokenize_text(tokenizer, ANSWER_PREFIX)
    rows, labels = [], []
    for r in records:
        q = tokenize_text(tokenizer, r["question"], max_question)
        parts = split_cot(r)
        seq = list(q)
        lab = [-100] * len(seq)
        for idx, part in enumerate(parts):
            if idx < stage:
                seq.append(tid)
                lab.append(tid)
            else:
                ids = tokenize_text(tokenizer, part + " ")
                seq.extend(ids); lab.extend(ids)
        ans = tokenize_text(tokenizer, ANSWER_PREFIX + r["answer"] + tokenizer.eos_token, max_answer + len(prefix_ids))
        seq.extend(ans); lab.extend(ans)
        rows.append(torch.tensor(seq, dtype=torch.long))
        labels.append(torch.tensor(lab, dtype=torch.long))
    max_len = max(x.numel() for x in rows)
    input_ids = torch.full((len(rows), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    label_t = torch.full((len(rows), max_len), -100, dtype=torch.long, device=device)
    attn = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        input_ids[i, : row.numel()] = row.to(device)
        label_t[i, : labels[i].numel()] = labels[i].to(device)
        attn[i, : row.numel()] = 1
    out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
    loss = heima_ce_loss(out.logits, label_t)
    shifted_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    shifted_mask[:, :-1] = input_ids[:, 1:].eq(tid)
    hidden = out.hidden_states[-1][shifted_mask]
    if stage:
        hidden = hidden.view(len(rows), stage, -1)
    else:
        hidden = torch.empty((len(rows), 0, out.hidden_states[-1].size(-1)), device=device)
    return loss, out.logits, label_t, hidden


def train_progressive(seed: int, steps_per_stage: int, lr: float, batch_size: int) -> tuple[dict, object, object, list[dict]]:
    set_seed(seed)
    train, val = generate_synthetic_split(128, 64, seed + 500)
    config = base_config(seed, batch_size)
    tokenizer, model_a, _, _ = _load_tokenizer_and_model(config, with_b=False)
    opt = torch.optim.AdamW(model_a.parameters(), lr=lr, weight_decay=0.0)
    stages = {}
    for stage in range(4):
        logs = []
        for step in range(1, steps_per_stage + 1):
            opt.zero_grad(set_to_none=True)
            batch = batch_records(train, batch_size, step - 1)
            loss, _, _, _ = progressive_forward(model_a, tokenizer, batch, stage, config["max_question_tokens"], config["max_answer_tokens"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0)
            opt.step()
            if step == 1 or step == steps_per_stage:
                logs.append({"step": step, "loss": float(loss.item())})
        geom = progressive_geometry(model_a, tokenizer, val[:48], stage, config)
        stages[str(stage)] = {"logs": logs, "geometry": geom}
        torch.save({"model_a": model_a.state_dict(), "optimizer": opt.state_dict(), "stage": stage}, CKPT / f"progressive_stage_{stage}.pt")
    return {"stages": stages}, tokenizer, model_a, val


def progressive_geometry(model_a, tokenizer, records, stage, config):
    if stage == 0:
        return {}
    chunks = []
    with torch.no_grad():
        for start in range(0, len(records), config["micro_batch_size"]):
            _, _, _, h = progressive_forward(model_a, tokenizer, records[start:start+config["micro_batch_size"]], stage, config["max_question_tokens"], config["max_answer_tokens"])
            chunks.append(h.cpu())
    h = torch.cat(chunks, dim=0)
    return {f"z{i+1}": {"geometry": hidden_geometry(h[:, i:i+1, :]), "retrieval": latent_retrieval(h[:, i:i+1, :])} for i in range(stage)}


def local_s1(seed: int, tokenizer, model_a, records: list[dict], steps: int, batch_size: int, lr: float) -> dict:
    set_seed(seed + 900)
    config = base_config(seed, batch_size)
    _, _, model_b, _ = _load_tokenizer_and_model(config, with_b=True)
    projector = HeimaOfficialAbstractProjection(model_a.config.n_embd, model_b.config.n_embd).to(next(model_b.parameters()).device)
    for p in model_a.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(list(model_b.parameters()) + list(projector.parameters()), lr=lr, weight_decay=0.0)
    train = records[:96]
    val = records[96:128] if len(records) >= 128 else records[:32]
    logs = []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        batch = batch_records(train, batch_size, step - 1)
        loss_total = 0
        for stage_idx in range(1, 4):
            with torch.no_grad():
                _, _, _, h = progressive_forward(model_a, tokenizer, batch, 3, config["max_question_tokens"], config["max_answer_tokens"])
            z = h[:, stage_idx - 1, :]
            target_records = []
            for r in batch:
                parts = split_cot(r)
                rr = dict(r); rr["cot"] = parts[stage_idx - 1]; rr["question"] = f"{r['question']}\nStage instruction: reconstruct cot{stage_idx}."
                target_records.append(rr)
            pred = h1_forward(model_b, tokenizer, target_records, z.unsqueeze(1).detach(), projector, config["max_cot_tokens"], mode="qz")
            loss_total = loss_total + pred.loss
        loss_total.backward()
        opt.step()
        if step == 1 or step == steps:
            logs.append({"step": step, "loss": float(loss_total.item())})
    evals = {}
    with torch.no_grad():
        _, _, _, h = progressive_forward(model_a, tokenizer, val, 3, config["max_question_tokens"], config["max_answer_tokens"])
        for stage_idx in range(1, 4):
            z_stage = h[:, stage_idx - 1, :]
            variants = make_latent_variants(z_stage)
            target_records = []
            for r in val:
                rr = dict(r); rr["cot"] = split_cot(r)[stage_idx - 1]; rr["question"] = f"{r['question']}\nStage instruction: reconstruct cot{stage_idx}."
                target_records.append(rr)
            stage_eval = {}
            for name, z in {"normal": z_stage, "shuffle": variants["shuffled"], "zero": torch.zeros_like(z_stage)}.items():
                pred = h1_forward(model_b, tokenizer, target_records, z_stage.unsqueeze(1), projector, config["max_cot_tokens"], latent_override=z, mode="qz")
                stage_eval[name] = cot_nll_breakdown(tokenizer, target_records, pred.logits, pred.labels)
            dummy = torch.zeros_like(z_stage).unsqueeze(1)
            q_pred = h1_forward(model_b, tokenizer, target_records, dummy, projector, config["max_cot_tokens"], mode="q")
            z_pred = h1_forward(model_b, tokenizer, target_records, z_stage.unsqueeze(1), projector, config["max_cot_tokens"], mode="z")
            stage_eval["q_only"] = cot_nll_breakdown(tokenizer, target_records, q_pred.logits, q_pred.labels)
            stage_eval["z_only"] = cot_nll_breakdown(tokenizer, target_records, z_pred.logits, z_pred.labels)
            stage_eval["deltas"] = {
                "shuffle_minus_normal_full": stage_eval["shuffle"]["full"] - stage_eval["normal"]["full"],
                "q_only_minus_normal_full": stage_eval["q_only"]["full"] - stage_eval["normal"]["full"],
                "numeric_shuffle_minus_normal": none_sub(stage_eval["shuffle"].get("numeric_tokens"), stage_eval["normal"].get("numeric_tokens")),
                "intermediate_shuffle_minus_normal": none_sub(stage_eval["shuffle"].get("intermediate_tokens"), stage_eval["normal"].get("intermediate_tokens")),
            }
            stage_eval["generation"] = local_generation_metrics(config, tokenizer, model_b, projector, target_records[:12], z_stage[:12], variants["shuffled"][:12])
            evals[f"cot{stage_idx}"] = stage_eval
    return {"logs": logs, "stage_eval": evals}


def local_generation_metrics(config, tokenizer, model_b, projector, records, z_normal, z_shuffle):
    device = next(model_b.parameters()).device
    out = {name: {"number_match": 0, "intermediate_result_match": 0, "expression_match": 0, "texts": []} for name in ["normal", "shuffle"]}
    for i, record in enumerate(records):
        prompt_ids, latent_pos = _decoder_prompt_ids(tokenizer, record["question"], "qz")
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        embeds = model_b.get_input_embeddings()(input_ids)
        for name, z in [("normal", z_normal), ("shuffle", z_shuffle)]:
            projected = projector(z[i : i + 1])
            inputs_embeds = torch.cat([embeds[:, :latent_pos, :], projected.unsqueeze(1), embeds[:, latent_pos + 1 :, :]], dim=1)
            generated = model_b.generate(inputs_embeds=inputs_embeds, attention_mask=torch.ones_like(input_ids), max_new_tokens=48, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=False)
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            nums = set(NUM_RE.findall(text))
            gold_nums = set(NUM_RE.findall(record["cot"]))
            out[name]["number_match"] += int(bool(gold_nums & nums))
            out[name]["intermediate_result_match"] += int(bool(gold_nums & nums))
            out[name]["expression_match"] += int(any(op in text for op in ["+", "-", "*", "/", "="]))
            out[name]["texts"].append({"id": record["id"], "text": text, "gold": record["cot"]})
    for name in ["normal", "shuffle"]:
        for key in ["number_match", "intermediate_result_match", "expression_match"]:
            out[name][key] /= max(len(records), 1)
    return out


def semantic_gate(tiny, prog, interp) -> dict:
    stage_eval = interp.get("stage_eval", {})
    normal_better = [
        v["deltas"]["shuffle_minus_normal_full"] > 0
        for v in stage_eval.values()
    ]
    numeric_margins = [
        v["deltas"]["numeric_shuffle_minus_normal"]
        for v in stage_eval.values()
        if v["deltas"]["numeric_shuffle_minus_normal"] is not None
    ]
    inter_margins = [
        v["deltas"]["intermediate_shuffle_minus_normal"]
        for v in stage_eval.values()
        if v["deltas"]["intermediate_shuffle_minus_normal"] is not None
    ]
    q_margins = [v["deltas"]["q_only_minus_normal_full"] for v in stage_eval.values()]
    gen_normal = sum(v["generation"]["normal"]["intermediate_result_match"] for v in stage_eval.values()) / max(len(stage_eval), 1)
    gen_shuffle = sum(v["generation"]["shuffle"]["intermediate_result_match"] for v in stage_eval.values()) / max(len(stage_eval), 1)
    gates = {
        "s0_not_floor": tiny["status"] == "pass" and tiny["generation_em"] > 0,
        "normal_better_than_shuffle": sum(normal_better) >= 2,
        "numeric_intermediate_margin": bool(numeric_margins and sum(x > 0 for x in numeric_margins) >= 2 and inter_margins and sum(x > 0 for x in inter_margins) >= 2),
        "q_normal_better_than_q_only": bool(q_margins and sum(x > 0 for x in q_margins) >= 2),
        "free_generation_normal_intermediate_better": gen_normal > gen_shuffle,
        "retrieval_above_random": any(v["retrieval"]["R@1"] > v["retrieval"]["random_R@1"] for stage in prog.get("stages", {}).values() for v in stage.get("geometry", {}).values()),
    }
    allow = all(v is True for v in gates.values())
    return {"allow_difference_rerun": allow, "gates": gates}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tiny-steps", type=int, default=800)
    p.add_argument("--progressive-steps", type=int, default=160)
    p.add_argument("--local-s1-steps", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
    logits = torch.randn(1, 3, 5, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2]])
    heima_ce_loss(logits, labels).backward()
    if backend_resolution_snapshot()["ce_loss"]["fallback_used"]:
        raise RuntimeError("loss fallback used")
    diagnosis, answer_audit = current_failure_diagnosis([42, 43, 44])
    write_json(OUT / "current_failure_diagnosis.json", diagnosis)
    write_json(OUT / "answer_evaluator_audit.json", answer_audit)
    tiny = tiny_overfit(args.seed, args.tiny_steps, args.lr, args.batch_size)
    write_json(OUT / "tiny_overfit_report.json", tiny)
    if tiny["status"] != "pass":
        gate = {"allow_difference_rerun": False, "stop_after": "tiny_overfit", "reason": tiny["failure_hypothesis"]}
        write_json(OUT / "semantic_gate.json", gate)
        write_json(OUT / "progressive_stage_results.json", {"status": "not_run_due_to_tiny_gate"})
        write_json(OUT / "local_interpreter_results.json", {"status": "not_run_due_to_tiny_gate"})
        write_json(OUT / "latent_geometry.json", {"status": "not_run_due_to_tiny_gate"})
        (OUT / "strict_baseline_repair_report.md").write_text(f"# STRICT-HEIMA-BASELINE-REPAIR\n\nStopped after tiny overfit gate. {tiny['failure_hypothesis']}\n", encoding="utf-8")
        print(json.dumps({"status": "stopped_after_tiny_overfit", "out": str(OUT)}, indent=2))
        return 2
    prog, tokenizer, model_a, prog_records = train_progressive(args.seed, args.progressive_steps, args.lr, args.batch_size)
    write_json(OUT / "progressive_stage_results.json", prog)
    write_json(OUT / "latent_geometry.json", prog["stages"])
    interp = local_s1(args.seed, tokenizer, model_a, prog_records, args.local_s1_steps, args.batch_size, args.lr)
    write_json(OUT / "local_interpreter_results.json", interp)
    gate = semantic_gate(tiny, prog, interp)
    write_json(OUT / "semantic_gate.json", gate)
    (OUT / "strict_baseline_repair_report.md").write_text(
        "# STRICT-HEIMA-BASELINE-REPAIR\n\n"
        f"Tiny overfit: {tiny['status']}, generation EM={tiny['generation_em']}.\n\n"
        f"Semantic gate allow_difference_rerun={gate['allow_difference_rerun']}.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "out": str(OUT), "allow_difference_rerun": gate["allow_difference_rerun"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
