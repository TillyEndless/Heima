#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.htext.formal_eval import hidden_geometry
from src.htext.heima_reuse import (
    HeimaOfficialAbstractProjection,
    extract_thinking_state,
    heima_ce_loss,
    official_embedding_replacement,
    prepare_latent_for_decoder,
)

THINKING_TOKENS = {
    "summary": "<THINKING_OF_SUMMARY>",
    "caption": "<THINKING_OF_CAPTION>",
    "reasoning": "<THINKING_OF_REASONING>",
}
SECTION_TAGS = {
    "summary": ("<SUMMARY>", "</SUMMARY>"),
    "caption": ("<CAPTION>", "</CAPTION>"),
    "reasoning": ("<REASONING>", "</REASONING>"),
}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_records(records: list[dict], batch_size: int, step: int) -> list[dict]:
    start = (step * batch_size) % len(records)
    if start + batch_size <= len(records):
        return records[start : start + batch_size]
    return records[start:] + records[: (start + batch_size) % len(records)]


def tok(tokenizer, text: str, max_len: int | None = None) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids[:max_len] if max_len else ids


def load_models(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_safetensors=True)
    tokenizer.pad_token = tokenizer.eos_token
    specials = []
    for section, token in THINKING_TOKENS.items():
        specials.extend([SECTION_TAGS[section][0], token, SECTION_TAGS[section][1]])
    tokenizer.add_special_tokens({"additional_special_tokens": specials})
    model_a = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, use_safetensors=True)
    decoders = {
        section: AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, use_safetensors=True)
        for section in THINKING_TOKENS
    }
    model_a.resize_token_embeddings(len(tokenizer))
    model_a.config.use_cache = False
    model_a.to(device)
    for model in decoders.values():
        model.resize_token_embeddings(len(tokenizer))
        model.config.use_cache = False
        model.to(device)
    projectors = {
        section: HeimaOfficialAbstractProjection(model_a.config.n_embd, decoders[section].config.n_embd).to(device)
        for section in THINKING_TOKENS
    }
    return tokenizer, model_a, decoders, projectors


def encoder_sequence(tokenizer, record: dict, max_q: int, max_answer: int):
    question = "Question:\n" + record["question"] + "\n"
    q_ids = tok(tokenizer, question, max_q)
    ids = list(q_ids)
    labels = [-100] * len(ids)
    for section in ("summary", "caption", "reasoning"):
        open_tag, close_tag = SECTION_TAGS[section]
        part = f"\n{open_tag} {THINKING_TOKENS[section]} {close_tag}\n"
        pids = tok(tokenizer, part)
        ids += pids
        labels += [x if x == tokenizer.convert_tokens_to_ids(THINKING_TOKENS[section]) else -100 for x in pids]
    answer = f"\n<CONCLUSION> {record['answer']} </CONCLUSION>" + tokenizer.eos_token
    ans_ids = tok(tokenizer, answer, max_answer)
    ids += ans_ids
    labels += ans_ids
    return ids, labels


def pad_batch(tokenizer, rows: list[list[int]], labels: list[list[int]], device: str):
    max_len = max(len(x) for x in rows)
    input_ids = torch.full((len(rows), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    label_t = torch.full_like(input_ids, -100)
    attn = torch.zeros_like(input_ids)
    for i, row in enumerate(rows):
        input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        label_t[i, : len(labels[i])] = torch.tensor(labels[i], dtype=torch.long, device=device)
        attn[i, : len(row)] = 1
    return input_ids, label_t, attn


def encoder_forward(model_a, tokenizer, records: list[dict], args):
    device = next(model_a.parameters()).device
    rows, labels = zip(*(encoder_sequence(tokenizer, r, args.max_q, args.max_answer) for r in records))
    input_ids, label_t, attn = pad_batch(tokenizer, list(rows), list(labels), str(device))
    out = model_a(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
    main_loss = heima_ce_loss(out.logits, label_t)
    z = {}
    pos_trace = {}
    for section, token in THINKING_TOKENS.items():
        tid = tokenizer.convert_tokens_to_ids(token)
        state = extract_thinking_state(
            input_ids=input_ids,
            last_hidden_state=out.hidden_states[-1],
            thinking_token_id=tid,
            mode="predictor",
        )
        z[section] = state.hidden
        pos_trace[section] = {
            "thinking_pos": state.thinking_positions.detach().cpu().tolist(),
            "selected_pos": state.selected_positions.detach().cpu().tolist(),
        }
    return main_loss, out.logits, label_t, z, pos_trace


def decoder_prompt(record: dict, section: str) -> str:
    question = record["question"]
    return (
        "Question:\n"
        + question
        + f"\n\nInstruction:\nReconstruct the {section} thought used by Heima for this question.\n\n"
        + f"{THINKING_TOKENS[section]}\n\n"
        + "Target:\n"
    )


def decoder_forward(model_b, projector, tokenizer, records: list[dict], section: str, z, args):
    device = next(model_b.parameters()).device
    projected = projector(z)
    rows, labels, slot_positions = [], [], []
    for record in records:
        trimmed = dict(record)
        q_ids = tok(tokenizer, record["question"], args.max_q)
        trimmed["question"] = tokenizer.decode(q_ids, skip_special_tokens=False)
        prompt_ids = tok(tokenizer, decoder_prompt(trimmed, section))
        target_ids = tok(tokenizer, record[section] + tokenizer.eos_token, args.max_target)
        rows.append(prompt_ids + target_ids)
        labels.append([-100] * len(prompt_ids) + target_ids)
        token_id = tokenizer.convert_tokens_to_ids(THINKING_TOKENS[section])
        locs = [i for i, value in enumerate(prompt_ids) if value == token_id]
        if len(locs) != 1:
            raise RuntimeError(f"Expected one {THINKING_TOKENS[section]} in decoder prompt, got {locs}")
        slot_positions.append(locs[0])
    input_ids, label_t, attn = pad_batch(tokenizer, rows, labels, str(device))
    embeds = model_b.get_input_embeddings()(input_ids)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for i, pos in enumerate(slot_positions):
        mask[i, pos] = True
    embeds = official_embedding_replacement(embeds, projected.unsqueeze(1), mask)
    out = model_b(inputs_embeds=embeds, attention_mask=attn, use_cache=False)
    return heima_ce_loss(out.logits, label_t), out.logits, label_t


def grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().norm().item() ** 2)
    return total ** 0.5


@torch.no_grad()
def evaluate(model_a, decoders, projectors, tokenizer, records: list[dict], args):
    model_a.eval()
    for m in decoders.values():
        m.eval()
    totals = {"main_nll": 0.0, "loss1_nll": {s: 0.0 for s in THINKING_TOKENS}, "count": 0}
    zs = {s: [] for s in THINKING_TOKENS}
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        main_loss, _logits, _labels, z, _trace = encoder_forward(model_a, tokenizer, batch, args)
        totals["main_nll"] += float(main_loss.item()) * len(batch)
        for section in THINKING_TOKENS:
            loss1, _l, _t = decoder_forward(decoders[section], projectors[section], tokenizer, batch, section, z[section], args)
            totals["loss1_nll"][section] += float(loss1.item()) * len(batch)
            zs[section].append(z[section].detach().cpu())
        totals["count"] += len(batch)
    out = {
        "main_nll": totals["main_nll"] / max(totals["count"], 1),
        "loss1_nll": {k: v / max(totals["count"], 1) for k, v in totals["loss1_nll"].items()},
        "latent_geometry": {},
    }
    for section, chunks in zs.items():
        if chunks:
            out["latent_geometry"][section] = hidden_geometry(torch.cat(chunks, dim=0).unsqueeze(1))
    return out


def run_seed(seed: int, args, train: list[dict], val: list[dict], test: list[dict]):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model_a, decoders, projectors = load_models(args.model_path, device)
    params = list(model_a.parameters())
    for section in THINKING_TOKENS:
        params += list(decoders[section].parameters()) + list(projectors[section].parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    initial = evaluate(model_a, decoders, projectors, tokenizer, val, args)
    logs = []
    pos_trace = None
    started = time.time()
    for step in range(1, args.steps + 1):
        batch = batch_records(train, args.batch_size, step - 1)
        opt.zero_grad(set_to_none=True)
        model_a.train()
        for m in decoders.values():
            m.train()
        main_loss, _logits, _labels, z_original, trace = encoder_forward(model_a, tokenizer, batch, args)
        if pos_trace is None:
            pos_trace = trace
        loss1_terms = {}
        loss1_sum = torch.zeros((), device=main_loss.device)
        for section in THINKING_TOKENS:
            z_for_decoder = prepare_latent_for_decoder(z_original[section], detach_encoder_latent=False)
            loss1, _l, _t = decoder_forward(
                decoders[section], projectors[section], tokenizer, batch, section, z_for_decoder, args
            )
            loss1_terms[section] = loss1
            loss1_sum = loss1_sum + loss1
        total = main_loss + args.lambda1 * loss1_sum
        total.backward()
        g_a = grad_norm(model_a.parameters())
        g_b = {s: grad_norm(decoders[s].parameters()) for s in THINKING_TOKENS}
        g_p = {s: grad_norm(projectors[s].parameters()) for s in THINKING_TOKENS}
        torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
        opt.step()
        if step == 1 or step == args.steps or step % args.log_every == 0:
            logs.append(
                {
                    "step": step,
                    "total": float(total.item()),
                    "main": float(main_loss.item()),
                    "loss1": {s: float(loss1_terms[s].item()) for s in THINKING_TOKENS},
                    "grad_A_total": g_a,
                    "grad_decoder": g_b,
                    "grad_projector": g_p,
                }
            )
    final_val = evaluate(model_a, decoders, projectors, tokenizer, val, args)
    final_test = evaluate(model_a, decoders, projectors, tokenizer, test, args)
    out_dir = args.out / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_a": model_a.state_dict(),
            "decoders": {s: decoders[s].state_dict() for s in THINKING_TOKENS},
            "projectors": {s: projectors[s].state_dict() for s in THINKING_TOKENS},
            "optimizer": opt.state_dict(),
            "seed": seed,
            "args": vars(args),
        },
        out_dir / "main_l1_checkpoint.pt",
    )
    result = {
        "seed": seed,
        "status": "complete",
        "runtime_sec": time.time() - started,
        "initial_validation": initial,
        "final_validation": final_val,
        "final_test": final_test,
        "logs": logs,
        "position_trace_first_batch": pos_trace,
    }
    write_json(out_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_available_images_v1"))
    parser.add_argument("--model-path", default="/data/zxl/models/openai-community-gpt2")
    parser.add_argument("--out", type=Path, default=Path("/data/zxl/Heima/reports/official_micro_main_l1"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lambda1", type=float, default=0.1)
    parser.add_argument("--max-q", type=int, default=160)
    parser.add_argument("--max-answer", type=int, default=48)
    parser.add_argument("--max-target", type=int, default=160)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    train = read_jsonl(args.subset / "train.jsonl")
    val = read_jsonl(args.subset / "validation.jsonl")
    test = read_jsonl(args.subset / "test.jsonl")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "experiment_type": "official_data_micro_main_plus_loss1",
        "important_caveat": "This run uses official LLaVA-CoT micro data and strict-core GPT2 adapters. It is not the official 11B MLLM Heima baseline because official 11B/8B weights are not present.",
        "subset": str(args.subset),
        "model_path": args.model_path,
        "seeds": args.seeds,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lambda1": args.lambda1,
        "loss": "L = Lmain + lambda1 * (L1_summary + L1_caption + L1_reasoning), detach_encoder_latent=False",
        "official_core_alignment": {
            "thinking_state_mode": "predictor",
            "projector": "HeimaOfficialAbstractProjection",
            "embedding_replacement": "official_embedding_replacement",
            "loss": "CEWithChunkedOutputLoss via heima_ce_loss where available",
            "typed_tokens": THINKING_TOKENS,
        },
    }
    write_json(args.out / "experiment_manifest.json", manifest)
    results = []
    for seed in args.seeds:
        result = run_seed(seed, args, train, val, test)
        results.append(result)
        write_json(args.out / "seed_results.json", results)
    summary = {
        "status": "complete",
        "num_seeds": len(results),
        "validation_main_nll": {str(r["seed"]): r["final_validation"]["main_nll"] for r in results},
        "test_main_nll": {str(r["seed"]): r["final_test"]["main_nll"] for r in results},
        "validation_loss1_nll": {str(r["seed"]): r["final_validation"]["loss1_nll"] for r in results},
        "test_loss1_nll": {str(r["seed"]): r["final_test"]["loss1_nll"] for r in results},
    }
    write_json(args.out / "cross_seed_summary.json", summary)
    manifest["status"] = "complete"
    write_json(args.out / "experiment_manifest.json", manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
