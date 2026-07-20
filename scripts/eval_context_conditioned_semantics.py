#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.htext.heima_reuse import HeimaOfficialAbstractProjection, official_embedding_replacement

import scripts.run_context_conditioned_heima_baseline as base


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def per_sample_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    valid = shift_labels != -100
    logp = F.log_softmax(shift_logits, dim=-1)
    losses = torch.zeros_like(shift_labels, dtype=torch.float)
    losses[valid] = -logp[valid, shift_labels[valid]]
    denom = valid.sum(dim=1).clamp_min(1)
    return losses.sum(dim=1) / denom


def load_recover(seed: int):
    tokenizer, models = base.load_models(with_b=0)
    model_a = models[0]
    ckpt = torch.load(base.CKPT / f"seed_{seed}" / "recover_encoder" / "checkpoint.pt", map_location="cpu")
    model_a.load_state_dict(ckpt["model_a"])
    model_a.eval()
    return tokenizer, model_a


def load_interpreter(seed: int, model_a, stage: int):
    _, models = base.load_models(with_b=1)
    model_b = models[1]
    projector = HeimaOfficialAbstractProjection(model_a.config.n_embd, model_b.config.n_embd).to(next(model_a.parameters()).device)
    ckpt = torch.load(base.CKPT / f"seed_{seed}" / f"interpreter_stage_{stage}" / "checkpoint.pt", map_location="cpu")
    model_b.load_state_dict(ckpt["model_b"])
    projector.load_state_dict(ckpt["projector"])
    model_b.eval()
    projector.eval()
    return model_b, projector


def get_val_z(model_a, tokenizer, rows: list[dict], batch_size: int) -> torch.Tensor:
    chunks = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            _, _, _, h, _ = base.encoder_forward(model_a, tokenizer, batch, 3)
            chunks.append(h.detach())
    return torch.cat(chunks, dim=0)


def paired_retrieval(model_b, tokenizer, projector, rows: list[dict], z_stage: torch.Tensor, stage: int) -> dict:
    by_group = defaultdict(list)
    for i, row in enumerate(rows):
        by_group[row["pair_group_id"]].append(i)
    ranks = []
    with torch.no_grad():
        for gid, idxs in by_group.items():
            for i in idxs:
                scores = []
                for j in idxs:
                    loss, logits, labels = base.decoder_forward(model_b, tokenizer, [rows[i]], z_stage[j : j + 1], projector, stage, "qz")
                    scores.append(float(per_sample_nll(logits, labels)[0].item()))
                order = sorted(range(len(scores)), key=lambda k: scores[k])
                own = idxs.index(i)
                ranks.append(order.index(own) + 1)
    return {
        "R@1": sum(r <= 1 for r in ranks) / len(ranks),
        "R@2": sum(r <= 2 for r in ranks) / len(ranks),
        "R@4": sum(r <= 4 for r in ranks) / len(ranks),
        "random_R@1": 0.25,
        "mean_rank": sum(ranks) / len(ranks),
    }


def greedy_decode(model_b, tokenizer, projector, record: dict, z: torch.Tensor, stage: int, max_new_tokens: int = 64) -> str:
    device = next(model_b.parameters()).device
    prompt = base.decoder_prompt(record, stage, "qz")
    pids = base.tok(tokenizer, prompt)
    tid = tokenizer.convert_tokens_to_ids(base.TOKENS[stage - 1])
    locs = [i for i, x in enumerate(pids) if x == tid]
    if not locs:
        raise RuntimeError("missing latent slot")
    generated: list[int] = []
    for _ in range(max_new_tokens):
        ids = pids + generated
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        embeds = model_b.get_input_embeddings()(input_ids)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        mask[0, locs[0]] = True
        projected = projector(z.to(device).view(1, -1))
        embeds = official_embedding_replacement(embeds, projected.unsqueeze(1), mask)
        attn = torch.ones_like(input_ids)
        out = model_b(inputs_embeds=embeds, attention_mask=attn, use_cache=False)
        nxt = int(out.logits[0, -1].argmax().item())
        generated.append(nxt)
        if nxt == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated, skip_special_tokens=True)


def fact_match(text: str, record: dict, stage: int) -> dict:
    nums = set(base.NUM_RE.findall(text))
    target_nums = set(base.NUM_RE.findall(base.stage_text(record, stage)))
    context_facts = set(record.get("context_facts", []))
    inter = set(record.get("intermediate_results", []))
    return {
        "number_match": bool(target_nums and target_nums.issubset(nums)),
        "context_fact_match": bool(context_facts and (context_facts & nums)),
        "intermediate_match": bool(inter and (inter & nums)),
    }


def free_generation_eval(model_b, tokenizer, projector, rows: list[dict], z_stage: torch.Tensor, stage: int, limit: int) -> dict:
    paired = base.make_paired_latent(z_stage, rows)
    stats = defaultdict(int)
    examples = []
    n = min(limit, len(rows))
    with torch.no_grad():
        for i in range(n):
            normal = greedy_decode(model_b, tokenizer, projector, rows[i], z_stage[i], stage)
            shuffled = greedy_decode(model_b, tokenizer, projector, rows[i], paired[i], stage)
            nm = fact_match(normal, rows[i], stage)
            sm = fact_match(shuffled, rows[i], stage)
            for k, v in nm.items():
                stats[f"normal_{k}"] += int(v)
            for k, v in sm.items():
                stats[f"paired_shuffle_{k}"] += int(v)
            if len(examples) < 12:
                examples.append({
                    "id": rows[i]["id"],
                    "stage": stage,
                    "target": base.stage_text(rows[i], stage),
                    "normal": normal,
                    "paired_shuffle": shuffled,
                    "normal_match": nm,
                    "paired_shuffle_match": sm,
                })
    rates = {k: v / n for k, v in stats.items()}
    rates["n"] = n
    rates["examples"] = examples
    return rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--generation-limit", type=int, default=24)
    args = parser.parse_args()
    retrieval, generation = {}, {}
    sample_lines = []
    for seed in args.seeds:
        tokenizer, model_a = load_recover(seed)
        val = base.read_jsonl(base.DATA / f"seed_{seed}" / "validation.jsonl")
        z = get_val_z(model_a, tokenizer, val, args.batch_size)
        retrieval[str(seed)] = {}
        generation[str(seed)] = {}
        for stage in [1, 2, 3]:
            model_b, projector = load_interpreter(seed, model_a, stage)
            retrieval[str(seed)][f"stage_{stage}"] = paired_retrieval(model_b, tokenizer, projector, val, z[:, stage - 1, :], stage)
            generation[str(seed)][f"stage_{stage}"] = free_generation_eval(model_b, tokenizer, projector, val, z[:, stage - 1, :], stage, args.generation_limit)
            for ex in generation[str(seed)][f"stage_{stage}"]["examples"][:2]:
                sample_lines.append(json.dumps({"seed": seed, **ex}, ensure_ascii=False))
    write_json(base.OUT / "paired_retrieval.json", retrieval)
    write_json(base.OUT / "free_generation_fact_match.json", generation)
    with (base.OUT / "sample_decodes.txt").open("a", encoding="utf-8") as f:
        f.write("\nSupplemental free-generation examples\n")
        for line in sample_lines:
            f.write(line + "\n")
    print(json.dumps({"status": "complete", "paired_retrieval": retrieval}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
