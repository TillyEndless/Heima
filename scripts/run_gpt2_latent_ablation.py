#!/usr/bin/env python3
"""Pure-text GPT2 latent-token supervision ablation.

This runner is intentionally independent from Heima/Qwen Stage2 code. It uses one
GPT2ForCausalLM as both producer and self-decoder, with no Model B, projector,
role embedding, cumulative latent, or Loss2.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

THINK_TOKEN = "<THINK>"
DEFAULT_DATASET = "/data/zxl/official_heima/datasets/LLaVA-CoT-100k/train.jsonl"
DEFAULT_MODEL = "/data/zxl/models/openai-community-gpt2"
GROUPS = ("G0", "G1", "G2", "G3")


@dataclass
class Example:
    question: str
    cot: str
    answer: str


@dataclass
class LossBundle:
    total_loss: torch.Tensor
    main_loss: torch.Tensor
    self_decode_loss: torch.Tensor
    latent_token_loss: torch.Tensor
    latent_token_accuracy: torch.Tensor
    z: Optional[torch.Tensor]


def read_jsonl(path: str, limit: Optional[int] = None) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _strip_tags(text: str) -> str:
    return re.sub(r"</?[A-Z_]+>", " ", text).replace("\n", " ").strip()


def _extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)(?=<[A-Z_]+>|$)", text, flags=re.S)
    return _strip_tags(m.group(1)) if m else ""


def adapt_record(row: dict) -> Optional[Example]:
    if all(k in row and str(row[k]).strip() for k in ("question", "cot", "answer")):
        return Example(str(row["question"]).strip(), str(row["cot"]).strip(), str(row["answer"]).strip())
    if all(k in row for k in ("question", "reasoning", "answer")):
        cot_parts = [str(row.get(k, "")).strip() for k in ("summary", "caption", "reasoning")]
        cot = " ".join(p for p in cot_parts if p)
        if row.get("question") and cot and row.get("answer"):
            return Example(str(row["question"]).strip(), cot, str(row["answer"]).strip())
    conversations = row.get("conversations") or row.get("conversation")
    if isinstance(conversations, list):
        question = ""
        assistant_text = ""
        for turn in conversations:
            role = str(turn.get("from") or turn.get("role") or "").lower()
            value = str(turn.get("value") or turn.get("content") or "")
            if not question and ("human" in role or role == "user"):
                question = re.sub(r"<image>", "", value, flags=re.I).strip()
            if not assistant_text and ("gpt" in role or "assistant" in role):
                assistant_text = value.strip()
        cot_parts = [_extract_tag(assistant_text, tag) for tag in ("SUMMARY", "CAPTION", "REASONING")]
        cot = " ".join(p for p in cot_parts if p)
        answer = _extract_tag(assistant_text, "CONCLUSION")
        if not answer:
            m = re.search(r"(?:answer|therefore)[:\s]+(.+)$", assistant_text, flags=re.I | re.S)
            answer = _strip_tags(m.group(1)) if m else _strip_tags(assistant_text[-256:])
        if question and cot and answer:
            return Example(question, cot, answer)
    return None


def build_split(dataset_path: str, output_dir: Path, train_samples: int, eval_samples: int, seed: int) -> Dict[str, object]:
    rows = read_jsonl(dataset_path)
    examples = [ex for row in rows if (ex := adapt_record(row)) is not None]
    rng = random.Random(seed)
    rng.shuffle(examples)
    need = train_samples + eval_samples
    if len(examples) < need:
        raise RuntimeError(f"Not enough pure-text CoT examples: have {len(examples)}, need {need}")
    train = examples[:train_samples]
    eval_set = examples[train_samples:need]
    payload = {
        "seed": seed,
        "dataset_path": dataset_path,
        "train": [asdict(x) for x in train],
        "eval": [asdict(x) for x in eval_set],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "data_split.json"
    split_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "dataset_path": dataset_path,
        "total_rows": len(rows),
        "usable_examples": len(examples),
        "train_samples": len(train),
        "eval_samples": len(eval_set),
        "split_path": str(split_path),
    }


def encode_text(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def make_lm_batch(tokenizer, records: Sequence[Example], mode: str, max_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
    input_ids, labels, think_positions = [], [], []
    for ex in records:
        if mode == "vanilla":
            prefix = f"Question:\n{ex.question}\n\nReasoning:\n"
            target = f"{ex.cot}\nAnswer:\n{ex.answer}{tokenizer.eos_token}"
            ids = encode_text(tokenizer, prefix + target)[:max_length]
            prefix_len = min(len(encode_text(tokenizer, prefix)), len(ids))
            lab = [-100] * prefix_len + ids[prefix_len:]
            think_positions.append(-1)
        elif mode == "latent_main":
            prefix = f"Question:\n{ex.question}\n\n"
            target = f"{THINK_TOKEN}\nAnswer:\n{ex.answer}{tokenizer.eos_token}"
            ids = encode_text(tokenizer, prefix + target)[:max_length]
            prefix_len = min(len(encode_text(tokenizer, prefix)), len(ids))
            lab = [-100] * len(ids)
            for pos in range(prefix_len + 1, len(ids)):
                lab[pos] = ids[pos]
            think_positions.append(prefix_len if prefix_len < len(ids) else -1)
        else:
            raise ValueError(mode)
        input_ids.append(torch.tensor(ids, dtype=torch.long))
        labels.append(torch.tensor(lab[: len(ids)], dtype=torch.long))
    return pad_batch(tokenizer, input_ids, labels, think_positions, device)


def make_self_decode_batch(tokenizer, records: Sequence[Example], max_length: int, device: torch.device, include_question: bool = True, include_latent: bool = True) -> Dict[str, torch.Tensor]:
    input_ids, labels, slot_positions = [], [], []
    for ex in records:
        parts = []
        if include_question:
            parts.append(f"Question:\n{ex.question}\n\n")
        if include_latent:
            parts.append(f"Latent:\n{THINK_TOKEN}\n")
        parts.append("Target reasoning:\n")
        prefix = "".join(parts)
        target = f"{ex.cot}{tokenizer.eos_token}"
        ids = encode_text(tokenizer, prefix + target)[:max_length]
        prefix_ids = encode_text(tokenizer, prefix)
        prefix_len = min(len(prefix_ids), len(ids))
        lab = [-100] * prefix_len + ids[prefix_len:]
        if include_latent:
            think_ids = encode_text(tokenizer, THINK_TOKEN)
            pos = find_subsequence(ids, think_ids)
        else:
            pos = -1
        input_ids.append(torch.tensor(ids, dtype=torch.long))
        labels.append(torch.tensor(lab[: len(ids)], dtype=torch.long))
        slot_positions.append(pos)
    return pad_batch(tokenizer, input_ids, labels, slot_positions, device)


def pad_batch(tokenizer, input_ids: Sequence[torch.Tensor], labels: Sequence[torch.Tensor], positions: Sequence[int], device: torch.device) -> Dict[str, torch.Tensor]:
    max_len = max(x.numel() for x in input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    padded_ids, padded_labels, masks = [], [], []
    for ids, lab in zip(input_ids, labels):
        pad = max_len - ids.numel()
        padded_ids.append(F.pad(ids, (0, pad), value=pad_id))
        padded_labels.append(F.pad(lab, (0, pad), value=-100))
        masks.append(F.pad(torch.ones_like(ids), (0, pad), value=0))
    return {
        "input_ids": torch.stack(padded_ids).to(device),
        "labels": torch.stack(padded_labels).to(device),
        "attention_mask": torch.stack(masks).to(device),
        "positions": torch.tensor(positions, dtype=torch.long, device=device),
    }


def find_subsequence(seq: Sequence[int], pattern: Sequence[int]) -> int:
    if not pattern:
        return -1
    for i in range(0, len(seq) - len(pattern) + 1):
        if list(seq[i : i + len(pattern)]) == list(pattern):
            return i
    return -1


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)


def token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    if mask.sum() == 0:
        return torch.zeros((), device=logits.device)
    return shift_logits.argmax(dim=-1).eq(shift_labels).masked_select(mask).float().mean()


def latent_token_labels(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    labels = torch.full_like(batch["input_ids"], -100)
    for row, pos in enumerate(batch["positions"].tolist()):
        if pos >= 0:
            labels[row, pos] = batch["input_ids"][row, pos]
    return labels


def extract_z(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(hidden.size(0), device=hidden.device)
    safe_pos = positions.clamp_min(0)
    return hidden[rows, safe_pos]


def replace_slot_embeddings(input_embeddings: torch.Tensor, z: torch.Tensor, positions: torch.Tensor, detach: bool = False) -> torch.Tensor:
    out = input_embeddings.clone()
    source = z.detach() if detach else z
    for row, pos in enumerate(positions.tolist()):
        if pos >= 0:
            out[row, pos, :] = source[row]
    return out


def compute_losses(model, tokenizer, records: Sequence[Example], group: str, lambda_self: float, lambda_latent: float, max_length: int, device: torch.device, detach_self: bool = False) -> LossBundle:
    if group == "G0":
        batch = make_lm_batch(tokenizer, records, "vanilla", max_length, device)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        main = causal_lm_loss(out.logits, batch["labels"])
        zero = torch.zeros((), device=device)
        return LossBundle(main, main, zero, zero, zero, None)

    main_batch = make_lm_batch(tokenizer, records, "latent_main", max_length, device)
    out = model(input_ids=main_batch["input_ids"], attention_mask=main_batch["attention_mask"], output_hidden_states=True)
    main = causal_lm_loss(out.logits, main_batch["labels"])
    z = extract_z(out.hidden_states[-1], main_batch["positions"])

    latent_labels = latent_token_labels(main_batch)
    latent_loss = causal_lm_loss(out.logits, latent_labels) if group in ("G1", "G3") else torch.zeros((), device=device)
    latent_acc = token_accuracy(out.logits, latent_labels) if group in ("G1", "G3") else torch.zeros((), device=device)

    if group in ("G2", "G3"):
        sd_batch = make_self_decode_batch(tokenizer, records, max_length, device)
        embeds = model.get_input_embeddings()(sd_batch["input_ids"])
        embeds = replace_slot_embeddings(embeds, z, sd_batch["positions"], detach=detach_self)
        sd_out = model(inputs_embeds=embeds, attention_mask=sd_batch["attention_mask"])
        self_loss = causal_lm_loss(sd_out.logits, sd_batch["labels"])
    else:
        self_loss = torch.zeros((), device=device)

    total = main + lambda_self * self_loss + lambda_latent * latent_loss
    return LossBundle(total, main, self_loss, latent_loss, latent_acc, z)


def evaluate_intervention(model, tokenizer, records: Sequence[Example], group: str, max_length: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        base = compute_losses(model, tokenizer, records, group if group != "G0" else "G0", 0.1, 0.05, max_length, device)
        main_loss = float(base.main_loss.cpu())
        latent_acc = float(base.latent_token_accuracy.cpu())
        if group not in ("G2", "G3"):
            return {
                "main_loss": main_loss,
                "latent_token_accuracy": latent_acc,
                "NLL_Q_correct": None,
                "NLL_Q_shuffle": None,
                "NLL_Q_only": None,
                "NLL_Z_only": None,
                "shuffle_margin": None,
                "latent_gain": None,
            }
        main_batch = make_lm_batch(tokenizer, records, "latent_main", max_length, device)
        out = model(input_ids=main_batch["input_ids"], attention_mask=main_batch["attention_mask"], output_hidden_states=True)
        z = extract_z(out.hidden_states[-1], main_batch["positions"])
        losses = {}
        for name, zz, include_q, include_z in (
            ("Q_correct", z, True, True),
            ("Q_shuffle", z[torch.randperm(z.size(0), device=device)], True, True),
            ("Q_only", z, True, False),
            ("Z_only", z, False, True),
        ):
            sd_batch = make_self_decode_batch(tokenizer, records, max_length, device, include_question=include_q, include_latent=include_z)
            if include_z:
                embeds = model.get_input_embeddings()(sd_batch["input_ids"])
                embeds = replace_slot_embeddings(embeds, zz, sd_batch["positions"], detach=False)
                sd_out = model(inputs_embeds=embeds, attention_mask=sd_batch["attention_mask"])
            else:
                sd_out = model(input_ids=sd_batch["input_ids"], attention_mask=sd_batch["attention_mask"])
            losses[name] = float(causal_lm_loss(sd_out.logits, sd_batch["labels"]).cpu())
        return {
            "main_loss": main_loss,
            "latent_token_accuracy": latent_acc,
            "NLL_Q_correct": losses["Q_correct"],
            "NLL_Q_shuffle": losses["Q_shuffle"],
            "NLL_Q_only": losses["Q_only"],
            "NLL_Z_only": losses["Z_only"],
            "shuffle_margin": losses["Q_shuffle"] - losses["Q_correct"],
            "latent_gain": losses["Q_only"] - losses["Q_correct"],
        }


def exact_answer_accuracy(generations: Sequence[str], records: Sequence[Example]) -> float:
    hits = 0
    for gen, ex in zip(generations, records):
        gold = ex.answer.strip().lower()
        hits += int(bool(gold) and gold in gen.lower())
    return hits / max(1, len(records))


def generate_condition(model, tokenizer, records: Sequence[Example], max_length: int, device: torch.device, condition: str) -> List[dict]:
    model.eval()
    outputs = []
    with torch.no_grad():
        z = None
        if condition in {"normal", "shuffle", "zero"}:
            main_batch = make_lm_batch(tokenizer, records, "latent_main", max_length, device)
            out = model(input_ids=main_batch["input_ids"], attention_mask=main_batch["attention_mask"], output_hidden_states=True)
            z = extract_z(out.hidden_states[-1], main_batch["positions"])
            if condition == "shuffle":
                z = z[torch.randperm(z.size(0), device=device)]
            elif condition == "zero":
                z = torch.zeros_like(z)
        for idx, ex in enumerate(records):
            if condition == "q_only":
                prompt = f"Question:\n{ex.question}\n\nTarget reasoning:\n"
                ids = torch.tensor([encode_text(tokenizer, prompt)], dtype=torch.long, device=device)
                gen = model.generate(input_ids=ids, max_new_tokens=48, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            else:
                prompt = f"Question:\n{ex.question}\n\nLatent:\n{THINK_TOKEN}\nTarget reasoning:\n"
                ids = torch.tensor([encode_text(tokenizer, prompt)], dtype=torch.long, device=device)
                embeds = model.get_input_embeddings()(ids)
                pos = find_subsequence(ids[0].tolist(), encode_text(tokenizer, THINK_TOKEN))
                embeds[0, pos, :] = z[idx]
                gen = model.generate(inputs_embeds=embeds, attention_mask=torch.ones(ids.shape, device=device), max_new_tokens=48, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(gen[0], skip_special_tokens=False)
            outputs.append({"condition": condition, "question": ex.question, "gold_answer": ex.answer, "generated_text": text})
    return outputs


def load_hf(model_path: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = str(model_path)
    if "gpt2-xl" in name.lower():
        raise ValueError("GPT2-xl is intentionally forbidden for this ablation")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    added = tokenizer.add_special_tokens({"additional_special_tokens": [THINK_TOKEN]})
    if added:
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    return model, tokenizer


def trainable_parameter_report(model, output_path: Path) -> Dict[str, object]:
    in_emb = model.get_input_embeddings()
    out_emb = model.get_output_embeddings()
    report = {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "embed_tokens_trainable": any(p.requires_grad for p in in_emb.parameters()),
        "lm_head_trainable": any(p.requires_grad for p in out_emb.parameters()),
        "embed_tokens_has_grad": any(p.grad is not None and float(p.grad.detach().abs().sum().cpu()) > 0 for p in in_emb.parameters()),
        "lm_head_has_grad": any(p.grad is not None and float(p.grad.detach().abs().sum().cpu()) > 0 for p in out_emb.parameters()),
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_smoke(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_info = build_split(args.dataset, out_dir, args.train_samples, args.eval_samples, args.seed)
    split = json.loads((out_dir / "data_split.json").read_text(encoding="utf-8"))
    train_records = [Example(**x) for x in split["train"]]
    eval_records = [Example(**x) for x in split["eval"]]
    comparison: Dict[str, object] = {"split": split_info, "groups": {}}

    for group in GROUPS:
        group_dir = out_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        model, tokenizer = load_hf(args.model_path, device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for step in range(args.steps):
            batch = train_records[(step * args.batch_size) % len(train_records) : ((step + 1) * args.batch_size) % len(train_records)]
            if not batch:
                batch = train_records[: args.batch_size]
            opt.zero_grad(set_to_none=True)
            losses = compute_losses(model, tokenizer, batch, group, args.lambda_self, args.lambda_latent, args.max_length, device)
            losses.total_loss.backward()
            opt.step()
        trainable_parameter_report(model, group_dir / "trainable_parameter_report.json")
        metrics = evaluate_intervention(model, tokenizer, eval_records, group, args.max_length, device)
        ckpt_path = group_dir / "checkpoint.pt"
        torch.save({"model": model.state_dict(), "group": group, "tokenizer_len": len(tokenizer)}, ckpt_path)
        reloaded, _ = load_hf(args.model_path, device)
        missing, unexpected = reloaded.load_state_dict(torch.load(ckpt_path, map_location=device)["model"], strict=False)
        metrics["checkpoint_save_load_ok"] = not unexpected and all("wte" not in m for m in missing)
        if group in ("G2", "G3"):
            normal = generate_condition(model, tokenizer, eval_records, args.max_length, device, "normal")
            shuffle = generate_condition(model, tokenizer, eval_records, args.max_length, device, "shuffle")
            zero = generate_condition(model, tokenizer, eval_records, args.max_length, device, "zero")
            write_jsonl(group_dir / "normal_generation.jsonl", normal)
            write_jsonl(group_dir / "shuffle_generation.jsonl", shuffle)
            write_jsonl(group_dir / "zero_generation.jsonl", zero)
            metrics["generation_accuracy"] = exact_answer_accuracy([x["generated_text"] for x in normal], eval_records)
            metrics["generation_token_overlap"] = token_overlap_summary(normal, shuffle)
        else:
            metrics["generation_accuracy"] = None
            metrics["generation_token_overlap"] = None
        (group_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        comparison["groups"][group] = metrics
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def token_overlap_summary(a_rows: Sequence[dict], b_rows: Sequence[dict]) -> float:
    vals = []
    for a, b in zip(a_rows, b_rows):
        sa = set(str(a["generated_text"]).split())
        sb = set(str(b["generated_text"]).split())
        vals.append(len(sa & sb) / max(1, len(sa | sb)))
    return sum(vals) / max(1, len(vals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run 32/8 small validation for G0-G3; not formal training.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="/data/zxl/runs/gpt2_latent_token_ablation_smoke")
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lambda-self", type=float, default=0.1)
    parser.add_argument("--lambda-latent", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    if not args.smoke:
        raise SystemExit("This entrypoint currently only runs --smoke; formal training is intentionally not started.")
    comparison = run_smoke(args)
    print(json.dumps({"output_dir": args.output_dir, "groups": list(comparison["groups"].keys())}, indent=2))


if __name__ == "__main__":
    main()
