#!/usr/bin/env python3
"""Run Heima HF decoder-adapter thinking-slot interventions.

Inference-only audit. The public Heima HF snapshot contains the Llama-3.1-8B
pure-LLM decoder LoRA adapters, but not the abstract_projector_*.pth files
required for full continuous-latent intervention. This runner audits whether the
official decoder adapter depends on the reserved thinking-token slot itself.
"""
from __future__ import annotations

import argparse, gc, json, math, re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/data/zxl/Heima-model-a-only-loss1-formal")
DEFAULT_BASE = Path("/data/zxl/models/meta-llama-Llama-3.1-8B-Instruct")
DEFAULT_ADAPTER = Path("/data/zxl/models/official_heima_hf")
DEFAULT_DATA = Path("/data/zxl/official_heima/datasets/LLaVA-CoT-100k/train.jsonl")
DEFAULT_OUT = ROOT / "reports" / "heima_hf_decoder_slot_intervention"

SECTIONS = {
    "summary": {"token_id": 128013, "tag": "SUMMARY", "prompt": "{question}\nCan you provide the details of thinking progress "},
    "caption": {"token_id": 128014, "tag": "CAPTION", "prompt": "{question}\nCan you provide the thinking progress "},
    "reasoning": {"token_id": 128015, "tag": "REASONING", "prompt": "{question}\nCan you provide the thinking progress "},
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def read_samples(path: Path, limit: int) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            conv = obj.get("conversations", [])
            if len(conv) < 2:
                continue
            q, a = conv[0].get("value", ""), conv[1].get("value", "")
            parts = {"question": q, "id": obj.get("id"), "image": obj.get("image")}
            ok = True
            for section, spec in SECTIONS.items():
                tag = spec["tag"]
                m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", a, flags=re.S | re.I)
                if not m or not m.group(1).strip():
                    ok = False
                    break
                parts[section] = f"<{tag}> {m.group(1).strip()} </{tag}>"
            if ok:
                rows.append(parts)
            if len(rows) >= limit:
                break
    return rows


def words(text: str):
    return re.findall(r"\w+", str(text).lower())


def bleu1(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    if not h:
        return 0.0
    counts = defaultdict(int)
    for t in r:
        counts[t] += 1
    hit = 0
    for t in h:
        if counts[t] > 0:
            hit += 1
            counts[t] -= 1
    bp = 1.0 if len(h) >= len(r) or not r else math.exp(1 - len(r) / max(1, len(h)))
    return bp * hit / len(h)


def rouge_l(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    if not r or not h:
        return 0.0
    dp = [0] * (len(h) + 1)
    for rt in r:
        prev = 0
        for j, ht in enumerate(h, 1):
            old = dp[j]
            dp[j] = prev + 1 if rt == ht else max(dp[j], dp[j - 1])
            prev = old
    return dp[-1] / len(r)


def get_module(model, name: str):
    cur = model
    for part in name.split("."):
        cur = getattr(cur, part)
    return cur


@torch.no_grad()
def merge_lora(model, adapter_file: Path, alpha: int = 32, rank: int = 16):
    sd = torch.load(adapter_file, map_location="cpu")
    prefixes = sorted({k.rsplit(".lora_A.weight", 1)[0] for k in sd if k.endswith(".lora_A.weight")})
    merged = []
    scale = alpha / rank
    for prefix in prefixes:
        module_name = prefix.removeprefix("base_model.model.")
        module = get_module(model, module_name)
        a = sd[prefix + ".lora_A.weight"].to(device=model.device, dtype=model.dtype)
        b = sd[prefix + ".lora_B.weight"].to(device=model.device, dtype=model.dtype)
        delta = torch.matmul(b, a) * scale
        module.weight.add_(delta)
        merged.append({"module": module_name, "shape": list(delta.shape)})
        del a, b, delta
        torch.cuda.empty_cache()
    return {"adapter_file": str(adapter_file), "merged_modules": len(merged), "first_modules": merged[:6], "last_modules": merged[-4:]}


def build_prompt_ids(tokenizer, question: str, section: str, include_slot: bool) -> Tuple[torch.Tensor, int | None]:
    spec = SECTIONS[section]
    prefix = spec["prompt"].format(question=question)
    prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if not include_slot:
        return prefix_ids, None
    return torch.cat([prefix_ids, torch.tensor([spec["token_id"]], dtype=torch.long)], dim=0), int(len(prefix_ids))


@torch.no_grad()
def nll_for_target(model, tokenizer, question: str, section: str, target: str, condition: str, device: str) -> float:
    prompt_ids, slot_pos = build_prompt_ids(tokenizer, question, section, include_slot=(condition != "deleted_slot"))
    target_ids = tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids[0]
    input_ids = torch.cat([prompt_ids, target_ids], dim=0).unsqueeze(0).to(device)
    labels = torch.full_like(input_ids, -100)
    labels[:, len(prompt_ids):] = input_ids[:, len(prompt_ids):]
    if condition == "zero_slot":
        embeds = model.get_input_embeddings()(input_ids)
        embeds[:, slot_pos, :] = 0
        out = model(inputs_embeds=embeds, labels=labels, use_cache=False)
    else:
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
    return float(out.loss.detach().cpu())


@torch.no_grad()
def generate_text(model, tokenizer, question: str, section: str, condition: str, max_new_tokens: int, device: str) -> str:
    prompt_ids, slot_pos = build_prompt_ids(tokenizer, question, section, include_slot=(condition != "deleted_slot"))
    input_ids = prompt_ids.unsqueeze(0).to(device)
    kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False, "pad_token_id": tokenizer.eos_token_id, "eos_token_id": tokenizer.eos_token_id}
    if condition == "zero_slot":
        embeds = model.get_input_embeddings()(input_ids)
        embeds[:, slot_pos, :] = 0
        out = model.generate(inputs_embeds=embeds, attention_mask=torch.ones(input_ids.shape, device=device), **kwargs)
        gen = out[0]
    else:
        out = model.generate(input_ids=input_ids, attention_mask=torch.ones(input_ids.shape, device=device), **kwargs)
        gen = out[0, input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=False)


def summarize(rows: Iterable[dict]):
    acc = defaultdict(list)
    for r in rows:
        for metric in ["nll", "bleu1", "rouge_l"]:
            acc[(r["section"], r["condition"], metric)].append(r[metric])
    out = {}
    for (section, condition, metric), vals in acc.items():
        out.setdefault(section, {}).setdefault(condition, {})[metric] = sum(vals) / len(vals)
    for sec in out.values():
        if "normal" in sec:
            base = sec["normal"]["nll"]
            for metrics in sec.values():
                metrics["nll_delta_vs_normal"] = metrics["nll"] - base
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["normal.jsonl", "zero_slot.jsonl", "deleted_slot.jsonl"]:
        p = args.out_dir / fname
        if p.exists():
            p.unlink()
    samples = read_samples(args.data, args.samples)
    meta = {
        "note": "Public Heima HF snapshot lacks abstract_projector_*.pth; this is official decoder-adapter thinking-slot ablation, not full continuous-latent intervention.",
        "base_model": str(args.base_model),
        "adapter_root": str(args.adapter_root),
        "data": str(args.data),
        "samples": len(samples),
        "conditions": ["normal", "zero_slot", "deleted_slot"],
        "thinking_token_ids": {k: v["token_id"] for k, v in SECTIONS.items()},
        "projector_files_found": [str(p) for p in args.adapter_root.rglob("*projector*")],
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_rows, merge_reports = [], {}
    for section in SECTIONS:
        print(f"[load] section={section}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=True, torch_dtype=torch.bfloat16, device_map={"": args.device}, low_cpu_mem_usage=True)
        model.eval()
        merge_reports[section] = merge_lora(model, args.adapter_root / "decoder" / section / "adapter_model.bin")
        for sample_idx, sample in enumerate(samples):
            for condition in ["normal", "zero_slot", "deleted_slot"]:
                nll = nll_for_target(model, tokenizer, sample["question"], section, sample[section], condition, args.device)
                gen = generate_text(model, tokenizer, sample["question"], section, condition, args.max_new_tokens, args.device)
                row = {"sample_index": sample_idx, "sample_id": sample["id"], "image": sample["image"], "section": section, "condition": condition, "question": sample["question"], "gold_text": sample[section], "generated_text": gen, "nll": nll, "bleu1": bleu1(sample[section], gen), "rouge_l": rouge_l(sample[section], gen)}
                all_rows.append(row)
                with (args.out_dir / f"{condition}.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[row] {section} sample={sample_idx} cond={condition} nll={nll:.4f}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    summary = {"summary": summarize(all_rows), "merge_reports": merge_reports, "metadata": meta}
    (args.out_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# Heima HF Decoder Slot Intervention", "", "Loaded local HF Llama-3.1-8B base and merged the official Heima decoder LoRA adapters section by section.", "", "Limitation: public Heima HF snapshot does not include `abstract_projector_summary.pth`, `abstract_projector_caption.pth`, or `abstract_projector_reasoning.pth`, so this is a decoder thinking-slot ablation, not full A-latent correct/shuffle intervention.", "", "|section|condition|NLL|delta vs normal|BLEU1|ROUGE-L|", "|---|---|---:|---:|---:|---:|"]
    for section, sec in summary["summary"].items():
        for condition, metrics in sec.items():
            md.append("|{}|{}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|".format(section, condition, metrics["nll"], metrics["nll_delta_vs_normal"], metrics["bleu1"], metrics["rouge_l"]))
    md += ["", "## Interpretation", "", "If `zero_slot` or `deleted_slot` is close to `normal`, the official decoder adapter can reconstruct largely from query/prompt and teacher-forced target prefix without requiring the reserved thinking-token embedding. This does not test the missing continuous latent projector."]
    (args.out_dir / "analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
