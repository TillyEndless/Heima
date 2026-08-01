#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, hashlib, json, math, os, random, re, subprocess, time, traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(os.environ.get("REPO_ROOT", "/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot"))
ASSET_ROOT = Path(os.environ.get("ASSET_ROOT", "/data2/zhouxiaoling/latent_cot/Heima-overnight-qwen7b"))
HF_HOME = Path(os.environ.get("HF_HOME", "/weights2/zhouxiaoling/hf_cache"))
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_REVISION = "916b56a44061fd5cd7d6a8fb632557ed4f724f60"
THINK = "<THINK>"
REPORT = REPO / "reports/stage1_repair"
STATUS = REPO / "status/stage1_repair"
MANIFEST = REPO / "data/manifests/stage1_debug32_no_truncation.json"
OLD_OVERFIT_ADAPTER = REPO / "checkpoints/stage_validity_gate_d_overfit/final_adapter"
DATASET_ID = "a-m-team/AM-DeepSeek-R1-Distilled-1.4M"
ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?)")
NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?")


def iso(t=None):
    return datetime.fromtimestamp(t or time.time(), tz=timezone.utc).isoformat()


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_status(name: str, state: str, **kw):
    write_json(STATUS / f"{name}.json", dict(stage=name, status=state, time=time.time(), iso=iso(), **kw))


def normalize_num(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "")
    if s.endswith(".0"):
        s = s[:-2]
    return s


def parse_answer(text):
    text = str(text or "")
    m = ANSWER_RE.search(text)
    if m:
        return normalize_num(m.group(1))
    low = text.lower()
    for marker in ["final answer is", "therefore", "answer is", "answer:", "<answer>", "boxed"]:
        idx = low.rfind(marker)
        if idx >= 0:
            nums = NUM_RE.findall(text[idx:])
            if nums:
                return normalize_num(nums[-1])
    nums = NUM_RE.findall(text)
    return normalize_num(nums[-1]) if nums else None


def parse_record(raw, idx):
    info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
    q = raw.get("question") or info.get("question") or raw.get("prompt") or raw.get("input")
    cot = info.get("think_content") or raw.get("think_content") or raw.get("cot")
    ans = info.get("answer_content") or raw.get("answer_content") or raw.get("answer")
    content = raw.get("content") or raw.get("assistant") or raw.get("response") or raw.get("messages")
    if isinstance(content, list):
        content = "\n".join(str(x.get("content", x)) if isinstance(x, dict) else str(x) for x in content)
    if (not cot or not ans) and content:
        tm = re.search(r"<think>(.*?)</think>", str(content), re.S | re.I)
        am = re.search(r"<answer>(.*?)</answer>", str(content), re.S | re.I)
        if not cot and tm:
            cot = tm.group(1).strip()
        if not ans and am:
            ans = am.group(1).strip()
    if not ans and content:
        ans = parse_answer(content)
    return dict(sample_id=str(raw.get("id", idx)), source=DATASET_ID, question=str(q or "").strip(), gold_cot=str(cot or "").strip(), answer=str(ans or "").strip(), reference_answer=str(ans or "").strip())


def load_stack():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    return torch, AutoTokenizer, AutoModelForCausalLM, LoraConfig, PeftModel, get_peft_model


def tokenizer_only():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(HF_HOME), trust_remote_code=True)
    tok.add_special_tokens({"additional_special_tokens": [THINK]})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def get_rows(limit=None):
    p = ASSET_ROOT / "data/processed_large/qwen7b_stage1/smoke1k.json"
    if p.exists():
        rows = json.loads(p.read_text())
    else:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(DATASET_ID, filename="am_0.9M_sample_1k.jsonl", repo_type="dataset", cache_dir=str(HF_HOME))
        rows = [parse_record(json.loads(line), i) for i, line in enumerate(open(local, encoding="utf-8"))]
    tok = tokenizer_only()
    out = []
    for i, r in enumerate(rows[:limit] if limit else rows):
        rr = dict(r)
        rr.setdefault("sample_id", str(i))
        rr["q_token_count"] = len(tok("Question:\n" + rr["question"] + "\n\n", add_special_tokens=False)["input_ids"])
        rr["answer_token_count"] = len(tok("\nAnswer:\n" + rr["answer"] + tok.eos_token, add_special_tokens=False)["input_ids"])
        rr["cot_token_count"] = len(tok(rr["gold_cot"], add_special_tokens=False)["input_ids"])
        rr["raw_K"] = max(1, round(0.5 * rr["cot_token_count"])) if rr["cot_token_count"] else 0
        out.append(rr)
    return [r for r in out if r["question"] and r["gold_cot"] and r["answer"]]


def load_model(adapter=None, train=False):
    torch, AutoTokenizer, AutoModel, LoraConfig, PeftModel, get_peft_model = load_stack()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(HF_HOME), trust_remote_code=True)
    tok.add_special_tokens({"additional_special_tokens": [THINK]})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(HF_HOME), torch_dtype=torch.bfloat16, trust_remote_code=True)
    base.resize_token_embeddings(len(tok))
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    if adapter:
        model = PeftModel.from_pretrained(base, adapter, is_trainable=train)
    else:
        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], modules_to_save=["embed_tokens", "lm_head"])
        model = get_peft_model(base, cfg)
    model.to("cuda:0")
    model.train(train)
    return torch, tok, model




def plan_stage1_lengths(q_tokens, raw_k, answer_tokens, *, max_q=256, max_latent=128, max_answer=128, max_seq=512, strict=False):
    """Plan Stage-1 lengths without silently truncating answer or boundary tokens."""
    reasons = []
    if q_tokens > max_q:
        reasons.append("question_too_long")
    if raw_k > max_latent:
        reasons.append("raw_K_exceeds_max_latent")
    if answer_tokens > max_answer:
        reasons.append("answer_too_long")
    used_k_for_seq = raw_k if strict else min(raw_k, max_latent)
    if min(q_tokens, max_q) + used_k_for_seq + min(answer_tokens, max_answer) > max_seq:
        reasons.append("sequence_too_long")
    if strict and reasons:
        return dict(keep=False, filter_reason=reasons, question_truncated=q_tokens > max_q, answer_truncated=answer_tokens > max_answer, latent_cap_hit=raw_k > max_latent)
    used_q = min(q_tokens, max_q)
    used_k = raw_k if strict else min(raw_k, max_latent)
    used_answer = min(answer_tokens, max_answer)
    if used_q + used_k + used_answer > max_seq:
        used_q = max(0, max_seq - used_k - used_answer)
    return dict(keep=True, filter_reason=reasons, used_q=used_q, used_K=used_k, used_answer=used_answer, question_truncated=used_q < q_tokens, answer_truncated=used_answer < answer_tokens, latent_cap_hit=used_k < raw_k, sequence_length=used_q + used_k + used_answer)


def build_stage1_batch(torch, tok, rows, *, max_q=256, max_latent=128, max_answer=128, max_seq=512, strict=False, balanced="weighted"):
    think_id = tok.convert_tokens_to_ids(THINK)
    ids_rows, labels_rows, seg_rows, metas = [], [], [], []
    for r in rows:
        qids_full = tok("Question:\n" + r["question"] + "\n\n", add_special_tokens=False)["input_ids"]
        ans_full = tok("\nAnswer:\n" + r["answer"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        raw_k = int(r.get("raw_K") or max(1, round(0.5 * len(tok(r["gold_cot"], add_special_tokens=False)["input_ids"]))))
        q_tr = len(qids_full) > max_q
        cap_hit = raw_k > max_latent
        a_tr = len(ans_full) > max_answer
        used_k = raw_k if strict else min(raw_k, max_latent)
        ids = qids_full[:max_q] + [think_id] * used_k + ans_full[:max_answer]
        seq_over = len(ids) > max_seq
        if strict and (q_tr or cap_hit or a_tr or seq_over):
            reason = []
            if q_tr: reason.append("question_too_long")
            if cap_hit: reason.append("raw_K_exceeds_max_latent")
            if a_tr: reason.append("answer_too_long")
            if seq_over: reason.append("sequence_too_long")
            raise ValueError("strict sample would truncate: " + ",".join(reason))
        if len(ids) > max_seq:
            keep_q = max(0, max_seq - used_k - len(ans_full[:max_answer]))
            qids = qids_full[:min(max_q, keep_q)]
            ids = qids + [think_id] * used_k + ans_full[:max_answer]
            q_tr = True
        else:
            qids = qids_full[:max_q]
        labels = [-100] * len(qids) + [think_id] * used_k + ans_full[:max_answer]
        seg = ["question"] * len(qids) + ["think"] * used_k + ["answer"] * len(ans_full[:max_answer])
        ids_rows.append(ids); labels_rows.append(labels); seg_rows.append(seg)
        metas.append(dict(sample_id=r.get("sample_id"), raw_K=raw_k, used_K=used_k, q_tokens=len(qids_full), answer_tokens=len(ans_full), stage1_sequence_length=len(ids), question_truncated=q_tr, answer_truncated=a_tr, latent_cap_hit=cap_hit, sequence_over_limit=seq_over, filter_reason=None))
    mx = max(map(len, ids_rows)); pad = tok.pad_token_id or tok.eos_token_id
    input_ids = torch.full((len(rows), mx), pad, dtype=torch.long, device="cuda:0")
    labels = torch.full((len(rows), mx), -100, dtype=torch.long, device="cuda:0")
    attention_mask = torch.zeros_like(input_ids)
    for i, ids in enumerate(ids_rows):
        input_ids[i, :len(ids)] = torch.tensor(ids, device="cuda:0")
        labels[i, :len(ids)] = torch.tensor(labels_rows[i], device="cuda:0")
        attention_mask[i, :len(ids)] = 1
        seg_rows[i] += ["pad"] * (mx - len(ids))
    return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask), seg_rows, metas


def token_acc(torch, logits, labels, segs):
    pred = logits[:, :-1].argmax(-1)
    lab = labels[:, 1:]
    out = {}
    for name in ["think", "answer"]:
        mask_rows = []
        for row in segs:
            mask_rows.append([s == name for s in row[1:]])
        mask = torch.tensor(mask_rows, dtype=torch.bool, device=lab.device) & lab.ne(-100)
        out[name + "_token_accuracy"] = float(pred[mask].eq(lab[mask]).float().mean().detach().cpu()) if mask.any() else 0.0
    return out


def classify_failure(prompt, generation, pred, gold, used_k, max_new_tokens):
    txt = str(generation)
    think_count = txt.count(THINK)
    reached = False
    if THINK in txt and "Answer:" not in txt:
        return "think_never_ended"
    if think_count and think_count != used_k:
        return "wrong_think_count"
    if "Answer:" not in prompt and "Answer:" not in txt and pred is None:
        return "answer_boundary_missing"
    if reached and pred is None:
        return "answer_truncated"
    if pred is None and NUM_RE.search(txt):
        return "answer_generated_but_parser_failed"
    if pred is None:
        return "malformed_answer"
    if gold is not None and pred != gold:
        return "valid_but_wrong_answer"
    return "other"


def generate_one(torch, tok, model, row, mode, fixed_k=64, max_new_tokens=128, forced_k_override=None):
    raw_k = int(row["raw_K"])
    if mode == "forced_k":
        k = int(forced_k_override) if forced_k_override is not None else raw_k
        prompt = "Question:\n" + row["question"] + "\n\n" + (THINK * k) + "\nAnswer:\n"
    elif mode == "fixed_k":
        k = fixed_k
        prompt = "Question:\n" + row["question"] + "\n\n" + (THINK * k) + "\nAnswer:\n"
    elif mode == "free_k":
        k = None
        prompt = "Question:\n" + row["question"] + "\n\n"
    else:
        raise ValueError(mode)
    enc = tok(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    raw = tok.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=False)
    pred = parse_answer(raw)
    gold = parse_answer(row["answer"])
    max_reached = gen.shape[1] - enc.input_ids.shape[1] >= max_new_tokens
    return dict(sample_id=row["sample_id"], mode=mode, question=row["question"], full_gold_cot=row["gold_cot"], full_gold_answer=row["answer"], raw_K=raw_k, used_K=k, raw_generation=raw, generated_THINK_count=raw.count(THINK), parsed_answer=pred, gold_answer=gold, stop_reason="max_new_tokens" if max_reached else "eos_or_stop", max_new_tokens_reached=bool(max_reached), prompt_template=prompt, correct=bool(pred is not None and gold is not None and pred == gold))


def summarize_generation(rows):
    n = max(len(rows), 1)
    valid = sum(r["parsed_answer"] is not None for r in rows)
    correct = sum(r["correct"] for r in rows)
    malformed = sum(r["parsed_answer"] is None for r in rows)
    stop = sum(not r["max_new_tokens_reached"] for r in rows)
    return dict(n=len(rows), valid_answer_rate=valid/n, answer_accuracy=correct/n, malformed_rate=malformed/n, stop_success_rate=stop/n, mean_generated_THINK_count=sum(r["generated_THINK_count"] for r in rows)/n)


def load_debug32_or_first32():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())["samples"]
    return get_rows()[:32]


def old_forensics():
    """Offline forensic taxonomy for the previous failed Gate-D set.

    The earlier full-generation attempt was not a clean ability test because the
    first-32 dataset includes very long prompts. We classify all 32 samples by
    protocol/length first, and preserve previous 8-case failure evidence if it
    exists. Clean generation is rerun later on debug32_no_truncation.
    """
    tok = tokenizer_only()
    rows = get_rows()[:32]
    previous = {}
    prev_path = REPO / "reports/stage_validity/failure_cases.jsonl"
    if prev_path.exists():
        for line in prev_path.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                previous[str(obj.get("sample_id"))] = obj
    cases = []
    for row in rows:
        used_k = min(int(row["raw_K"]), 128)
        prompt = "Question:\n" + row["question"] + "\n\n" + (THINK * used_k) + "\nAnswer:\n"
        prompt_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
        prev = previous.get(str(row["sample_id"]), {})
        pred = prev.get("pred")
        gold = parse_answer(row["answer"])
        correct = bool(pred is not None and gold is not None and str(pred) == str(gold))
        if prompt_len > 4096:
            klass = "prompt_mismatch"
            stop = "skipped_prompt_too_long"
        elif prev:
            klass = "correct" if correct else "valid_but_wrong_answer" if pred is not None else "malformed_answer"
            stop = "reused_previous_gate_d_case"
        else:
            klass = "other"
            stop = "not_regenerated_old_dirty_set"
        cases.append(dict(sample_id=row["sample_id"], question=row["question"], full_gold_cot=row["gold_cot"], full_gold_answer=row["answer"], raw_K=int(row["raw_K"]), used_K=used_k, prompt_token_length=prompt_len, raw_generation=prev.get("text", ""), generated_THINK_count=str(prev.get("text", "")).count(THINK), parsed_answer=pred, gold_answer=gold, stop_reason=stop, max_new_tokens_reached=False, prompt_template=prompt[:2000], correct=correct, failure_class=klass))
    counts = Counter(c["failure_class"] for c in cases)
    write_json(REPORT / "free_generation_failure_taxonomy.json", dict(checkpoint=str(OLD_OVERFIT_ADAPTER), evaluated_samples=32, mode="offline_old_dirty_set_taxonomy", counts=dict(counts), summary=summarize_generation(cases), cases=cases))
    md = ["# Free Generation Failure Cases", "", "Old Gate-D first32 is a dirty/truncated set. This report classifies all 32 without launching slow generation; clean generation is performed on debug32_no_truncation.", "", "| class | count |", "|---|---:|"]
    for k, v in sorted(counts.items()):
        md.append(f"| {k} | {v} |")
    md += [""]
    for c in cases:
        if c["failure_class"] != "correct":
            md += [f"## sample {c['sample_id']} - {c['failure_class']}", "", f"gold: `{c['gold_answer']}` pred: `{c['parsed_answer']}` raw_K={c['raw_K']} used_K={c['used_K']} prompt_tokens={c['prompt_token_length']}", "", "```text", c["raw_generation"][:1600], "```", ""]
    (REPORT / "free_generation_failure_cases.md").write_text("\n".join(md) + "\n")
    write_status("old_forensics", "complete", summary=summarize_generation(cases), counts=dict(counts))

def adapter_audit():
    torch, tok, model = load_model(OLD_OVERFIT_ADAPTER if OLD_OVERFIT_ADAPTER.exists() else None, train=True)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    embed = [(n, p.numel(), p.requires_grad) for n, p in model.named_parameters() if "embed_tokens" in n]
    lm_head = [(n, p.numel(), p.requires_grad) for n, p in model.named_parameters() if "lm_head" in n]
    modules = [(n, p.numel(), p.requires_grad) for n, p in model.named_parameters() if "modules_to_save" in n]
    emb_w = model.get_input_embeddings().weight
    out_w = model.get_output_embeddings().weight
    tied = emb_w.data_ptr() == out_w.data_ptr()
    adapter_file = OLD_OVERFIT_ADAPTER / "adapter_model.safetensors"
    ck_size = adapter_file.stat().st_size if adapter_file.exists() else None
    rep = dict(model_id=MODEL_ID, revision=MODEL_REVISION, adapter=str(OLD_OVERFIT_ADAPTER), total_model_parameters=total, peft_trainable_parameters=trainable, lora_ab_parameters=lora, embed_tokens_parameters=sum(x[1] for x in embed), embed_tokens_trainable=any(x[2] for x in embed), lm_head_parameters=sum(x[1] for x in lm_head), lm_head_trainable=any(x[2] for x in lm_head), modules_to_save_parameters=sum(x[1] for x in modules), modules_to_save_entries=[dict(name=n, parameters=c, trainable=t) for n, c, t in modules[:80]], embed_lm_head_tied=tied, adapter_checkpoint_size_bytes=ck_size, explanation="The ~1.107B adapter count is dominated by modules_to_save copies of full embed_tokens and lm_head. DeepSeek/Qwen has a large vocab times hidden-size matrix; saving both input embedding and output head as trainable full modules makes the PEFT adapter much larger than LoRA A/B alone.")
    write_json(REPORT / "adapter_parameter_audit.json", rep)
    md = f"""# Adapter Parameter Audit

The large adapter is expected from `modules_to_save=["embed_tokens","lm_head"]`, not from LoRA A/B alone.

| item | value |
|---|---:|
| total model parameters | {total:,} |
| PEFT trainable parameters | {trainable:,} |
| LoRA A/B parameters | {lora:,} |
| embed_tokens parameters seen in PEFT names | {rep["embed_tokens_parameters"]:,} |
| lm_head parameters seen in PEFT names | {rep["lm_head_parameters"]:,} |
| modules_to_save parameters | {rep["modules_to_save_parameters"]:,} |
| embed/lm_head tied at loaded model accessors | {tied} |
| adapter checkpoint size bytes | {ck_size} |

Interpretation: full embedding/output-head saves dominate the parameter count. This matches the teacher concern that output layer must be trainable for THINK-token NTP supervision.
"""
    (REPORT / "adapter_parameter_audit.md").write_text(md)
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    write_status("adapter_audit", "complete", peft_trainable_parameters=trainable, lora_ab_parameters=lora)


def build_debug32(max_q=256, max_latent=512, max_answer=256, max_seq=1024):
    rows = get_rows()
    selected, rejects = [], []
    for r in rows:
        reasons = []
        if r["q_token_count"] > max_q: reasons.append("question_too_long")
        if r["raw_K"] > max_latent: reasons.append("raw_K_exceeds_max_latent")
        if r["answer_token_count"] > max_answer: reasons.append("answer_too_long")
        if r["q_token_count"] + r["raw_K"] + r["answer_token_count"] > max_seq: reasons.append("sequence_too_long")
        if reasons:
            rejects.append(dict(sample_id=r["sample_id"], reasons=reasons, q_token_count=r["q_token_count"], cot_token_count=r["cot_token_count"], answer_token_count=r["answer_token_count"], raw_K=r["raw_K"]))
            continue
        selected.append(r)
        if len(selected) == 32:
            break
    if len(selected) < 32:
        write_json(REPORT / "debug32_length_audit.json", dict(status="failed", selected_count=len(selected), max_q=max_q, max_latent=max_latent, max_answer=max_answer, max_seq=max_seq, rejects=rejects[:200]))
        raise SystemExit("not enough no-truncation debug32 samples")
    split_hash = hashlib.sha256("\n".join(r["sample_id"] for r in selected).encode()).hexdigest()
    manifest = dict(seed=42, selection="first deterministic rows satisfying no-truncation strict 0.5N constraints", max_q=max_q, max_latent=max_latent, max_answer=max_answer, max_seq=max_seq, split_hash=split_hash, samples=selected)
    write_json(MANIFEST, manifest)
    audit = dict(status="passed", selected_count=32, split_hash=split_hash, max_q=max_q, max_latent=max_latent, max_answer=max_answer, max_seq=max_seq, latent_cap_hit=False, question_truncated=False, answer_truncated=False, sequence_truncated=False, selected_lengths=[{k:r[k] for k in ["sample_id", "q_token_count", "cot_token_count", "answer_token_count", "raw_K"]} for r in selected], rejected_prefix=rejects[:80])
    write_json(REPORT / "debug32_length_audit.json", audit)
    write_status("debug32", "complete", split_hash=split_hash)


def eval_modes(torch, tok, model, rows, step, fixed_k=64, max_new_tokens=128):
    raw_cases = []
    by_mode = {}
    for mode in ["forced_k", "fixed_k", "free_k"]:
        cases = [generate_one(torch, tok, model, r, mode, fixed_k=fixed_k, max_new_tokens=max_new_tokens) for r in rows]
        for c in cases:
            c["failure_class"] = "correct" if c["correct"] else classify_failure(c["prompt_template"], c["raw_generation"], c["parsed_answer"], c["gold_answer"], c["used_K"] or 0, max_new_tokens)
            c["step"] = step
        raw_cases.extend(cases)
        s = summarize_generation(cases)
        s["format_compliance"] = s["valid_answer_rate"] if mode != "free_k" else sum(c["parsed_answer"] is not None for c in cases) / max(len(cases), 1)
        s["failure_counts"] = dict(Counter(c["failure_class"] for c in cases))
        by_mode[mode] = s
    return by_mode, raw_cases


def run_o0():
    rows = load_debug32_or_first32()
    cfg = dict(max_q=256, max_latent=512, max_answer=256, max_seq=1024, strict=True, max_steps=500, lr=2e-5, fixed_k=64, eval_every=50, seed=42)
    random.seed(cfg["seed"])
    torch, tok, model = load_model(None, train=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
    evals, logs, all_cases = [], [], []
    passed = False
    for step in range(1, cfg["max_steps"] + 1):
        row = rows[(step - 1) % len(rows)]
        batch, segs, metas = build_stage1_batch(torch, tok, [row], max_q=cfg["max_q"], max_latent=cfg["max_latent"], max_answer=cfg["max_answer"], max_seq=cfg["max_seq"], strict=True)
        out = model(**batch)
        loss = out.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == 1:
            acc = token_acc(torch, out.logits, batch["labels"], segs)
            logs.append(dict(step=step, loss=float(loss.detach().cpu()), **acc))
        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            model.eval()
            by_mode, raw_cases = eval_modes(torch, tok, model, rows, step, fixed_k=cfg["fixed_k"], max_new_tokens=16)
            model.train()
            evals.append(dict(step=step, modes=by_mode))
            for c in raw_cases:
                all_cases.append(c)
            save_dir = REPO / f"checkpoints/stage1_repair_o0/step{step}"
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            if logs[-1].get("think_token_accuracy", 0) >= 0.99 and logs[-1].get("answer_token_accuracy", 0) >= 0.99 and by_mode["forced_k"]["answer_accuracy"] >= 0.95 and by_mode["forced_k"]["valid_answer_rate"] >= 0.95 and by_mode["fixed_k"]["valid_answer_rate"] >= 0.95 and by_mode["fixed_k"]["format_compliance"] >= 0.95:
                passed = True
                break
    final_dir = REPO / "checkpoints/stage1_repair_o0/final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir); tok.save_pretrained(REPO / "checkpoints/stage1_repair_o0/tokenizer")
    write_jsonl(REPORT / "o0_generation_cases.jsonl", all_cases)
    rep = dict(experiment="O0_token_weighted", config=cfg, status="pass" if passed else "fail", steps=step, final_checkpoint=str(final_dir), train_logs=logs, generation_evals=evals, pass_criteria=dict(teacher_forcing_think_token_accuracy=">=0.99", teacher_forcing_answer_token_accuracy=">=0.99", forced_k_valid_answer_rate=">=0.95", forced_k_answer_accuracy=">=0.95", fixed_k_valid_answer_rate=">=0.95", fixed_k_format_compliance=">=0.95"), decision="Gate H can recommend 10k only if O0 passes all criteria; if Forced-K fails, run O1/O2 later under explicit instruction.")
    write_json(REPORT / "o0_overfit_debug32.json", rep)
    write_status("o0_overfit", "complete" if passed else "failed", passed=passed, steps=step)
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


def final_report():
    statuses = {p.stem: json.loads(p.read_text()) for p in STATUS.glob("*.json")}
    report = ["# Qwen-7B Stage-1 Repair Report", "", "This round repairs Stage-1 validity only. No Stage2, 10k pilot, ratio sweep, Loss2, Model B, or long training was launched.", "", "## Status", ""]
    for k in sorted(statuses):
        report.append(f"- {k}: {statuses[k].get('status')} {statuses[k].get('passed', '')}")
    report += ["", "## Decision", ""]
    o0 = statuses.get("o0_overfit", {})
    dbg = statuses.get("debug32", {})
    if o0.get("status") == "complete" and o0.get("passed"):
        report.append("Gate H conditions are partially satisfied by O0; verify reload reproducibility before recommending 10k.")
    else:
        report.append("Do not resume 10k pilot yet. O0 did not pass, or has not run. If Forced-K failed, next explicit run should be O1/O2 balanced/answer-heavy controls.")
    report.append("")
    report.append(f"debug32 split hash: `{dbg.get('split_hash', 'missing')}`")
    (REPORT / "stage1_repair_final_report.md").write_text("\n".join(report) + "\n")
    write_json(REPORT / "stage1_repair_final_status.json", statuses)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    REPORT.mkdir(parents=True, exist_ok=True); STATUS.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["old_forensics", "adapter_audit", "debug32", "o0", "final", "all"])
    args = ap.parse_args()
    if args.phase == "old_forensics": old_forensics()
    elif args.phase == "adapter_audit": adapter_audit()
    elif args.phase == "debug32": build_debug32()
    elif args.phase == "o0": run_o0()
    elif args.phase == "final": final_report()
    elif args.phase == "all":
        try:
            old_forensics(); adapter_audit(); build_debug32(); run_o0()
        finally:
            final_report()

if __name__ == "__main__":
    main()
