#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

THINK = "<THINK>"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATASET_ID = "a-m-team/AM-DeepSeek-R1-Distilled-1.4M"
ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?)")
NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def append_status(out: Path, name: str, status: str, **extra: Any) -> None:
    write_json(out / "status" / f"{name}.json", {"stage": name, "status": status, "time": time.time(), **extra})


def status_done(out: Path, name: str) -> bool:
    p = out / "status" / f"{name}.json"
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("status") == "complete"
    except Exception:
        return False


def run(cmd: list[str]) -> dict[str, Any]:
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-4000:], "stderr": p.stderr[-4000:]}


def parse_answer(text: str) -> str | None:
    text = str(text)
    m = ANSWER_RE.search(text)
    if m:
        return normalize_number(m.group(1))
    lowered = text.lower()
    for marker in ["final answer is", "answer is", "answer:", "<answer>"]:
        idx = lowered.rfind(marker)
        if idx >= 0:
            nums = NUM_RE.findall(text[idx:])
            if nums:
                return normalize_number(nums[-1])
    nums = NUM_RE.findall(text)
    if not nums:
        return None
    return normalize_number(nums[-1])


def normalize_number(s: str) -> str:
    s = str(s).strip().replace(",", "")
    if s.endswith(".0"):
        s = s[:-2]
    return s


def run_parser_tests(out: Path) -> bool:
    cases = [
        ("#### 42", "42"),
        ("answer is -7", "-7"),
        ("final answer is 3.5 meters", "3.5"),
        ("#### 1/2", "1/2"),
        ("The answer is 1,234 dollars.", "1234"),
        ("numbers 1 2 3 final answer is 4", "4"),
        ("no answer here", None),
        ("<answer> -12.50 </answer>", "-12.50"),
    ]
    rows = []
    ok = True
    for text, exp in cases:
        got = parse_answer(text)
        rows.append({"text": text, "expected": exp, "got": got, "ok": got == exp})
        ok = ok and got == exp
    write_json(out / "reports" / "answer_parser_tests.json", {"ok": ok, "cases": rows})
    return ok


def environment(out: Path, hf_home: Path) -> None:
    import importlib

    mods: dict[str, str] = {}
    for m in ["torch", "transformers", "peft", "datasets", "accelerate", "huggingface_hub"]:
        try:
            mod = importlib.import_module(m)
            mods[m] = getattr(mod, "__version__", "unknown")
        except Exception as e:
            mods[m] = f"MISSING: {e!r}"
    cuda_available = False
    torch_cuda = None
    device_count = 0
    if torch is not None:
        cuda_available = torch.cuda.is_available()
        torch_cuda = getattr(torch.version, "cuda", None)
        device_count = torch.cuda.device_count()
    info = {
        "python": sys.version,
        "executable": sys.executable,
        "modules": mods,
        "cuda_available": cuda_available,
        "torch_cuda": torch_cuda,
        "device_count": device_count,
        "gpu": run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"]),
        "disk": run(["df", "-h", str(out), str(hf_home)]),
        "git": run(["git", "rev-parse", "HEAD"]),
        "git_status": run(["git", "status", "--short"]),
        "branch": run(["git", "branch", "--show-current"]),
        "hf_home": str(hf_home),
    }
    write_json(out / "reports" / "overnight" / "environment.json", info)


def ensure_tokenizer(model_id: str, hf_home: Path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=str(hf_home), trust_remote_code=True)
    added = tok.add_special_tokens({"additional_special_tokens": [THINK]})
    tid = tok.convert_tokens_to_ids(THINK)
    check = tok.encode(THINK, add_special_tokens=False)
    return tok, {"added": added, "think_id": tid, "think_encode": check, "single_token": len(check) == 1}


def extract_embedding_weight(model) -> torch.Tensor | None:
    base = model
    if hasattr(base, "get_base_model"):
        base = base.get_base_model()
    try:
        return base.get_input_embeddings().weight
    except Exception:
        return None


def trainable_parameter_report(out: Path, model) -> dict[str, Any]:
    rows = []
    total = 0
    trainable = 0
    embed_trainable = False
    lm_head_trainable = False
    opt_names = []
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            opt_names.append(name)
        if "embed_tokens" in name and p.requires_grad:
            embed_trainable = True
        if "lm_head" in name and p.requires_grad:
            lm_head_trainable = True
        if p.requires_grad or "embed_tokens" in name or "lm_head" in name:
            rows.append({"name": name, "shape": list(p.shape), "numel": n, "requires_grad": bool(p.requires_grad)})
    report = {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_fraction": trainable / max(total, 1),
        "embed_tokens_trainable_or_modules_to_save": embed_trainable,
        "lm_head_trainable_or_modules_to_save": lm_head_trainable,
        "trainable_head": rows[:300],
        "optimizer_membership_names_head": opt_names[:300],
    }
    write_json(out / "reports" / "overnight" / "trainable_parameter_report.json", report)
    return report


def nested_get(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def parse_record(raw: dict[str, Any], idx: int) -> dict[str, Any]:
    info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
    question = raw.get("question") or raw.get("prompt") or raw.get("input") or info.get("question")
    cot = nested_get(raw, "info.think_content") or info.get("think_content") or raw.get("think_content") or raw.get("cot")
    answer = nested_get(raw, "info.answer_content") or info.get("answer_content") or raw.get("answer_content")
    content = raw.get("content") or raw.get("messages") or raw.get("assistant") or raw.get("response")
    parse_status = "direct"
    if isinstance(content, list):
        content = "\n".join(str(x.get("content", x)) if isinstance(x, dict) else str(x) for x in content)
    if (not cot or not answer) and content:
        text = str(content)
        tm = re.search(r"<think>(.*?)</think>", text, flags=re.S | re.I)
        am = re.search(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
        if not cot and tm:
            cot = tm.group(1).strip()
            parse_status = "assistant_tags"
        if not answer and am:
            answer = am.group(1).strip()
            parse_status = "assistant_tags"
    if not answer and content:
        answer = parse_answer(str(content))
        parse_status = "answer_parser"
    if not question and raw.get("messages"):
        msgs = raw["messages"]
        if isinstance(msgs, list):
            user_parts = [m.get("content", "") for m in msgs if isinstance(m, dict) and m.get("role") in {"user", "human"}]
            question = "\n".join(map(str, user_parts))
    return {
        "sample_id": str(raw.get("id", raw.get("sample_id", idx))),
        "source": raw.get("source", DATASET_ID),
        "question": str(question or "").strip(),
        "gold_cot": str(cot or "").strip(),
        "answer": str(answer or "").strip(),
        "reference_answer": str(answer or "").strip(),
        "parse_status": parse_status,
    }


def dataset_audit(out: Path, hf_home: Path, model_id: str, dataset_id: str, limit: int = 1000, config_name: str = "am_0.9M_sample_1k") -> list[dict[str, Any]]:
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    tok, tok_report = ensure_tokenizer(model_id, hf_home)
    files = []
    try:
        files = HfApi().list_repo_files(dataset_id, repo_type="dataset")
    except Exception as e:
        files = [f"LIST_FAILED: {e!r}"]
    rows: list[dict[str, Any]] = []
    raw_seen = 0
    errors = []
    try:
        ds = load_dataset(dataset_id, config_name, split="train", streaming=True, cache_dir=str(hf_home))
        for raw in ds.take(limit):
            raw_seen += 1
            rec = parse_record(raw, raw_seen - 1)
            rec["cot_token_count"] = len(tok(rec["gold_cot"], add_special_tokens=False)["input_ids"])
            rec["latent_ratio"] = 0.5
            rec["latent_count"] = max(1, round(0.5 * rec["cot_token_count"])) if rec["cot_token_count"] else 0
            rows.append(rec)
    except Exception as e:
        errors.append(repr(e))
        try:
            filename = f"{config_name}.jsonl"
            local = hf_hub_download(repo_id=dataset_id, filename=filename, repo_type="dataset", cache_dir=str(hf_home))
            with open(local, encoding="utf-8") as f:
                for line in f:
                    if raw_seen >= limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    raw_seen += 1
                    raw = json.loads(line)
                    rec = parse_record(raw, raw_seen - 1)
                    rec["cot_token_count"] = len(tok(rec["gold_cot"], add_special_tokens=False)["input_ids"])
                    rec["latent_ratio"] = 0.5
                    rec["latent_count"] = max(1, round(0.5 * rec["cot_token_count"])) if rec["cot_token_count"] else 0
                    rec["parse_status"] = f"jsonl_fallback:{rec['parse_status']}"
                    rows.append(rec)
        except Exception as e2:
            errors.append(f"jsonl_fallback_failed: {e2!r}")
    valid = [r for r in rows if r["question"] and r["gold_cot"] and r["answer"] and r["latent_count"] > 0]
    audit = {
        "dataset_id": dataset_id,
        "config_name": config_name,
        "repo_files_head": files[:200],
        "raw_seen": raw_seen,
        "parsed": len(rows),
        "valid": len(valid),
        "invalid": len(rows) - len(valid),
        "tokenizer": tok_report,
        "errors": errors,
        "parse_status_counts": count_by(rows, "parse_status"),
        "cot_token_count": stats([r["cot_token_count"] for r in valid]),
        "latent_count": stats([r["latent_count"] for r in valid]),
    }
    write_json(out / "reports" / "dataset_audit.json", audit)
    md = ["# Dataset Samples", ""]
    for r in valid[:10]:
        md += [f"## {r['sample_id']}", f"Question: {r['question'][:500]}", f"CoT: {r['gold_cot'][:500]}", f"Answer: {r['answer']}", ""]
    (out / "reports" / "samples.md").write_text("\n".join(md), encoding="utf-8")
    return valid


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    d: dict[str, int] = {}
    for r in rows:
        d[str(r.get(key))] = d.get(str(r.get(key)), 0) + 1
    return d


def stats(xs: list[int]) -> dict[str, float | int | None]:
    if not xs:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {"count": len(xs), "mean": sum(xs) / len(xs), "min": min(xs), "max": max(xs)}


def write_splits(out: Path, rows: list[dict[str, Any]]) -> None:
    proc = out / "data" / "processed_large" / "qwen7b_stage1"
    proc.mkdir(parents=True, exist_ok=True)
    write_json(proc / "overfit32.json", rows[:32])
    write_json(proc / "smoke1k.json", rows[:1000])
    report = {
        "overfit32": str(proc / "overfit32.json"),
        "smoke1k": str(proc / "smoke1k.json"),
        "counts": {"overfit32": min(32, len(rows)), "smoke1k": min(1000, len(rows))},
        "dynamic_latent_length": {
            "current": "oracle-K",
            "definition": "K=max(1, round(0.5 * gold CoT token count)); diagnostic/training smoke only, not leakage-free formal inference.",
            "stage1_smoke_runtime_cap": "collator uses min(K, 128) with max_q=256 and max_answer=128 to fit Qwen-7B on one 40G A100.",
            "fixed-K": "not used tonight",
            "free-K": "not used tonight",
        },
    }
    write_json(out / "reports" / "dynamic_latent_length_report.json", report)


@dataclasses.dataclass
class Batch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    segments: list[list[str]]


def make_stage1_batch(tok, rows: list[dict[str, Any]], max_q: int, max_latent: int, max_answer: int, device) -> Batch:
    ids_rows, lab_rows, seg_rows = [], [], []
    think_id = tok.convert_tokens_to_ids(THINK)
    for r0 in rows:
        qids = tok(f"Question:\n{r0['question']}\n\n", add_special_tokens=False)["input_ids"][:max_q]
        ans = tok("\nAnswer:\n" + r0["answer"] + tok.eos_token, add_special_tokens=False)["input_ids"][:max_answer]
        k = min(int(r0["latent_count"]), max_latent)
        ids = qids + [think_id] * k + ans
        labels = [-100] * len(qids) + [think_id] * k + ans
        seg = ["question"] * len(qids) + ["think"] * k + ["answer"] * len(ans)
        ids_rows.append(ids)
        lab_rows.append(labels)
        seg_rows.append(seg)
    mx = max(len(x) for x in ids_rows)
    input_ids = torch.full((len(rows), mx), tok.pad_token_id or tok.eos_token_id, dtype=torch.long, device=device)
    labels = torch.full((len(rows), mx), -100, dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    for i, ids in enumerate(ids_rows):
        input_ids[i, : len(ids)] = torch.tensor(ids, device=device)
        labels[i, : len(ids)] = torch.tensor(lab_rows[i], device=device)
        attention[i, : len(ids)] = 1
        seg_rows[i] += ["pad"] * (mx - len(ids))
    return Batch(input_ids, labels, attention, seg_rows)


def split_loss(logits: torch.Tensor, labels: torch.Tensor, segments: list[list[str]]) -> dict[str, Any]:
    if torch is None or F is None:
        raise RuntimeError("torch is required for split_loss")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    seg_shift = [s[1:] for s in segments]
    out: dict[str, Any] = {}
    for seg_name in ["think", "answer"]:
        mask_vals = []
        for i, segs in enumerate(seg_shift):
            for j, seg in enumerate(segs):
                if seg == seg_name and int(shift_labels[i, j]) != -100:
                    mask_vals.append(flat_loss[i, j])
        if mask_vals:
            vals = torch.stack(mask_vals)
            out[f"loss_{seg_name}"] = float(vals.mean().detach().cpu())
            out[f"{seg_name}_token_count"] = int(vals.numel())
        else:
            out[f"loss_{seg_name}"] = None
            out[f"{seg_name}_token_count"] = 0
    return out


def token_audit(out: Path, tok, batch: Batch) -> None:
    rows = []
    for i in range(min(2, batch.input_ids.size(0))):
        for j in range(batch.input_ids.size(1)):
            tid = int(batch.input_ids[i, j].cpu())
            lab = int(batch.labels[i, j].cpu())
            rows.append({
                "sample": i,
                "position": j,
                "token_id": tid,
                "token": tok.decode([tid]),
                "segment": batch.segments[i][j],
                "label": lab,
                "contributes_to_loss": lab != -100,
            })
    think_id = tok.convert_tokens_to_ids(THINK)
    checks = {
        "question_labels_all_ignore": all(r["label"] == -100 for r in rows if r["segment"] == "question"),
        "think_labels_are_think_id": all(r["label"] == think_id for r in rows if r["segment"] == "think"),
        "think_id": think_id,
    }
    write_json(out / "reports" / "overnight" / "token_level_audit.json", {"checks": checks, "rows": rows})


def download_model(out: Path, hf_home: Path, model_id: str) -> bool:
    from huggingface_hub import HfApi, snapshot_download
    before = shutil.disk_usage(hf_home)
    if before.free < 80 * 1024**3:
        write_json(out / "reports" / "model_download.json", {"status": "blocked_low_disk", "free_bytes": before.free})
        return False
    api = HfApi()
    info = api.model_info(model_id)
    path = snapshot_download(repo_id=model_id, cache_dir=str(hf_home), resume_download=True)
    files = list(Path(path).glob("*"))
    report = {"status": "complete", "model_id": model_id, "sha": info.sha, "local_path": path, "files": [p.name for p in files], "disk_free_before": before.free}
    write_json(out / "reports" / "model_download.json", report)
    return True


def qwen_baseline_sanity(out: Path, hf_home: Path, rows_path: Path, model_id: str, n: int = 16) -> bool:
    if torch is None:
        write_json(out / "reports" / "qwen_baseline_sanity_summary.json", {"status": "skipped", "reason": "torch_missing"})
        return False
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda:0"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))[:n]
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=str(hf_home), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=str(hf_home),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    cases = []
    t0 = time.time()
    total_new = 0
    for r in rows:
        prompt = f"Question:\n{r['question']}\n\nPlease reason step by step and give the final answer."
        if hasattr(tok, "apply_chat_template") and tok.chat_template:
            messages = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt + "\nAnswer:"
        enc = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        new_ids = gen[0, enc.input_ids.size(1):]
        total_new += int(new_ids.numel())
        raw = tok.decode(new_ids, skip_special_tokens=True)
        cases.append({
            "sample_id": r["sample_id"],
            "question": r["question"],
            "gold_answer": r["answer"],
            "raw_output": raw,
            "parsed_answer": parse_answer(raw),
            "stop_reason": "eos" if len(new_ids) and int(new_ids[-1]) == tok.eos_token_id else "max_new_tokens_or_other",
        })
    elapsed = time.time() - t0
    acc = sum(1 for c in cases if c["parsed_answer"] == normalize_number(c["gold_answer"])) / max(len(cases), 1)
    report = {
        "model_id": model_id,
        "num_cases": len(cases),
        "answer_accuracy": acc,
        "tokens_per_second": total_new / max(elapsed, 1e-6),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "cases_path": str(out / "reports" / "qwen_baseline_sanity.jsonl"),
    }
    p = out / "reports" / "qwen_baseline_sanity.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    write_json(out / "reports" / "qwen_baseline_sanity_summary.json", report)
    return True


def load_qwen(model_id: str, hf_home: Path, device: str, lora: bool):
    if torch is None:
        raise RuntimeError("torch is required for Qwen loading")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=str(hf_home), trust_remote_code=True)
    tok.add_special_tokens({"additional_special_tokens": [THINK]})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=str(hf_home),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tok))
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if lora:
        cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            modules_to_save=["embed_tokens", "lm_head"],
        )
        model = get_peft_model(model, cfg)
    model.to(device)
    return tok, model


def train_stage1(out: Path, hf_home: Path, rows_path: Path, max_steps: int, run_name: str) -> bool:
    if torch is None:
        write_json(out / "reports" / f"{run_name}_stage1_smoke.json", {"ok": False, "reason": "torch_missing"})
        return False
    device = "cuda:0"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    tok, model = load_qwen(MODEL_ID, hf_home, device, lora=True)
    trainable_parameter_report(out, model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
    logs = []
    think_id = tok.convert_tokens_to_ids(THINK)
    ok = True
    for step in range(1, max_steps + 1):
        batch_rows = [rows[(step - 1) % len(rows)]]
        batch = make_stage1_batch(tok, batch_rows, 256, 128, 128, device)
        outm = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels)
        loss = outm.loss
        if not torch.isfinite(loss):
            ok = False
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            pred = outm.logits[:, :-1].argmax(-1)
            lab = batch.labels[:, 1:]
            m = lab.eq(think_id)
            think_acc = float(pred[m].eq(think_id).float().mean().detach().cpu()) if m.any() else 0.0
        log_row = {"step": step, "loss_total": float(loss.detach().cpu()), "think_token_accuracy": think_acc}
        log_row.update(split_loss(outm.logits.detach(), batch.labels, batch.segments))
        log_row["loss_self_decode"] = None
        log_row["cot_target_token_count"] = 0
        logs.append(log_row)
        if step == 1:
            token_audit(out, tok, batch)
    run_dir = out / "checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(run_dir / "final_adapter")
    tok.save_pretrained(run_dir / "tokenizer")
    reload_ok = False
    try:
        from peft import PeftModel
        base_tok, base = load_qwen(MODEL_ID, hf_home, device, lora=False)
        base.resize_token_embeddings(len(tok))
        reloaded = PeftModel.from_pretrained(base, run_dir / "final_adapter").to(device)
        emb = extract_embedding_weight(reloaded)
        reload_ok = emb is not None and emb.size(0) == len(tok)
        del reloaded, base, base_tok
    except Exception as e:
        write_json(out / "reports" / f"{run_name}_reload_error.json", {"error": repr(e)})
    write_json(out / "reports" / f"{run_name}_stage1_smoke.json", {
        "ok": ok,
        "reload_ok": reload_ok,
        "steps": len(logs),
        "logs": logs,
        "checkpoint": str(run_dir / "final_adapter"),
    })
    return ok and (len(logs) >= max_steps)


def write_status_markdown(out: Path) -> None:
    status_dir = out / "status"
    reports = out / "reports"
    lines = ["# Overnight Status", ""]
    lines += ["## Branch And Git", ""]
    lines += [f"- branch: `{run(['git', 'branch', '--show-current'])['stdout'].strip()}`"]
    lines += [f"- head: `{run(['git', 'rev-parse', '--short', 'HEAD'])['stdout'].strip()}`", ""]
    lines += ["## Stage Status", ""]
    if status_dir.exists():
        for p in sorted(status_dir.glob("*.json")):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                lines.append(f"- {p.stem}: {obj.get('status')}")
            except Exception:
                lines.append(f"- {p.stem}: unreadable")
    lines += ["", "## Key Artifacts", ""]
    for rel in [
        "overnight/environment.json",
        "answer_parser_tests.json",
        "dataset_audit.json",
        "dynamic_latent_length_report.json",
        "model_download.json",
        "qwen_baseline_sanity_summary.json",
        "qwen7b_stage1_overfit32_stage1_smoke.json",
        "qwen7b_stage1_smoke1k_200_stage1_smoke.json",
    ]:
        p = reports / rel
        lines.append(f"- `{p}`: {'exists' if p.exists() else 'missing'}")
    lines += ["", "## Notes", ""]
    lines += [
        "- 7B run is Stage-1 only: `L_main = L_think + L_answer`.",
        "- No Model B, Loss2, projector, full-data training, or Stage2 is launched by this script.",
        "- Checkpoints and HF cache live outside git-tracked artifacts.",
    ]
    (reports / "overnight_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data2/zhouxiaoling/latent_cot/Heima-overnight-qwen7b")
    ap.add_argument("--hf-home", default="/weights2/zhouxiaoling/hf_cache")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--dataset-id", default=DATASET_ID)
    ap.add_argument("--dataset-config", default="am_0.9M_sample_1k")
    ap.add_argument("--phase", choices=["all", "audit", "download", "data", "qwen-sanity", "stage1"], default="all")
    ap.add_argument("--stage1", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    hf_home = Path(args.hf_home)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    hf_home.mkdir(parents=True, exist_ok=True)
    (out / "reports" / "overnight").mkdir(parents=True, exist_ok=True)
    append_status(out, "start", "running", phase=args.phase)
    environment(out, hf_home)
    ok_parser = run_parser_tests(out)
    append_status(out, "answer_parser_tests", "complete" if ok_parser else "failed")
    downloaded = False
    if args.phase in {"all", "download"} and not status_done(out, "download_model"):
        downloaded = download_model(out, hf_home, args.model_id)
        append_status(out, "download_model", "complete" if downloaded else "failed")
    if args.phase in {"all", "data"} and not status_done(out, "dataset_audit"):
        rows = dataset_audit(out, hf_home, args.model_id, args.dataset_id, limit=1000, config_name=args.dataset_config)
        write_splits(out, rows)
        append_status(out, "dataset_audit", "complete", valid=len(rows))
    if args.phase in {"all", "qwen-sanity"} and not status_done(out, "qwen_baseline_sanity"):
        smoke = out / "data" / "processed_large" / "qwen7b_stage1" / "smoke1k.json"
        if smoke.exists():
            ok_sanity = qwen_baseline_sanity(out, hf_home, smoke, args.model_id, n=16)
            append_status(out, "qwen_baseline_sanity", "complete" if ok_sanity else "failed")
        else:
            append_status(out, "qwen_baseline_sanity", "failed", reason="missing smoke split")
    if args.phase in {"all", "stage1"} and args.stage1:
        lock_path = out / "stage1_gpu1.lock"
        with lock_path.open("w") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            overfit = out / "data" / "processed_large" / "qwen7b_stage1" / "overfit32.json"
            smoke = out / "data" / "processed_large" / "qwen7b_stage1" / "smoke1k.json"
            ok32 = train_stage1(out, hf_home, overfit, 30, "qwen7b_stage1_overfit32")
            append_status(out, "stage1_overfit32", "complete" if ok32 else "failed")
            if ok32:
                ok200 = train_stage1(out, hf_home, smoke, 200, "qwen7b_stage1_smoke1k_200")
                append_status(out, "stage1_smoke1k_200", "complete" if ok200 else "failed")
    append_status(out, "finish", "complete")
    write_status_markdown(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
