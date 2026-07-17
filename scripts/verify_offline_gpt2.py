#!/usr/bin/env python3
import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    model_path = Path("/mnt/nas/share2/home/zxl/models/openai-community-gpt2")
    required = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "merges.txt",
        "vocab.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing = [name for name in required if not (model_path / name).exists()]
    report = {
        "model_path": str(model_path),
        "required_files": required,
        "missing_files": missing,
        "offline_env": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
    }
    status = "pass"
    warning_text = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path), local_files_only=True, use_safetensors=True
            )
            ids = tokenizer("Compute (2 + 3) * 4.", return_tensors="pt")["input_ids"]
            out = model(input_ids=ids, use_cache=False)
            warning_text = [str(w.message) for w in caught]
        logits_finite = torch.isfinite(out.logits).all().item()
        report.update(
            {
                "tokenizer_loaded": True,
                "model_loaded": True,
                "n_embd": model.config.n_embd,
                "n_layer": model.config.n_layer,
                "vocab_size": model.config.vocab_size,
                "logits_finite": bool(logits_finite),
                "warnings": warning_text,
                "network_requests": "blocked_by_offline_env",
            }
        )
        bad_warning = any(
            "newly initialized" in text.lower() or "missing" in text.lower()
            for text in warning_text
        )
        if (
            missing
            or model.config.n_embd != 768
            or model.config.n_layer != 12
            or model.config.vocab_size != 50257
            or not logits_finite
            or bad_warning
        ):
            status = "fail"
    except Exception as exc:
        status = "fail"
        report["error"] = repr(exc)
        report["warnings"] = warning_text
    report["status"] = status
    out_path = Path("experiments/g1_gpt2/reports/offline_model_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
