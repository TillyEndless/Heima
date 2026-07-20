#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.g1.latent_reasoner import main_forward
from src.g1.synthetic_data import read_jsonl
from src.g1.trainer import load_config, load_models
from src.g1.whole_cot_decoder import loss1_forward
from src.g1.gradient_monitor import grad_norm, tensor_norm


def run_case(config_path: str, freeze_b: bool) -> dict:
    config = load_config(config_path)
    tokenizer, model_a, model_b, _ = load_models(config)
    records = read_jsonl(config["train_path"])[:2]
    for p in model_a.parameters():
        p.requires_grad_(True)
    for p in model_b.parameters():
        p.requires_grad_(not freeze_b)
    model_a.zero_grad(set_to_none=True)
    model_b.zero_grad(set_to_none=True)
    main = main_forward(
        model_a,
        tokenizer,
        records,
        config["max_question_tokens"],
        config["max_answer_tokens"],
    )
    dec = loss1_forward(model_b, tokenizer, records, main.z, config["max_cot_tokens"])
    dec.loss.backward()
    a_norm, a_finite = grad_norm(model_a.parameters())
    b_norm, b_finite = grad_norm(model_b.parameters())
    b_grad_count = sum(1 for p in model_b.parameters() if p.grad is not None)
    return {
        "freeze_b": freeze_b,
        "loss1": dec.loss.item(),
        "z_grad_norm": tensor_norm(main.z.grad),
        "model_a_grad_norm": a_norm,
        "model_b_grad_norm": b_norm,
        "model_b_grad_count": b_grad_count,
        "model_a_grad_finite": a_finite,
        "model_b_grad_finite": b_finite,
        "pass": (
            tensor_norm(main.z.grad) > 0
            and a_norm > 0
            and a_finite
            and b_finite
            and ((freeze_b and b_grad_count == 0) or ((not freeze_b) and b_norm > 0))
        ),
    }


def main() -> int:
    config_path = "experiments/g1_gpt2/configs/main_l1.yaml"
    report = {
        "trainable_b": run_case(config_path, freeze_b=False),
        "frozen_b": run_case(config_path, freeze_b=True),
    }
    report["status"] = "pass" if report["trainable_b"]["pass"] and report["frozen_b"]["pass"] else "fail"
    out = Path("experiments/g1_gpt2/reports/pretrained_bridge_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
