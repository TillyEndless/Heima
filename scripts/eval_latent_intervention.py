#!/usr/bin/env python3
"""Evaluate saved GPT2 latent ablation checkpoints with four-way intervention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from run_gpt2_latent_ablation import Example, evaluate_intervention, load_hf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--group", choices=["G0", "G1", "G2", "G3"], required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--model-path", default="/data/zxl/models/openai-community-gpt2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer = load_hf(args.model_path, device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    records = [Example(**x) for x in split["eval"]]
    metrics = evaluate_intervention(model, tokenizer, records, args.group, args.max_length, device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
