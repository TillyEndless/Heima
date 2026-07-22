#!/usr/bin/env python3
"""Strict Heima Stage2 comparison entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.heima_stage2 import Stage2Mode


OFFICIAL_TRAINER = ROOT / "heima/main_python/2-training-pipeline-main_lora-pure_llm_decoder_lora-split_3_stages-fix_num-progressive.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-checkpoint", required=True, help="B* interpreter checkpoint root from Stage1.")
    parser.add_argument("--config", required=True, help="Official Heima Stage2 torchtune YAML.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage2-mode", choices=[m.value for m in Stage2Mode], required=True)
    parser.add_argument("--lambda-interp", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mode = Stage2Mode(args.stage2_mode)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "stage2_compare",
        "mode": mode.value,
        "official_trainer_reference": str(OFFICIAL_TRAINER),
        "stage1_checkpoint_B_star": args.stage1_checkpoint,
        "config": args.config,
        "lambda_interp": args.lambda_interp,
        "model_A": "trainable",
        "model_B_teacher": "frozen, loaded from Stage1 B*",
        "optimizer_contains_B": False,
        "loss_contract": {
            "heima_baseline": "L_total = L_NTP; L_interp is forward/log/eval only and receives detached z.",
            "ours_interp_supervision": "L_total = L_NTP + lambda_interp * L_interp; B frozen but z is not detached, so L_interp updates A.",
        },
        "only_algorithmic_difference_vs_pair": "whether z is detached before the frozen interpreter loss",
        "required_audits": [
            "teacher B requires_grad is false",
            "optimizer excludes B parameters",
            "grad_A_from_interp == 0 for heima_baseline",
            "grad_A_from_interp > 0 for ours_interp_supervision",
            "answer accuracy, NTP loss, interpreter loss",
            "B(z) generation quality and correct/shuffle/zero latent interventions",
        ],
    }
    (out / f"stage2_{mode.value}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    raise SystemExit(
        "Stage2 full training adapter is intentionally gated behind dry-run until an official config/checkpoint path is supplied and validated."
    )


if __name__ == "__main__":
    raise SystemExit(main())
