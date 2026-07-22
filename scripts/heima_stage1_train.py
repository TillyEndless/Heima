#!/usr/bin/env python3
"""Stage1 entrypoint that delegates to the official Heima trainer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_TRAINER = ROOT / "heima/main_python/2-training-pipeline-main_lora-pure_llm_decoder_lora-split_3_stages-fix_num-progressive.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Official Heima Stage1 torchtune YAML.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nproc-per-node", default="1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(args.nproc_per_node),
        str(OFFICIAL_TRAINER),
        "--config",
        args.config,
        "freeze_base_model=True",
        f"output_dir={out}",
    ] + list(args.overrides)
    manifest = {
        "stage": "stage1_interpreter_learning",
        "official_trainer": str(OFFICIAL_TRAINER),
        "command": command,
        "invariants": [
            "Model A frozen via freeze_base_model=True",
            "official dataset and dataset_decoder config",
            "official CEWithChunkedOutputLoss",
            "official shifted thinking hidden extraction",
            "official decoder embedding replacement and projector",
            "official checkpoint format",
        ],
    }
    (out / "stage1_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.dry_run:
        print(" ".join(command))
        return 0
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
