#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.htext.trainer import train_h0, train_h1, train_joint_main_l1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["h0", "h1", "joint"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--h0-checkpoint-dir")
    args = parser.parse_args()
    checkpoint_dir = str(Path(args.report).with_suffix("")) + "_checkpoint"
    overrides = {
        "max_steps": 2,
        "log_interval": 1,
        "eval_samples": 4,
        "report_path": args.report,
        "checkpoint_dir": checkpoint_dir,
    }
    if args.h0_checkpoint_dir:
        overrides["h0_checkpoint_dir"] = args.h0_checkpoint_dir
    if args.stage == "h0":
        report = train_h0(args.config, overrides)
    elif args.stage == "h1":
        report = train_h1(args.config, overrides)
    else:
        report = train_joint_main_l1(args.config, overrides)
    summary = {
        "status": report["status"],
        "stage": report["stage"],
        "stop_gates_triggered": report["stop_gates_triggered"],
        "first_log": report["logs"][0] if report["logs"] else None,
        "last_log": report["logs"][-1] if report["logs"] else None,
        "eval": report["eval"],
        "checkpoint_dir": checkpoint_dir,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not report["stop_gates_triggered"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
