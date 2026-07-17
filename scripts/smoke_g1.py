#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.g1.trainer import train


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = train(
        args.config,
        overrides={
            "max_steps": 2,
            "log_interval": 1,
            "eval_samples": 4,
            "report_path": args.report,
            "checkpoint_dir": str(Path(args.report).with_suffix("")) + "_checkpoint",
            "sample_decode_path": str(Path(args.report).with_suffix(".samples.txt")),
        },
    )
    summary = {
        "status": report["status"],
        "experiment": report["experiment"],
        "stop_gates_triggered": report["stop_gates_triggered"],
        "warnings": report["warnings"],
        "first_log": report["logs"][0] if report["logs"] else None,
        "last_log": report["logs"][-1] if report["logs"] else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not report["stop_gates_triggered"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
