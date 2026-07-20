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
    args = parser.parse_args()
    if args.stage == "h0":
        report = train_h0(args.config)
    elif args.stage == "h1":
        report = train_h1(args.config)
    else:
        report = train_joint_main_l1(args.config)
    print(json.dumps({k: report[k] for k in ("status", "stage", "experiment", "eval")}, indent=2))
    return 0 if not report["stop_gates_triggered"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
