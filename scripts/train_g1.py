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

from src.g1.trainer import train


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = train(args.config)
    print(json.dumps({k: report[k] for k in ("status", "experiment", "eval", "warnings")}, indent=2))
    return 0 if report["stop_gates_triggered"] == [] else 1


if __name__ == "__main__":
    raise SystemExit(main())
