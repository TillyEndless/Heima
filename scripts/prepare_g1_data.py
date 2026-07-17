#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.g1.synthetic_data import generate_synthetic_split, write_jsonl


def main() -> None:
    train, val = generate_synthetic_split(train_size=64, validation_size=32, seed=42)
    write_jsonl(train, Path("experiments/g1_gpt2/data/synthetic_g1_train.jsonl"))
    write_jsonl(val, Path("experiments/g1_gpt2/data/synthetic_g1_validation.jsonl"))
    write_jsonl(train, Path("data/synthetic_g1_train.jsonl"))
    write_jsonl(val, Path("data/synthetic_g1_validation.jsonl"))
    print({"train": len(train), "validation": len(val)})


if __name__ == "__main__":
    main()
