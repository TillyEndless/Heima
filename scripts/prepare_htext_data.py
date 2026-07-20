#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.htext.synthetic_data import generate_synthetic_split, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument("--output-dir", default="experiments/htext_gpt2/data")
    args = parser.parse_args()
    train, val = generate_synthetic_split(
        train_size=args.train_size,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    output = Path(args.output_dir)
    write_jsonl(train, output / f"synthetic_htext_train_seed_{args.seed}.jsonl")
    write_jsonl(val, output / f"synthetic_htext_validation_seed_{args.seed}.jsonl")
    if args.seed == 42:
        write_jsonl(train, output / "synthetic_htext_train.jsonl")
        write_jsonl(val, output / "synthetic_htext_validation.jsonl")
    print({"seed": args.seed, "train": len(train), "validation": len(val)})


if __name__ == "__main__":
    main()
