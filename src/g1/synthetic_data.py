from __future__ import annotations

import json
import random
from pathlib import Path


GENERATOR_VERSION = "synthetic_arithmetic_g1_v1"


def make_record(index: int, split: str, a: int, b: int, c: int) -> dict:
    s = a + b
    y = s * c
    steps = [
        f"First add {a} and {b}: {a} + {b} = {s}.",
        f"Then multiply the result by {c}: {s} * {c} = {y}.",
    ]
    return {
        "id": f"g1_{split}_{index:04d}",
        "source": GENERATOR_VERSION,
        "split": split,
        "question": f"Compute ({a} + {b}) * {c}.",
        "cot": " ".join(steps),
        "answer": str(y),
        "steps": steps,
        "metadata": {
            "a": a,
            "b": b,
            "c": c,
            "sum": s,
            "generator_version": GENERATOR_VERSION,
        },
    }


def validate_record(record: dict) -> bool:
    meta = record["metadata"]
    a, b, c = meta["a"], meta["b"], meta["c"]
    s = a + b
    y = s * c
    expected_steps = [
        f"First add {a} and {b}: {a} + {b} = {s}.",
        f"Then multiply the result by {c}: {s} * {c} = {y}.",
    ]
    return (
        record["question"] == f"Compute ({a} + {b}) * {c}."
        and record["steps"] == expected_steps
        and record["cot"] == " ".join(expected_steps)
        and record["answer"] == str(y)
    )


def generate_synthetic_split(
    train_size: int = 64,
    validation_size: int = 32,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    combos = [(a, b, c) for a in range(2, 30) for b in range(2, 30) for c in range(2, 12)]
    rng.shuffle(combos)
    needed = train_size + validation_size
    selected = combos[:needed]
    train = [make_record(i, "train", *combo) for i, combo in enumerate(selected[:train_size])]
    val = [
        make_record(i, "validation", *combo)
        for i, combo in enumerate(selected[train_size:needed])
    ]
    return train, val


def write_jsonl(records: list[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

