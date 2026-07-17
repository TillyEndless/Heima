from __future__ import annotations

import json
import random
from pathlib import Path


GENERATOR_VERSION = "synthetic_htext_v2"


ADD_MUL_TEMPLATES = [
    (
        "Compute ({a} + {b}) * {c}.",
        [
            "First add {a} and {b}: {a} + {b} = {s}.",
            "Then multiply the result by {c}: {s} * {c} = {y}.",
        ],
    ),
    (
        "Take the sum of {a} and {b}, then multiply it by {c}. What is the result?",
        [
            "The sum is {a} + {b} = {s}.",
            "Multiplying by {c} gives {s} * {c} = {y}.",
        ],
    ),
    (
        "What do you get if {c} times the quantity {a} plus {b} is evaluated?",
        [
            "Evaluate the parentheses first: {a} + {b} = {s}.",
            "Now compute {c} times that value: {c} * {s} = {y}.",
        ],
    ),
]

MUL_SUB_TEMPLATES = [
    (
        "Compute {a} * {b} - {c}.",
        [
            "First multiply {a} by {b}: {a} * {b} = {p}.",
            "Then subtract {c}: {p} - {c} = {y}.",
        ],
    ),
    (
        "What remains after subtracting {c} from the product of {a} and {b}?",
        [
            "The product is {a} * {b} = {p}.",
            "Subtracting {c} leaves {p} - {c} = {y}.",
        ],
    ),
    (
        "Find the value of ({a} times {b}) minus {c}.",
        [
            "Compute the multiplication: {a} * {b} = {p}.",
            "Finish with the subtraction: {p} - {c} = {y}.",
        ],
    ),
]

SUB_DIV_TEMPLATES = [
    (
        "Compute ({a} - {b}) / {c}.",
        [
            "First subtract {b} from {a}: {a} - {b} = {d}.",
            "Then divide by {c}: {d} / {c} = {y}.",
        ],
    ),
    (
        "After taking {b} away from {a}, divide the result by {c}. What is the answer?",
        [
            "The difference is {a} - {b} = {d}.",
            "Dividing that by {c} gives {d} / {c} = {y}.",
        ],
    ),
    (
        "What is the quotient when {a} minus {b} is divided by {c}?",
        [
            "Calculate the numerator first: {a} - {b} = {d}.",
            "The quotient is {d} / {c} = {y}.",
        ],
    ),
]

MIXED_TEMPLATES = [
    (
        "Compute {a} + {b} * {c} - {d}.",
        [
            "First multiply {b} by {c}: {b} * {c} = {p}.",
            "Then add {a}: {a} + {p} = {s}.",
            "Finally subtract {d}: {s} - {d} = {y}.",
        ],
    ),
    (
        "Start with {a}, add the product of {b} and {c}, then subtract {d}. What is left?",
        [
            "The product is {b} * {c} = {p}.",
            "Adding {a} gives {a} + {p} = {s}.",
            "Subtract {d} to get {s} - {d} = {y}.",
        ],
    ),
    (
        "Evaluate {a} plus {b} times {c}, reduced by {d}.",
        [
            "Use multiplication first: {b} * {c} = {p}.",
            "Combine with {a}: {a} + {p} = {s}.",
            "Reduce by {d}: {s} - {d} = {y}.",
        ],
    ),
]


def _render(template_pair, values: dict) -> tuple[str, list[str]]:
    question_template, step_templates = template_pair
    return (
        question_template.format(**values),
        [template.format(**values) for template in step_templates],
    )


def _add_mul(index: int, split: str, a: int, b: int, c: int, variant: int) -> dict:
    s = a + b
    y = s * c
    values = {"a": a, "b": b, "c": c, "s": s, "y": y}
    question, steps = _render(ADD_MUL_TEMPLATES[variant], values)
    return _record(index, split, "add_mul", variant, question, steps, y, [s, y], {"a": a, "b": b, "c": c})


def _mul_sub(index: int, split: str, a: int, b: int, c: int, variant: int) -> dict:
    p = a * b
    y = p - c
    values = {"a": a, "b": b, "c": c, "p": p, "y": y}
    question, steps = _render(MUL_SUB_TEMPLATES[variant], values)
    return _record(index, split, "mul_sub", variant, question, steps, y, [p, y], {"a": a, "b": b, "c": c})


def _sub_div(index: int, split: str, a: int, b: int, c: int, variant: int) -> dict:
    d = a - b
    y = d // c
    values = {"a": a, "b": b, "c": c, "d": d, "y": y}
    question, steps = _render(SUB_DIV_TEMPLATES[variant], values)
    return _record(index, split, "sub_div", variant, question, steps, y, [d, y], {"a": a, "b": b, "c": c})


def _mixed(index: int, split: str, a: int, b: int, c: int, d: int, variant: int) -> dict:
    p = b * c
    s = a + p
    y = s - d
    values = {"a": a, "b": b, "c": c, "d": d, "p": p, "s": s, "y": y}
    question, steps = _render(MIXED_TEMPLATES[variant], values)
    return _record(index, split, "mixed_add_mul_sub", variant, question, steps, y, [p, s, y], {"a": a, "b": b, "c": c, "d": d})


def _record(
    index: int,
    split: str,
    operation_type: str,
    template_variant: int,
    question: str,
    steps: list[str],
    answer: int,
    intermediate_results: list[int],
    meta: dict,
) -> dict:
    return {
        "id": f"htext_{split}_{index:04d}",
        "source": GENERATOR_VERSION,
        "split": split,
        "question": question,
        "cot": " ".join(steps),
        "whole_cot": " ".join(steps),
        "cot_steps_raw": steps,
        "cot_steps_text": steps,
        "answer": str(answer),
        "answer_normalized": str(answer),
        "num_steps": len(steps),
        "operation_type": operation_type,
        "intermediate_results": [str(x) for x in intermediate_results],
        "metadata": {
            **meta,
            "operation_type": operation_type,
            "template_variant": template_variant,
            "generator_version": GENERATOR_VERSION,
        },
    }


def validate_record(record: dict) -> bool:
    if record["cot"] != record["whole_cot"]:
        return False
    if record["cot"] != " ".join(record["cot_steps_text"]):
        return False
    meta = record["metadata"]
    op = record["operation_type"]
    variant = meta["template_variant"]
    if op == "add_mul":
        expected = _add_mul(0, record["split"], meta["a"], meta["b"], meta["c"], variant)
    elif op == "mul_sub":
        expected = _mul_sub(0, record["split"], meta["a"], meta["b"], meta["c"], variant)
    elif op == "sub_div":
        expected = _sub_div(0, record["split"], meta["a"], meta["b"], meta["c"], variant)
    elif op == "mixed_add_mul_sub":
        expected = _mixed(0, record["split"], meta["a"], meta["b"], meta["c"], meta["d"], variant)
    else:
        return False
    keys = ["question", "cot_steps_text", "whole_cot", "answer", "operation_type", "intermediate_results"]
    return all(record[key] == expected[key] for key in keys)


def generate_synthetic_split(
    train_size: int = 128,
    validation_size: int = 64,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    specs = []
    for variant in range(3):
        for a in range(2, 80):
            for b in range(2, 80):
                for c in range(2, 20):
                    specs.append(("add_mul", (a, b, c, variant)))
                    if a * b > c:
                        specs.append(("mul_sub", (a, b, c, variant)))
                    diff = a - b
                    if diff > 0 and diff % c == 0:
                        specs.append(("sub_div", (a, b, c, variant)))
                    d = (a + b * c) // 3
                    if d > 0 and a + b * c > d:
                        specs.append(("mixed_add_mul_sub", (a, b, c, d, variant)))
    rng.shuffle(specs)
    records = []
    needed = train_size + validation_size
    seen = set()
    for name, args in specs:
        if len(records) >= needed:
            break
        combo_key = (name, args)
        if combo_key in seen:
            continue
        seen.add(combo_key)
        split = "train" if len(records) < train_size else "validation"
        idx = len(records) if split == "train" else len(records) - train_size
        if name == "add_mul":
            records.append(_add_mul(idx, split, *args))
        elif name == "mul_sub":
            records.append(_mul_sub(idx, split, *args))
        elif name == "sub_div":
            records.append(_sub_div(idx, split, *args))
        else:
            records.append(_mixed(idx, split, *args))
    if len(records) != needed:
        raise RuntimeError(f"could only generate {len(records)} records, need {needed}")
    if not all(validate_record(record) for record in records):
        raise RuntimeError("generated invalid synthetic HText record")
    return records[:train_size], records[train_size:needed]


def write_jsonl(records: list[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
