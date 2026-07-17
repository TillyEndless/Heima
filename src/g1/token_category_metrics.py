from __future__ import annotations

import math
import re
from collections import Counter

import torch
import torch.nn.functional as F


NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
EXPR_RE = re.compile(r"-?\d+\s*[+\-*/]\s*-?\d+\s*=\s*-?\d+")


def cyclic_derangement(n: int) -> list[int]:
    if n < 2:
        raise ValueError("derangement requires at least two items")
    perm = [(i + 1) % n for i in range(n)]
    assert all(i != j for i, j in enumerate(perm))
    return perm


def token_nll(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    flat = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shift_labels)
    mask = shift_labels != -100
    return flat, mask


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum().item() == 0:
        return float("nan")
    return values[mask].mean().item()


def kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = F.log_softmax(p_logits.float(), dim=-1)
    q = F.log_softmax(q_logits.float(), dim=-1)
    return F.kl_div(q, p.exp(), reduction="none").sum(dim=-1)


def decoded_token_categories(tokenizer, token_ids: list[int]) -> dict[str, list[bool]]:
    numeric = []
    operator = []
    sample_specific = []
    for token_id in token_ids:
        piece = tokenizer.decode([token_id])
        low = piece.lower()
        is_num = bool(NUMBER_RE.search(piece))
        is_op = any(op in piece for op in ["+", "-", "*", "/"]) or any(
            word in low for word in ["add", "multiply", "subtract", "divide", "result"]
        )
        numeric.append(is_num)
        operator.append(is_op)
        sample_specific.append(is_num or is_op)
    return {
        "numeric": numeric,
        "operator": operator,
        "sample_specific": sample_specific,
    }


def template_without_numbers(text: str) -> str:
    return NUMBER_RE.sub("<NUM>", text)


def entropy(counts: Counter, total: int) -> float:
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        p = count / total
        value -= p * math.log2(p)
    return value


def template_statistics(records: list[dict], tokenizer) -> dict:
    templates = [template_without_numbers(r["cot"]) for r in records]
    tokenized = [tokenizer(r["cot"], add_special_tokens=False)["input_ids"] for r in records]
    max_len = max(len(ids) for ids in tokenized)
    entropies = []
    for pos in range(max_len):
        counts = Counter(ids[pos] for ids in tokenized if pos < len(ids))
        entropies.append(entropy(counts, sum(counts.values())))
    total_tokens = sum(len(ids) for ids in tokenized)
    numeric_tokens = 0
    specific_tokens = 0
    for ids in tokenized:
        cats = decoded_token_categories(tokenizer, ids)
        numeric_tokens += sum(cats["numeric"])
        specific_tokens += sum(cats["sample_specific"])
    overlap_pairs = []
    token_sets = [set(ids) for ids in tokenized]
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            inter = len(token_sets[i] & token_sets[j])
            union = len(token_sets[i] | token_sets[j])
            overlap_pairs.append(inter / union if union else 0.0)
    return {
        "unique_templates_without_numbers": len(set(templates)),
        "dataset_template_homogeneity": len(set(templates)) <= 4,
        "mean_token_overlap": sum(overlap_pairs) / len(overlap_pairs),
        "position_entropy_mean": sum(entropies) / len(entropies),
        "position_entropy": entropies,
        "numeric_token_fraction": numeric_tokens / total_tokens,
        "sample_specific_token_fraction": specific_tokens / total_tokens,
        "total_tokens": total_tokens,
    }


def extract_numbers(text: str) -> list[str]:
    return NUMBER_RE.findall(text)


def extract_expressions(text: str) -> list[str]:
    return [re.sub(r"\s+", "", m.group(0)) for m in EXPR_RE.finditer(text)]
