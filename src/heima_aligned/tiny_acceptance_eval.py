from __future__ import annotations

from dataclasses import dataclass, asdict
import re
import string
from typing import Any, Iterable, Mapping, Sequence

INTERVENTIONS = ("q_only", "correct", "shuffle", "zero")
TOKEN_LEVEL_METRICS = (
    "full_nll",
    "content_token_nll",
    "numeric_token_accuracy",
    "entity_token_accuracy",
    "answer_token_accuracy",
)
INTERVENTION_METRICS = (
    "full_nll",
    "content_nll",
    "generation_exact_match",
)
WARM_B_COMPARISONS = (
    "cold_b_joint",
    "warm_b_joint",
)

_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|x)?$", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.'%+-]*")
_TEMPLATE_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "for",
    "in",
    "on",
    "by",
    "with",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "it",
    "its",
    "question",
    "asks",
    "answer",
    "given",
    "image",
    "figure",
    "chart",
    "graph",
    "shows",
    "shown",
    "using",
    "single",
    "word",
    "phrase",
    "reasoning",
    "therefore",
    "because",
    "based",
    "determine",
    "identify",
    "indicates",
}


@dataclass(frozen=True)
class TokenDiagnostics:
    tokens: list[str]
    content_mask: list[bool]
    numeric_mask: list[bool]
    entity_mask: list[bool]
    answer_mask: list[bool]

    def counts(self) -> dict[str, int]:
        return {
            "tokens": len(self.tokens),
            "content_tokens": sum(self.content_mask),
            "numeric_tokens": sum(self.numeric_mask),
            "entity_tokens": sum(self.entity_mask),
            "answer_tokens": sum(self.answer_mask),
        }


def simple_word_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def normalize_for_exact_match(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return " ".join(text.lower().translate(table).split())


def generation_exact_match(prediction: str, target: str) -> bool:
    return normalize_for_exact_match(prediction) == normalize_for_exact_match(target)


def build_reasoning_token_diagnostics(reasoning: str, answer: str) -> TokenDiagnostics:
    tokens = simple_word_tokens(reasoning)
    answer_terms = {t.lower() for t in simple_word_tokens(answer)}
    content_mask: list[bool] = []
    numeric_mask: list[bool] = []
    entity_mask: list[bool] = []
    answer_mask: list[bool] = []
    for tok in tokens:
        low = tok.lower().strip(".,;:!?()[]{}\"'")
        is_num = bool(_NUMERIC_RE.match(low))
        is_entity = tok[:1].isupper() and low not in _TEMPLATE_WORDS and not is_num
        is_answer = low in answer_terms if answer_terms else False
        is_content = (is_num or is_entity or is_answer or (len(low) > 3 and low not in _TEMPLATE_WORDS))
        numeric_mask.append(is_num)
        entity_mask.append(is_entity)
        answer_mask.append(is_answer)
        content_mask.append(is_content)
    return TokenDiagnostics(tokens, content_mask, numeric_mask, entity_mask, answer_mask)


def required_reasoning_metrics() -> dict[str, Any]:
    return {
        "token_level": list(TOKEN_LEVEL_METRICS),
        "interventions": list(INTERVENTIONS),
        "intervention_metrics": list(INTERVENTION_METRICS),
        "generation": {"do_sample": False, "exact_match_normalization": "lowercase_strip_punctuation_whitespace"},
        "primary_signal": "correct.full_nll < shuffle.full_nll and correct.content_nll < shuffle.content_nll",
    }


def build_intervention_eval_plan(section: str = "reasoning") -> dict[str, Any]:
    if section != "reasoning":
        raise ValueError("tiny acceptance evaluator is reasoning-only")
    return {
        "section": section,
        "keep_fixed": ["question", "decoder_prompt", "teacher_target", "attention_mask", "labels"],
        "vary_only": "injected_latent",
        "interventions": {
            "q_only": {"question": True, "latent": False},
            "correct": {"question": True, "latent": "z_i"},
            "shuffle": {"question": True, "latent": "z_j_derangement"},
            "zero": {"question": True, "latent": "0"},
        },
        "metrics": required_reasoning_metrics(),
    }


def build_warm_b_interface() -> dict[str, Any]:
    return {
        "stage_after": "freeze_A_train_B",
        "shared_start": "reasoning interpreter checkpoint after frozen-A training",
        "comparisons": list(WARM_B_COMPARISONS),
        "cold_b_joint": {
            "B_initialization": "fresh pretrained Model B",
            "purpose": "tests whether joint Loss1 can learn B and A from a cold interpreter",
        },
        "warm_b_joint": {
            "B_initialization": "B_reasoning from freeze_A_train_B",
            "purpose": "isolates whether a competent interpreter improves Loss1 feedback to A",
        },
        "invariant_between_comparisons": [
            "Model A checkpoint",
            "data split",
            "batch order",
            "optimizer hyperparameters",
            "lambda_loss1",
            "reasoning target",
            "latent intervention evaluator",
        ],
    }


def build_tiny_acceptance_eval_manifest(example: Mapping[str, str] | None = None) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "purpose": "mechanism validation: test whether reasoning latent carries sample-specific information",
        "not_a_benchmark_reproduction": True,
        "reasoning_token_level_evaluation": required_reasoning_metrics(),
        "latent_intervention_evaluation": build_intervention_eval_plan("reasoning"),
        "warm_b_evaluation_interface": build_warm_b_interface(),
        "answer_metric": "answer_accuracy from deterministic Model A generation",
    }
    if example is not None:
        diag = build_reasoning_token_diagnostics(example.get("reasoning", ""), example.get("answer", ""))
        manifest["example_token_diagnostics"] = asdict(diag) | {"counts": diag.counts()}
    return manifest
