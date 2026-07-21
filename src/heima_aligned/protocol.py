from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

SECTIONS = ("summary", "caption", "reasoning")
THINKING_TOKENS = {
    "summary": "<THINKING_OF_SUMMARY>",
    "caption": "<THINKING_OF_CAPTION>",
    "reasoning": "<THINKING_OF_REASONING>",
}
STAGES = (
    "stage_0_explicit",
    "stage_1_summary",
    "stage_2_caption",
    "stage_3_reasoning",
    "stage_4_recover",
)
STAGE_REPLACED = {
    "stage_0_explicit": (),
    "stage_1_summary": ("summary",),
    "stage_2_caption": ("summary", "caption"),
    "stage_3_reasoning": ("summary", "caption", "reasoning"),
    "stage_4_recover": ("summary", "caption", "reasoning"),
}
MODE_NAMES = {
    "heima_scaled_baseline",
    "compute_matched_main_only",
    "ours_warm_b_fixed",
    "ours_warm_b_joint",
    "ours_cold_b_joint",
    "ours_warm_b_fixed_loss1_loss2",
    "ours_warm_b_joint_loss1_loss2",
    "ours_cold_b_joint_loss1_loss2",
    "main_loss1_only",
    "main_loss1_loss2",
}
IGNORE_INDEX = -100


@dataclass(frozen=True)
class HeimaRecord:
    id: str
    image: str
    question: str
    summary: str
    caption: str
    reasoning: str
    answer: str

    @classmethod
    def from_mapping(cls, obj: Mapping[str, Any], index: int = 0) -> "HeimaRecord":
        def first(*keys: str, default: str = "") -> str:
            for key in keys:
                value = obj.get(key)
                if value not in (None, ""):
                    return str(value)
            return default
        return cls(
            id=first("id", "sample_id", default=str(index)),
            image=first("image", "image_path"),
            question=first("question", "prompt"),
            summary=first("summary"),
            caption=first("caption"),
            reasoning=first("reasoning", "cot", "whole_cot"),
            answer=first("answer", "final_answer"),
        )

    def section_text(self, section: str) -> str:
        return getattr(self, section)


def load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict):
        for key in ("data", "samples", "train", "validation", "test"):
            if isinstance(data.get(key), list):
                return data[key]
        raise ValueError(f"cannot find sample list in {path}")
    if not isinstance(data, list):
        raise ValueError(f"expected list/dict dataset in {path}")
    return data


def load_heima_records(path: str | Path, *, limit: int | None = None) -> list[HeimaRecord]:
    rows = load_json_or_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    records = [HeimaRecord.from_mapping(row, i) for i, row in enumerate(rows)]
    validate_records(records)
    return records


def validate_records(records: Sequence[HeimaRecord]) -> None:
    if not records:
        raise ValueError("empty Heima dataset split")
    missing = []
    for rec in records:
        for field in ("image", "question", "summary", "caption", "reasoning", "answer"):
            if not getattr(rec, field):
                missing.append((rec.id, field))
    if missing:
        raise ValueError(f"missing required Heima fields: {missing[:8]}")


def split_hash(records: Sequence[HeimaRecord]) -> str:
    h = hashlib.sha256()
    for rec in records:
        h.update(json.dumps(asdict(rec), sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def assert_disjoint_splits(*splits: Sequence[HeimaRecord]) -> None:
    seen: dict[str, int] = {}
    for split_idx, split in enumerate(splits):
        for rec in split:
            prev = seen.setdefault(rec.id, split_idx)
            if prev != split_idx:
                raise ValueError(f"sample id {rec.id!r} appears in split {prev} and {split_idx}")


def encoder_user_prompt(record: HeimaRecord) -> str:
    return f"<image>\nQuestion: {record.question}\n"


def build_stage_response(record: HeimaRecord, stage: str) -> str:
    if stage not in STAGE_REPLACED:
        raise ValueError(f"unknown stage: {stage}")
    replaced = set(STAGE_REPLACED[stage])
    pieces: list[str] = []
    for section in SECTIONS:
        value = THINKING_TOKENS[section] if section in replaced else record.section_text(section)
        pieces.append(f"<{section.upper()}> {value} </{section.upper()}>")
    pieces.append(f"<CONCLUSION> {record.answer} </CONCLUSION>")
    return "\n\n".join(pieces)


def build_encoder_sample(record: HeimaRecord, stage: str) -> dict[str, Any]:
    return {
        "id": record.id,
        "image": record.image,
        "stage": stage,
        "prompt": encoder_user_prompt(record),
        "response": build_stage_response(record, stage),
        "replaced_sections": list(STAGE_REPLACED[stage]),
    }


def decoder_prompt(record: HeimaRecord, section: str, *, template: int = 0) -> str:
    token = THINKING_TOKENS[section]
    templates = {
        "summary": [
            "{question}\nCan you provide the details of thinking progress {token} for summarizing the given question?",
            "For the question \"{question}\", how does the thinking progress {token} unfold during the summarization?",
        ],
        "caption": [
            "{question}\nCan you provide the thinking progress {token} for the caption of the given question?",
            "What is the thinking progress {token} involved in crafting the caption for the question: {question}?",
        ],
        "reasoning": [
            "{question}\nCan you provide the thinking progress {token} for the reasoning of the given question?",
            "Explain how the thinking progress {token} unfolds during reasoning for the question: {question}?",
        ],
    }
    return templates[section][template % len(templates[section])].format(question=record.question, token=token)


def decoder_target(record: HeimaRecord, section: str) -> str:
    prefixes = {
        "summary": "The thinking progress for the summary of the given question is: ",
        "caption": "The thinking progress for the caption of the given question can be explained as follows: ",
        "reasoning": "The thinking progress for the reasoning of the given question is illustrated as follows: ",
    }
    return prefixes[section] + record.section_text(section)


def build_decoder_sample(record: HeimaRecord, section: str) -> dict[str, Any]:
    return {
        "id": f"{record.id}-{section}",
        "source_id": record.id,
        "section": section,
        "prompt": decoder_prompt(record, section),
        "target": decoder_target(record, section),
        "thinking_token": THINKING_TOKENS[section],
    }


def build_main_labels(input_ids: Sequence[int], prompt_len: int, pad_token_id: int | None = None) -> list[int]:
    labels = [IGNORE_INDEX] * len(input_ids)
    for i in range(prompt_len, len(input_ids)):
        labels[i] = int(input_ids[i])
    if pad_token_id is not None:
        labels = [IGNORE_INDEX if int(tok) == int(pad_token_id) else lab for tok, lab in zip(input_ids, labels)]
    if all(x == IGNORE_INDEX for x in labels):
        raise ValueError("zero non-ignored main labels")
    return labels


def build_decoder_labels(input_ids: Sequence[int], prompt_len: int, *, train_on_input: bool = True) -> list[int]:
    if train_on_input:
        labels = [int(x) for x in input_ids]
    else:
        labels = [IGNORE_INDEX] * len(input_ids)
        for i in range(prompt_len, len(input_ids)):
            labels[i] = int(input_ids[i])
    if all(x == IGNORE_INDEX for x in labels):
        raise ValueError("zero non-ignored decoder labels")
    return labels


@dataclass(frozen=True)
class ThinkingState:
    hidden: torch.Tensor
    thinking_positions: torch.Tensor
    selected_positions: torch.Tensor
    mode: str


def extract_predictor_hidden(input_ids: torch.Tensor, last_hidden_state: torch.Tensor, token_id: int) -> ThinkingState:
    mask = input_ids.eq(int(token_id))
    if mask.sum(dim=1).min().item() != 1 or mask.sum(dim=1).max().item() != 1:
        raise ValueError("expected exactly one typed thinking token per sample")
    thinking_pos = mask.float().argmax(dim=1).long()
    selected = thinking_pos - 1
    if (selected < 0).any().item():
        raise ValueError("thinking token at position 0 has no predictor hidden")
    batch = torch.arange(input_ids.shape[0], device=input_ids.device)
    return ThinkingState(last_hidden_state[batch, selected], thinking_pos, selected, "predictor")


class HeimaOfficialAbstractProjection(nn.Module):
    """Official Torchtune-compatible projector: Linear -> ReLU -> Linear -> Dropout(0)."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
            nn.Dropout(0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def replace_embeddings_after_lookup(base_embeds: torch.Tensor, projected_latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype is not torch.bool:
        mask = mask.bool()
    expected = int(mask.sum().item())
    flat_latent = projected_latent.reshape(-1, base_embeds.shape[-1])
    if flat_latent.shape[0] != expected:
        raise ValueError(f"latent count {flat_latent.shape[0]} != mask true count {expected}")
    out = base_embeds.clone()
    flat = out.reshape(-1, out.shape[-1])
    flat[mask.reshape(-1)] = flat_latent.to(dtype=flat.dtype, device=flat.device)
    return flat.reshape_as(out)


def stage_loss_sections(stage: str) -> tuple[str, ...]:
    if stage == "joint_stage_1":
        return ("summary",)
    if stage == "joint_stage_2":
        return ("summary", "caption")
    if stage in {"joint_stage_3", "joint_recover"}:
        return SECTIONS
    raise ValueError(stage)


def mode_plan(mode: str) -> list[dict[str, Any]]:
    if mode == "heima_scaled_baseline":
        return [
            {"stage": "explicit_cot_sft", "train_a": True, "train_b": False, "loss1": False},
            {"stage": "progressive_summary", "encoder_stage": "stage_1_summary", "train_a": True, "train_b": False, "loss1": False},
            {"stage": "progressive_caption", "encoder_stage": "stage_2_caption", "train_a": True, "train_b": False, "loss1": False},
            {"stage": "progressive_reasoning", "encoder_stage": "stage_3_reasoning", "train_a": True, "train_b": False, "loss1": False},
            {"stage": "recover", "encoder_stage": "stage_4_recover", "train_a": True, "train_b": False, "loss1": False},
            {"stage": "train_interpreter_summary", "train_a": False, "train_b": True, "loss1": True, "section": "summary"},
            {"stage": "train_interpreter_caption", "train_a": False, "train_b": True, "loss1": True, "section": "caption"},
            {"stage": "train_interpreter_reasoning", "train_a": False, "train_b": True, "loss1": True, "section": "reasoning"},
            {"stage": "eval_encoder"}, {"stage": "eval_decoder"}, {"stage": "eval_causal"},
        ]
    if mode == "compute_matched_main_only":
        return [{"stage": s, "train_a": True, "train_b": False, "lambda_loss1": 0.0} for s in ("ours_joint_summary", "ours_joint_caption", "ours_joint_reasoning", "ours_recover")]
    if mode == "ours_warm_b_fixed":
        return [{"stage": s, "train_a": True, "train_b": False, "b_frozen_differentiable": True, "loss1": True} for s in ("ours_joint_summary", "ours_joint_caption", "ours_joint_reasoning", "ours_recover")]
    if mode == "ours_warm_b_joint":
        return [{"stage": s, "train_a": True, "train_b": True, "loss1": True} for s in ("ours_joint_summary", "ours_joint_caption", "ours_joint_reasoning", "ours_recover")]
    if mode == "ours_cold_b_joint":
        return [{"stage": s, "train_a": True, "train_b": True, "loss1": True, "cold_b": True} for s in ("ours_joint_summary", "ours_joint_caption", "ours_joint_reasoning", "ours_recover")]
    if mode == "ours_warm_b_fixed_loss1_loss2":
        return [{"stage": s, "train_a": True, "train_b": False, "b_frozen_differentiable": True, "loss1": True, "loss2": True, "teacher_frozen": True} for s in ("ours_joint_summary", "ours_joint_caption", "ours_joint_reasoning", "ours_recover")]
    if mode == "ours_warm_b_joint_loss1_loss2":
        return [{"stage": s, "train_a": True, "train_b": True, "loss1": True, "loss2": True, "teacher_frozen": True} for s in ("ours_joint_summary", "ours_joint_caption", "ours_joint_reasoning", "ours_recover")]
    if mode == "ours_cold_b_joint_loss1_loss2":
        return [{"stage": s, "train_a": True, "train_b": True, "cold_b": True, "loss1": True, "loss2": True, "teacher_frozen": True} for s in ("ours_joint_summary", "ours_joint_caption", "ours_joint_reasoning", "ours_recover")]
    if mode == "main_loss1_only":
        return [{"stage": s, "train_a": True, "train_b": True, "loss1": True, "loss2": False, "lambda_loss2": 0.0} for s in ("ours_joint_summary", "ours_joint_caption", "ours_joint_reasoning", "ours_recover")]
    if mode == "main_loss1_loss2":
        return [{"stage": s, "train_a": True, "train_b": True, "loss1": True, "loss2": True, "teacher_frozen": True} for s in ("ours_joint_summary", "ours_joint_caption", "ours_joint_reasoning", "ours_recover")]
    raise ValueError(f"unknown mode: {mode}")


def config_hash(obj: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()
