from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .heima_reuse import (
    ThinkingStateMode,
    extract_thinking_state,
    heima_ce_loss,
    official_embedding_replacement,
)
from .schema_adapter import LEGACY_THINKING_TOKEN, OFFICIAL_REASONING_TOKEN


THINKING_TOKEN = OFFICIAL_REASONING_TOKEN
LATENT_TOKEN = "<LATENT>"
ANSWER_PREFIX = "\nAnswer: "
DECODER_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Instruction:\nExplain the reasoning information encoded in the latent state.\n\n"
    "Latent:\n"
    f"{THINKING_TOKEN}\n\n"
    "Reasoning:\n"
)


@dataclass
class H0ForwardOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    thinking_hidden: torch.Tensor
    thinking_positions: torch.Tensor
    selected_positions: torch.Tensor
    thinking_state_semantics: str


@dataclass
class H1ForwardOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    inputs_embeds: torch.Tensor


@dataclass
class AnswerHiddenForwardOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    inputs_embeds: torch.Tensor


class LatentProjector(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(z))


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return heima_ce_loss(logits, labels)


def tokenize_text(tokenizer, text: str, max_length: int | None = None) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if max_length is not None:
        ids = ids[:max_length]
    return ids


def setup_special_tokens(tokenizer, *models) -> dict[str, int]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [THINKING_TOKEN, LEGACY_THINKING_TOKEN, LATENT_TOKEN]}
    )
    if added:
        for model in models:
            model.resize_token_embeddings(len(tokenizer))
    return {
        "thinking_id": tokenizer.convert_tokens_to_ids(THINKING_TOKEN),
        "legacy_thinking_id": tokenizer.convert_tokens_to_ids(LEGACY_THINKING_TOKEN),
        "latent_id": tokenizer.convert_tokens_to_ids(LATENT_TOKEN),
    }


def build_h0_labels(
    total_len: int,
    question_len: int,
    num_thinking_tokens: int,
    answer_prefix_len: int,
    answer_ids: torch.Tensor,
    thinking_id: int,
    pad_to: int | None = None,
) -> torch.Tensor:
    length = pad_to or total_len
    labels = torch.full((length,), -100, dtype=torch.long)
    thinking_start = question_len
    labels[thinking_start : thinking_start + num_thinking_tokens] = thinking_id
    answer_start = question_len + num_thinking_tokens + answer_prefix_len
    labels[answer_start : answer_start + answer_ids.numel()] = answer_ids.cpu()
    return labels


def build_h1_labels(
    total_len: int,
    target_start: int,
    target_ids: torch.Tensor,
    pad_to: int | None = None,
) -> torch.Tensor:
    length = pad_to or total_len
    labels = torch.full((length,), -100, dtype=torch.long)
    labels[target_start : target_start + target_ids.numel()] = target_ids.cpu()
    return labels


def h0_forward(
    model_a,
    tokenizer,
    records: list[dict],
    max_question_tokens: int,
    max_answer_tokens: int,
    num_thinking_tokens: int,
    thinking_state_mode: ThinkingStateMode = "predictor",
) -> H0ForwardOutput:
    device = next(model_a.parameters()).device
    ids_rows = []
    label_rows = []
    thinking_positions = []
    thinking_id = tokenizer.convert_tokens_to_ids(THINKING_TOKEN)
    prefix_ids = tokenize_text(tokenizer, ANSWER_PREFIX)
    for record in records:
        question_ids = tokenize_text(tokenizer, record["question"], max_question_tokens)
        answer_ids = tokenize_text(tokenizer, record["answer"] + tokenizer.eos_token, max_answer_tokens)
        ids = question_ids + [thinking_id] * num_thinking_tokens + prefix_ids + answer_ids
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        ids_rows.append(ids_tensor)
        thinking_positions.append(list(range(len(question_ids), len(question_ids) + num_thinking_tokens)))
        label_rows.append(
            build_h0_labels(
                total_len=len(ids),
                question_len=len(question_ids),
                num_thinking_tokens=num_thinking_tokens,
                answer_prefix_len=len(prefix_ids),
                answer_ids=torch.tensor(answer_ids, dtype=torch.long),
                thinking_id=thinking_id,
            )
        )
    max_len = max(row.numel() for row in ids_rows)
    input_ids = torch.full((len(records), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(records), max_len), dtype=torch.long, device=device)
    labels = torch.full((len(records), max_len), -100, dtype=torch.long, device=device)
    for i, row in enumerate(ids_rows):
        input_ids[i, : row.numel()] = row.to(device)
        attention_mask[i, : row.numel()] = 1
        labels[i, : label_rows[i].numel()] = label_rows[i].to(device)
    out = model_a(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    if num_thinking_tokens != 1:
        raise ValueError("strict single-stage HText currently requires num_thinking_tokens=1")
    state = extract_thinking_state(
        input_ids=input_ids,
        last_hidden_state=out.hidden_states[-1],
        thinking_token_id=thinking_id,
        mode=thinking_state_mode,
    )
    thinking_hidden = state.hidden.unsqueeze(1)
    pos_tensor = state.thinking_positions.view(input_ids.shape[0], num_thinking_tokens)
    selected_pos_tensor = state.selected_positions.view(input_ids.shape[0], num_thinking_tokens)
    loss = causal_lm_loss(out.logits, labels)
    return H0ForwardOutput(
        loss,
        out.logits,
        labels,
        input_ids,
        attention_mask,
        thinking_hidden,
        pos_tensor,
        selected_pos_tensor,
        state.semantics,
    )


def extract_thinking_hidden(
    model_a,
    tokenizer,
    records: list[dict],
    max_question_tokens: int,
    num_thinking_tokens: int,
    thinking_state_mode: ThinkingStateMode = "predictor",
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model_a.parameters()).device
    thinking_id = tokenizer.convert_tokens_to_ids(THINKING_TOKEN)
    ids_rows = []
    positions = []
    for record in records:
        question_ids = tokenize_text(tokenizer, record["question"], max_question_tokens)
        ids = question_ids + [thinking_id] * num_thinking_tokens
        ids_rows.append(torch.tensor(ids, dtype=torch.long))
        positions.append(list(range(len(question_ids), len(question_ids) + num_thinking_tokens)))
    max_len = max(row.numel() for row in ids_rows)
    input_ids = torch.full((len(records), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(records), max_len), dtype=torch.long, device=device)
    for i, row in enumerate(ids_rows):
        input_ids[i, : row.numel()] = row.to(device)
        attention_mask[i, : row.numel()] = 1
    out = model_a(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    if num_thinking_tokens != 1:
        raise ValueError("strict single-stage HText currently requires num_thinking_tokens=1")
    state = extract_thinking_state(
        input_ids=input_ids,
        last_hidden_state=out.hidden_states[-1],
        thinking_token_id=thinking_id,
        mode=thinking_state_mode,
    )
    return state.hidden.unsqueeze(1), state.selected_positions.view(input_ids.shape[0], num_thinking_tokens)


def h0_answer_from_hidden_forward(
    model_a,
    tokenizer,
    records: list[dict],
    z: torch.Tensor,
    max_question_tokens: int,
    max_answer_tokens: int,
) -> AnswerHiddenForwardOutput:
    device = next(model_a.parameters()).device
    embed = model_a.get_input_embeddings()
    prefix_ids = tokenize_text(tokenizer, ANSWER_PREFIX)
    seqs = []
    labels = []
    for i, record in enumerate(records):
        question_ids = tokenize_text(tokenizer, record["question"], max_question_tokens)
        answer_ids = tokenize_text(tokenizer, record["answer"] + tokenizer.eos_token, max_answer_tokens)
        q_ids = torch.tensor(question_ids, dtype=torch.long, device=device)
        p_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device)
        a_ids = torch.tensor(answer_ids, dtype=torch.long, device=device)
        seq = torch.cat(
            [
                embed(q_ids.unsqueeze(0)).squeeze(0),
                z[i : i + 1],
                embed(p_ids.unsqueeze(0)).squeeze(0),
                embed(a_ids.unsqueeze(0)).squeeze(0),
            ],
            dim=0,
        )
        seqs.append(seq)
        labels.append(
            build_h1_labels(
                total_len=seq.size(0),
                target_start=len(question_ids) + 1 + len(prefix_ids),
                target_ids=torch.tensor(answer_ids, dtype=torch.long),
            )
        )
    max_len = max(seq.size(0) for seq in seqs)
    hidden = seqs[0].size(-1)
    inputs_embeds = torch.zeros((len(records), max_len, hidden), dtype=seqs[0].dtype, device=device)
    attention_mask = torch.zeros((len(records), max_len), dtype=torch.long, device=device)
    label_tensor = torch.full((len(records), max_len), -100, dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        inputs_embeds[i, : seq.size(0), :] = seq
        attention_mask[i, : seq.size(0)] = 1
        label_tensor[i, : labels[i].numel()] = labels[i].to(device)
    out = model_a(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return AnswerHiddenForwardOutput(causal_lm_loss(out.logits, label_tensor), out.logits, label_tensor, inputs_embeds)


def _decoder_template(question: str, mode: str) -> str:
    if mode == "qz":
        return DECODER_TEMPLATE.format(question=question)
    if mode == "q":
        return (
            "Question:\n{question}\n\n"
            "Instruction:\nExplain the full reasoning for this question.\n\n"
            "Reasoning:\n"
        ).format(question=question)
    if mode == "z":
        return (
            "Instruction:\nExplain the reasoning information encoded in the latent state.\n\n"
            "Latent:\n"
            f"{THINKING_TOKEN}\n\n"
            "Reasoning:\n"
        )
    raise ValueError(f"unknown H1 mode: {mode}")


def _decoder_prompt_ids(tokenizer, question: str, mode: str = "qz") -> tuple[list[int], int | None]:
    prompt = _decoder_template(question, mode)
    ids = tokenize_text(tokenizer, prompt)
    if mode == "q":
        return ids, None
    thinking_id = tokenizer.convert_tokens_to_ids(THINKING_TOKEN)
    positions = [i for i, token_id in enumerate(ids) if token_id == thinking_id]
    if len(positions) != 1:
        raise ValueError(f"expected exactly one {THINKING_TOKEN}, found {len(positions)}")
    return ids, positions[0]


def h1_forward(
    model_b,
    tokenizer,
    records: list[dict],
    z: torch.Tensor,
    projector: nn.Module,
    max_cot_tokens: int,
    latent_index: int = 0,
    latent_override: torch.Tensor | None = None,
    mode: str = "qz",
) -> H1ForwardOutput:
    device = next(model_b.parameters()).device
    if mode in {"qz", "z"}:
        use_z = z[:, latent_index, :] if latent_override is None else latent_override
        projected = projector(use_z)
    else:
        projected = None
    embed = model_b.get_input_embeddings()
    rows = []
    labels = []
    embed_rows = []
    for record in records:
        prompt_ids, latent_pos = _decoder_prompt_ids(tokenizer, record["question"], mode)
        target_ids = tokenize_text(tokenizer, record["cot"] + tokenizer.eos_token, max_cot_tokens)
        ids = prompt_ids + target_ids
        rows.append(torch.tensor(ids, dtype=torch.long))
        labels.append(build_h1_labels(len(ids), len(prompt_ids), torch.tensor(target_ids, dtype=torch.long)))
        embed_rows.append(latent_pos)
    max_len = max(row.numel() for row in rows)
    input_ids = torch.full((len(records), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(records), max_len), dtype=torch.long, device=device)
    label_tensor = torch.full((len(records), max_len), -100, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        input_ids[i, : row.numel()] = row.to(device)
        attention_mask[i, : row.numel()] = 1
        label_tensor[i, : labels[i].numel()] = labels[i].to(device)
    token_embeds = embed(input_ids)
    if mode == "q":
        inputs_embeds = token_embeds
    else:
        assert projected is not None
        replacement_mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=device)
        for i, latent_pos in enumerate(embed_rows):
            if latent_pos is None:
                raise ValueError("latent mode requires thinking-token replacement position")
            replacement_mask[i, latent_pos] = True
        inputs_embeds = official_embedding_replacement(token_embeds, projected.unsqueeze(1), replacement_mask)
    out = model_b(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    loss = causal_lm_loss(out.logits, label_tensor)
    return H1ForwardOutput(loss, out.logits, label_tensor, inputs_embeds)
