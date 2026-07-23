"""Strict Stage2 Loss2 hidden alignment in frozen Model-B space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

Loss2Pool = Literal["mean", "last"]
Loss2Distance = Literal["normalized_mse", "mse", "cosine"]
Loss2TextContext = Literal["cumulative", "section_only"]


@dataclass(frozen=True)
class Loss2Features:
    loss: Tensor
    h_latent: Tensor
    h_text: Tensor
    latent_shape: tuple[int, ...]
    text_shape: tuple[int, ...]
    pool: str
    distance: str
    h_text_detached: bool


def causal_lm_loss(logits: Tensor, labels: Tensor) -> Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def tokenize_text(tokenizer, text: str, max_len: int | None = None) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids[:max_len] if max_len is not None else ids


def pad_rows(tokenizer, rows: list[list[int]], label_rows: list[list[int]], device):
    max_len = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for i, row in enumerate(rows):
        n = len(row)
        input_ids[i, :n] = torch.tensor(row, dtype=torch.long, device=device)
        attention_mask[i, :n] = 1
        labels[i, : len(label_rows[i])] = torch.tensor(label_rows[i], dtype=torch.long, device=device)
    return input_ids, attention_mask, labels


def pool_target_hidden(hidden: Tensor, labels: Tensor, pool: Loss2Pool) -> Tensor:
    target_mask = labels.ne(-100)
    if int(target_mask.sum().item()) == 0:
        raise RuntimeError("Loss2 target hidden pool received no target tokens")
    if pool == "mean":
        weights = target_mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    if pool == "last":
        positions = target_mask.long().sum(dim=1)
        idx = []
        for row in range(labels.size(0)):
            target_positions = torch.where(target_mask[row])[0]
            idx.append(target_positions[-1])
        idx_t = torch.stack(idx)
        batch = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch, idx_t, :]
    raise ValueError(pool)


def loss2_distance(h_latent: Tensor, h_text: Tensor, distance: Loss2Distance) -> Tensor:
    if h_latent.shape != h_text.shape:
        raise ValueError(f"Loss2 hidden shape mismatch: {tuple(h_latent.shape)} vs {tuple(h_text.shape)}")
    if distance == "normalized_mse":
        return (F.normalize(h_latent.float(), dim=-1) - F.normalize(h_text.float(), dim=-1)).pow(2).mean(dim=-1).mean()
    if distance == "mse":
        return (h_latent.float() - h_text.float()).pow(2).mean()
    if distance == "cosine":
        return (1.0 - F.cosine_similarity(h_latent.float(), h_text.float(), dim=-1)).mean()
    raise ValueError(distance)


def default_latent_prompt(record: Mapping[str, str], section: str, tokenizer, max_q: int, thinking_token: str) -> str:
    q_ids = tokenize_text(tokenizer, str(record["question"]), max_q)
    question = tokenizer.decode(q_ids, skip_special_tokens=False)
    return (
        f"Question:\n{question}\n\n"
        f"Instruction:\nReconstruct the Heima {section} thought from the latent. Do not use the image.\n\n"
        f"{thinking_token}\n\nTarget:\n"
    )


def default_text_prompt(
    record: Mapping[str, str],
    section: str,
    sections: Sequence[str],
    tokenizer,
    max_q: int,
    context_mode: Loss2TextContext,
) -> str:
    q_ids = tokenize_text(tokenizer, str(record["question"]), max_q)
    question = tokenizer.decode(q_ids, skip_special_tokens=False)
    before = []
    if context_mode == "cumulative":
        for name in sections:
            if name == section:
                break
            before.append(f"{name}:\n{record[name]}\n")
    elif context_mode != "section_only":
        raise ValueError(context_mode)
    prefix = "\n".join(before)
    return f"Question:\n{question}\n\nExplicit Heima thoughts so far:\n{prefix}\n{section}:\n"


def build_latent_path(
    model_b: nn.Module,
    projector,
    tokenizer,
    records: Sequence[Mapping[str, str]],
    section: str,
    z: Tensor,
    *,
    max_q: int,
    max_target: int,
    thinking_token: str,
):
    device = next(model_b.parameters()).device
    rows, label_rows, slots = [], [], []
    token_id = tokenizer.convert_tokens_to_ids(thinking_token)
    for rec in records:
        prompt_ids = tokenize_text(tokenizer, default_latent_prompt(rec, section, tokenizer, max_q, thinking_token))
        target_ids = tokenize_text(tokenizer, str(rec[section]) + tokenizer.eos_token, max_target)
        rows.append(prompt_ids + target_ids)
        label_rows.append([-100] * len(prompt_ids) + target_ids)
        locs = [idx for idx, value in enumerate(prompt_ids) if value == token_id]
        if len(locs) != 1:
            raise RuntimeError(f"expected one latent slot for {section}, got {locs}")
        slots.append(locs[0])
    input_ids, attention_mask, labels = pad_rows(tokenizer, rows, label_rows, device)
    embeds = model_b.get_input_embeddings()(input_ids)
    projected = projector(z).unsqueeze(1)
    if projected.size(0) != len(records):
        raise ValueError("projected latent batch size mismatch")
    flat = embeds.clone().reshape(-1, embeds.shape[-1])
    for row, pos in enumerate(slots):
        flat[row * embeds.shape[1] + pos] = projected[row, 0].to(flat.dtype)
    inputs_embeds = flat.reshape_as(embeds)
    return inputs_embeds, attention_mask, labels


def build_text_path(
    model_b: nn.Module,
    tokenizer,
    records: Sequence[Mapping[str, str]],
    section: str,
    sections: Sequence[str],
    *,
    max_q: int,
    max_target: int,
    context_mode: Loss2TextContext,
):
    device = next(model_b.parameters()).device
    rows, label_rows = [], []
    for rec in records:
        prompt_ids = tokenize_text(tokenizer, default_text_prompt(rec, section, sections, tokenizer, max_q, context_mode))
        target_ids = tokenize_text(tokenizer, str(rec[section]) + tokenizer.eos_token, max_target)
        rows.append(prompt_ids + target_ids)
        label_rows.append([-100] * len(prompt_ids) + target_ids)
    return pad_rows(tokenizer, rows, label_rows, device)


def loss2_forward(
    *,
    model_b: nn.Module,
    projector,
    tokenizer,
    records: Sequence[Mapping[str, str]],
    section: str,
    sections: Sequence[str],
    z: Tensor,
    max_q: int,
    max_target: int,
    thinking_token: str,
    pool: Loss2Pool = "mean",
    distance: Loss2Distance = "normalized_mse",
    text_context_mode: Loss2TextContext = "cumulative",
    detach_latent: bool = False,
) -> Loss2Features:
    z_for_latent = z.detach() if detach_latent else z
    inputs_embeds, latent_attention, latent_labels = build_latent_path(
        model_b,
        projector,
        tokenizer,
        records,
        section,
        z_for_latent,
        max_q=max_q,
        max_target=max_target,
        thinking_token=thinking_token,
    )
    latent_out = model_b(
        inputs_embeds=inputs_embeds,
        attention_mask=latent_attention,
        output_hidden_states=True,
        use_cache=False,
    )
    h_latent = pool_target_hidden(latent_out.hidden_states[-1], latent_labels, pool)
    with torch.no_grad():
        text_ids, text_attention, text_labels = build_text_path(
            model_b,
            tokenizer,
            records,
            section,
            sections,
            max_q=max_q,
            max_target=max_target,
            context_mode=text_context_mode,
        )
        text_out = model_b(
            input_ids=text_ids,
            attention_mask=text_attention,
            output_hidden_states=True,
            use_cache=False,
        )
        h_text = pool_target_hidden(text_out.hidden_states[-1], text_labels, pool).detach()
    loss = loss2_distance(h_latent, h_text, distance)
    return Loss2Features(
        loss=loss,
        h_latent=h_latent,
        h_text=h_text,
        latent_shape=tuple(h_latent.shape),
        text_shape=tuple(h_text.shape),
        pool=pool,
        distance=distance,
        h_text_detached=not h_text.requires_grad,
    )


def compute_grad_norm(parameters) -> tuple[float, bool]:
    total = 0.0
    finite = True
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        finite = finite and bool(torch.isfinite(grad).all().item())
        total += float(grad.float().pow(2).sum().item())
    return total**0.5, finite
