from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


ANSWER_PREFIX = "\nAnswer: "


@dataclass
class MainForwardOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    z: torch.Tensor
    labels: torch.Tensor
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor


def assert_parameter_independence(model_a, model_b) -> None:
    ids_a = {id(p) for p in model_a.parameters()}
    ids_b = {id(p) for p in model_b.parameters()}
    if not ids_a.isdisjoint(ids_b):
        raise ValueError("Model A and Model B share Parameter objects")


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def last_valid_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    positions = attention_mask.long().sum(dim=1) - 1
    batch = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch, positions, :]


def build_main_labels(
    total_len: int,
    question_len: int,
    latent_len: int,
    answer_prefix_len: int,
    answer_ids: torch.Tensor,
    pad_to: int | None = None,
) -> torch.Tensor:
    length = pad_to or total_len
    labels = torch.full((length,), -100, dtype=torch.long)
    start = question_len + latent_len + answer_prefix_len
    labels[start : start + answer_ids.numel()] = answer_ids.cpu()
    return labels


def tokenize_text(tokenizer, text: str, max_length: int | None = None) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if max_length is not None:
        ids = ids[:max_length]
    return ids


def encode_question_batch(tokenizer, records: list[dict], max_question_tokens: int, device):
    encoded = [
        tokenize_text(tokenizer, record["question"], max_question_tokens)
        for record in records
    ]
    max_len = max(len(ids) for ids in encoded)
    pad = tokenizer.pad_token_id
    input_ids = torch.full((len(records), max_len), pad, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(records), max_len), dtype=torch.long, device=device)
    for i, ids in enumerate(encoded):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[i, : len(ids)] = 1
    return input_ids, attention_mask, encoded


def extract_latent(model_a, question_ids, question_mask) -> torch.Tensor:
    out = model_a(
        input_ids=question_ids,
        attention_mask=question_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    z = last_valid_hidden(out.hidden_states[-1], question_mask)
    if z.requires_grad:
        z.retain_grad()
    return z


def main_forward(
    model_a,
    tokenizer,
    records: list[dict],
    max_question_tokens: int,
    max_answer_tokens: int,
    latent_override: torch.Tensor | None = None,
) -> MainForwardOutput:
    device = next(model_a.parameters()).device
    question_ids, question_mask, question_lists = encode_question_batch(
        tokenizer, records, max_question_tokens, device
    )
    z = extract_latent(model_a, question_ids, question_mask)
    use_z = z if latent_override is None else latent_override

    embed = model_a.get_input_embeddings()
    prefix_ids = tokenize_text(tokenizer, ANSWER_PREFIX)
    answer_lists = [
        tokenize_text(tokenizer, record["answer"] + tokenizer.eos_token, max_answer_tokens)
        for record in records
    ]

    seq_embeds = []
    labels = []
    for i, record in enumerate(records):
        q_ids = torch.tensor(question_lists[i], dtype=torch.long, device=device)
        p_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device)
        a_ids = torch.tensor(answer_lists[i], dtype=torch.long, device=device)
        parts = [
            embed(q_ids.unsqueeze(0)).squeeze(0),
            use_z[i : i + 1],
            embed(p_ids.unsqueeze(0)).squeeze(0),
            embed(a_ids.unsqueeze(0)).squeeze(0),
        ]
        seq = torch.cat(parts, dim=0)
        seq_embeds.append(seq)
        labels.append(
            build_main_labels(
                total_len=seq.size(0),
                question_len=len(q_ids),
                latent_len=1,
                answer_prefix_len=len(p_ids),
                answer_ids=a_ids.cpu(),
            )
        )

    max_len = max(seq.size(0) for seq in seq_embeds)
    hidden = seq_embeds[0].size(-1)
    inputs_embeds = torch.zeros(
        len(records), max_len, hidden, dtype=seq_embeds[0].dtype, device=device
    )
    attention_mask = torch.zeros(len(records), max_len, dtype=torch.long, device=device)
    label_tensor = torch.full((len(records), max_len), -100, dtype=torch.long, device=device)
    for i, seq in enumerate(seq_embeds):
        inputs_embeds[i, : seq.size(0), :] = seq
        attention_mask[i, : seq.size(0)] = 1
        label_tensor[i, : labels[i].numel()] = labels[i].to(device)

    out = model_a(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    loss = causal_lm_loss(out.logits, label_tensor)
    return MainForwardOutput(loss, out.logits, z, label_tensor, inputs_embeds, attention_mask)
