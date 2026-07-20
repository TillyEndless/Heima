from __future__ import annotations

from dataclasses import dataclass

import torch

from .latent_reasoner import causal_lm_loss, tokenize_text


LATENT_TOKEN = "<|endoftext|>"
DECODER_PROMPT = (
    "Instruction:\n"
    "Decode the complete reasoning encoded in the latent state.\n\n"
    "Latent:\n"
    f"{LATENT_TOKEN}\n\n"
    "Reasoning:\n"
)


@dataclass
class Loss1ForwardOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    inputs_embeds: torch.Tensor


def ensure_latent_token(tokenizer, *models) -> int:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer.convert_tokens_to_ids(LATENT_TOKEN)


def replace_latent_with_cat(
    token_embeds: torch.Tensor,
    latent: torch.Tensor,
    latent_pos: int,
) -> torch.Tensor:
    return torch.cat(
        [
            token_embeds[:, :latent_pos, :],
            latent.unsqueeze(1),
            token_embeds[:, latent_pos + 1 :, :],
        ],
        dim=1,
    )


def build_loss1_labels(
    total_len: int,
    prompt_len: int,
    latent_len: int,
    cot_ids: torch.Tensor,
    pad_to: int | None = None,
) -> torch.Tensor:
    length = pad_to or total_len
    labels = torch.full((length,), -100, dtype=torch.long)
    start = prompt_len + latent_len
    labels[start : start + cot_ids.numel()] = cot_ids.cpu()
    return labels


def prompt_ids_and_latent_pos(tokenizer) -> tuple[list[int], int]:
    ids = tokenize_text(tokenizer, DECODER_PROMPT)
    latent_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
    positions = [i for i, token_id in enumerate(ids) if token_id == latent_id]
    if len(positions) != 1:
        raise ValueError(f"expected one {LATENT_TOKEN}, found {len(positions)}")
    return ids, positions[0]


def loss1_forward(
    model_b,
    tokenizer,
    records: list[dict],
    z: torch.Tensor,
    max_cot_tokens: int,
    latent_override: torch.Tensor | None = None,
) -> Loss1ForwardOutput:
    device = next(model_b.parameters()).device
    prompt_ids, latent_pos = prompt_ids_and_latent_pos(tokenizer)
    latent_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
    use_z = z if latent_override is None else latent_override
    embed = model_b.get_input_embeddings()

    seq_ids = []
    labels = []
    for record in records:
        cot_ids = tokenize_text(tokenizer, record["cot"] + tokenizer.eos_token, max_cot_tokens)
        ids = prompt_ids + cot_ids
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)
        if prompt_tensor.eq(latent_id).sum().item() != 1:
            raise ValueError("latent placeholder count must be exactly one")
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        seq_ids.append(ids_tensor)
        labels.append(
            build_loss1_labels(
                total_len=len(ids),
                prompt_len=latent_pos,
                latent_len=1,
                cot_ids=torch.tensor(cot_ids, dtype=torch.long),
            )
        )

    max_len = max(ids.numel() for ids in seq_ids)
    input_ids = torch.full(
        (len(records), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(len(records), max_len, dtype=torch.long, device=device)
    label_tensor = torch.full((len(records), max_len), -100, dtype=torch.long, device=device)
    for i, ids in enumerate(seq_ids):
        input_ids[i, : ids.numel()] = ids.to(device)
        attention_mask[i, : ids.numel()] = 1
        label_tensor[i, : labels[i].numel()] = labels[i].to(device)

    token_embeds = embed(input_ids)
    inputs_embeds = replace_latent_with_cat(token_embeds, use_z, latent_pos)
    out = model_b(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    loss = causal_lm_loss(out.logits, label_tensor)
    return Loss1ForwardOutput(loss, out.logits, label_tensor, inputs_embeds)
