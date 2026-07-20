from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

from .embedding_injection import inject_single_latent


@dataclass
class Loss1Output:
    loss: torch.Tensor
    logits: torch.Tensor
    inputs_embeds: torch.Tensor
    labels: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None


def make_labels(
    input_ids: torch.Tensor,
    target_start: int,
    target_len: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    labels = torch.full_like(input_ids, -100)
    labels[:, target_start : target_start + target_len] = input_ids[
        :, target_start : target_start + target_len
    ]
    if attention_mask is not None:
        labels = labels.masked_fill(attention_mask == 0, -100)
    return labels


def compute_loss1(
    model_b: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    latent: torch.Tensor,
    latent_pos: int,
    target_start: int,
    target_len: int,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> Loss1Output:
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if labels is None:
        labels = make_labels(input_ids, target_start, target_len, attention_mask)

    token_embeds = model_b.get_input_embeddings()(input_ids)
    inputs_embeds = inject_single_latent(token_embeds, latent, latent_pos)
    out = model_b(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    loss = manual_causal_lm_loss(out.logits, labels)
    return Loss1Output(
        loss=loss,
        logits=out.logits,
        inputs_embeds=inputs_embeds,
        labels=labels,
        hidden_states=out.hidden_states,
    )


def manual_causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
