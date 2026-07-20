from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class EncoderOutput:
    logits: torch.Tensor
    thinking_hidden: torch.Tensor
    thinking_mask: torch.Tensor
    loss_metadata: dict


@dataclass
class DecoderOutput:
    logits: torch.Tensor
    loss_metadata: dict


class HeimaEncoderInterface(Protocol):
    def forward_encoder(self, *args, **kwargs) -> EncoderOutput:
        ...


class HeimaDecoderInterface(Protocol):
    def forward_decoder(self, *args, **kwargs) -> DecoderOutput:
        ...


class HFGPT2EncoderAdapter:
    def __init__(self, model):
        self.model = model

    def forward_encoder(self, input_ids, attention_mask, thinking_mask) -> EncoderOutput:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = out.hidden_states[-1][thinking_mask].view(
            input_ids.shape[0], -1, out.hidden_states[-1].shape[-1]
        )
        return EncoderOutput(
            logits=out.logits,
            thinking_hidden=hidden,
            thinking_mask=thinking_mask,
            loss_metadata={"backend": "hf_gpt2_adapter"},
        )


class HFGPT2DecoderAdapter:
    def __init__(self, model):
        self.model = model

    def forward_decoder(self, inputs_embeds, attention_mask) -> DecoderOutput:
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return DecoderOutput(logits=out.logits, loss_metadata={"backend": "hf_gpt2_adapter"})

