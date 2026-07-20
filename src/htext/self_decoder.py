from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from .heima_reuse import heima_ce_loss, official_embedding_replacement


SelfDecodeLabelMode = Literal["text_only", "latent_and_text"]
SelfDecodeAdapterType = Literal["identity", "ln_linear"]
SelfDecodeRoleMode = Literal["none", "typed"]


@dataclass(frozen=True)
class SelfDecoderBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    replacement_mask: torch.Tensor
    latent_slot_positions: torch.Tensor
    target_start_positions: torch.Tensor
    target_token_counts: torch.Tensor
    stage: str
    token_id: int
    prompt_texts: list[str]


@dataclass(frozen=True)
class SelfDecoderForwardOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    inputs_embeds: torch.Tensor
    base_embeds: torch.Tensor
    injected_latent: torch.Tensor
    audit: dict
    raw_output: object


class SelfDecodeLatentInterface(nn.Module):
    """Trainable latent-stage interface without owning or copying Model A."""

    def __init__(
        self,
        hidden_size: int,
        stages: list[str] | tuple[str, ...],
        *,
        adapter_type: SelfDecodeAdapterType = "identity",
        role_mode: SelfDecodeRoleMode = "none",
        token_embedding_weight: torch.Tensor | None = None,
        stage_token_ids: dict[str, int] | None = None,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.stages = tuple(stages)
        self.adapter_type = adapter_type
        self.role_mode = role_mode
        if adapter_type == "identity":
            self.adapters = nn.ModuleDict({stage: nn.Identity() for stage in self.stages})
        elif adapter_type == "ln_linear":
            modules = {}
            for stage in self.stages:
                linear = nn.Linear(hidden_size, hidden_size)
                nn.init.eye_(linear.weight)
                nn.init.zeros_(linear.bias)
                modules[stage] = nn.Sequential(nn.LayerNorm(hidden_size), linear)
            self.adapters = nn.ModuleDict(modules)
        else:
            raise ValueError(f"unsupported adapter_type: {adapter_type}")

        if role_mode == "none":
            self.roles = nn.ParameterDict()
        elif role_mode == "typed":
            if token_embedding_weight is None or stage_token_ids is None:
                raise ValueError("typed role mode requires token_embedding_weight and stage_token_ids")
            roles = {}
            for stage in self.stages:
                token_id = int(stage_token_ids[stage])
                init = token_embedding_weight[token_id].detach().clone().float()
                if init.numel() != hidden_size:
                    raise ValueError("role init dimension does not match hidden_size")
                roles[stage] = nn.Parameter(init)
            self.roles = nn.ParameterDict(roles)
        else:
            raise ValueError(f"unsupported role_mode: {role_mode}")

    def forward(self, stage: str, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if stage not in self.adapters:
            raise KeyError(stage)
        param = next(self.adapters[stage].parameters(), None)
        adapter_input = z if param is None else z.to(dtype=param.dtype)
        adapter_out = self.adapters[stage](adapter_input)
        if self.role_mode == "typed":
            role = self.roles[stage].to(device=z.device, dtype=adapter_out.dtype).view(1, -1)
        else:
            role = torch.zeros((1, adapter_out.shape[-1]), device=z.device, dtype=adapter_out.dtype)
        injected = adapter_out + role
        audit = {
            "stage": stage,
            "adapter_type": self.adapter_type,
            "role_mode": self.role_mode,
            "norm_z": z.detach().float().norm(dim=-1).cpu().tolist(),
            "norm_adapter": adapter_out.detach().float().norm(dim=-1).cpu().tolist(),
            "norm_role": role.detach().float().norm(dim=-1).expand(z.shape[0]).cpu().tolist(),
            "norm_injected": injected.detach().float().norm(dim=-1).cpu().tolist(),
        }
        return injected, audit


def self_decoder_prompt(question: str, stage: str, thinking_token: str, *, max_question_chars: int | None = None) -> str:
    if max_question_chars is not None:
        question = question[:max_question_chars]
    return (
        f"Question:\n{question}\n\n"
        f"Instruction:\nReconstruct the Heima {stage} thought from the latent. Do not use the image.\n\n"
        f"Latent:\n{thinking_token}\n\n"
        "Reasoning:\n"
    )


def _tokenize(tokenizer, text: str, max_len: int | None = None) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids[:max_len] if max_len is not None else ids


def build_self_decoder_batch(
    *,
    tokenizer,
    records: list[dict],
    stage: str,
    thinking_token: str,
    target_key: str,
    label_mode: SelfDecodeLabelMode,
    device: torch.device,
    max_question_chars: int | None = None,
    max_target_tokens: int | None = None,
) -> SelfDecoderBatch:
    token_id = tokenizer.convert_tokens_to_ids(thinking_token)
    if token_id is None or token_id < 0:
        raise ValueError(f"tokenizer does not know thinking token {thinking_token!r}")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer.pad_token_id must be set")

    rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    slot_positions: list[int] = []
    target_starts: list[int] = []
    target_counts: list[int] = []
    prompts: list[str] = []
    for record in records:
        prompt = self_decoder_prompt(
            str(record["question"]),
            stage,
            thinking_token,
            max_question_chars=max_question_chars,
        )
        prompt_ids = _tokenize(tokenizer, prompt)
        target_ids = _tokenize(tokenizer, str(record[target_key]) + tokenizer.eos_token, max_target_tokens)
        locs = [idx for idx, value in enumerate(prompt_ids) if value == token_id]
        if len(locs) != 1:
            raise RuntimeError(f"expected one latent slot for {stage}, got {locs}")
        slot = locs[0]
        labels = [-100] * len(prompt_ids) + target_ids
        if label_mode == "latent_and_text":
            labels[slot] = token_id
        elif label_mode != "text_only":
            raise ValueError(f"unsupported label_mode: {label_mode}")
        rows.append(prompt_ids + target_ids)
        label_rows.append(labels)
        slot_positions.append(slot)
        target_starts.append(len(prompt_ids))
        target_counts.append(len(target_ids))
        prompts.append(prompt)

    max_len = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long, device=device)
    labels = torch.full_like(input_ids, -100)
    attention_mask = torch.zeros_like(input_ids)
    replacement_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        n = len(row)
        input_ids[i, :n] = torch.tensor(row, dtype=torch.long, device=device)
        labels[i, :n] = torch.tensor(label_rows[i], dtype=torch.long, device=device)
        attention_mask[i, :n] = 1
        replacement_mask[i, slot_positions[i]] = True

    if int((labels != -100).sum().item()) == 0:
        raise RuntimeError("self-decoder batch has zero non-ignored labels")

    return SelfDecoderBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        replacement_mask=replacement_mask,
        latent_slot_positions=torch.tensor(slot_positions, dtype=torch.long, device=device),
        target_start_positions=torch.tensor(target_starts, dtype=torch.long, device=device),
        target_token_counts=torch.tensor(target_counts, dtype=torch.long, device=device),
        stage=stage,
        token_id=token_id,
        prompt_texts=prompts,
    )


def inject_latent_into_model_a_embeds(model_a, input_ids: torch.Tensor, replacement_mask: torch.Tensor, injected_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    base_embeds = model_a.get_input_embeddings()(input_ids)
    replaced = official_embedding_replacement(base_embeds, injected_latent.unsqueeze(1), replacement_mask)
    return base_embeds, replaced


def forward_model_a_text_only_self_decoder(
    *,
    model_a,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    replacement_mask: torch.Tensor,
    injected_latent: torch.Tensor,
    output_hidden_states: bool = True,
) -> SelfDecoderForwardOutput:
    base_embeds, inputs_embeds = inject_latent_into_model_a_embeds(
        model_a,
        input_ids,
        replacement_mask,
        injected_latent,
    )
    out = model_a(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        use_cache=False,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    loss = out.loss if getattr(out, "loss", None) is not None else heima_ce_loss(out.logits, labels)
    audit = {
        "used_inputs_embeds": True,
        "passed_input_ids": False,
        "pixel_values": None,
        "image_grid_thw": None,
        "pixel_values_videos": None,
        "video_grid_thw": None,
        "use_cache": False,
        "non_ignored_labels": int((labels != -100).sum().item()),
        "replacement_count": int(replacement_mask.sum().item()),
    }
    return SelfDecoderForwardOutput(
        loss=loss,
        logits=out.logits,
        labels=labels,
        inputs_embeds=inputs_embeds,
        base_embeds=base_embeds,
        injected_latent=injected_latent,
        audit=audit,
        raw_output=out,
    )


def prepare_self_decoder_latent(z: torch.Tensor, *, detach_latent: bool) -> torch.Tensor:
    return z.detach() if detach_latent else z
