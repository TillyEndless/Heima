"""Model-A-only online self-decode supervision helpers.

The mechanism uses one trainable Model A object for both the first-pass latent
production and the second-pass CoT reconstruction forwards. It deliberately has
no Model B, no trainable projector, and no role embedding parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

DEFAULT_SECTIONS = ("summary", "caption", "reasoning")
DEFAULT_EXPLAIN_PROMPTS = {
    "summary": "Reconstruct the Heima summary thought from the continuous latent.",
    "caption": "Reconstruct the Heima caption thought from the continuous latent.",
    "reasoning": "Reconstruct the Heima reasoning thought from the continuous latent.",
}


class AOnlySelfDecodeMode(str, Enum):
    """Training/eval modes for Model-A-only self-decode."""

    A_ONLY_MAIN_BASELINE = "a_only_main_baseline"
    A_ONLY_SELF_DECODE = "a_only_self_decode"


@dataclass(frozen=True)
class SelfDecodeFeatures:
    inputs_embeds: Tensor
    attention_mask: Tensor
    labels: Tensor
    prompt_lengths: list[int]
    latent_positions: list[int | None]
    prefix_lengths: list[int]
    target_lengths: list[int]


@dataclass(frozen=True)
class FirstPassOutput:
    main_loss: Tensor
    latents: Mapping[str, Tensor]


@dataclass(frozen=True)
class AOnlyStepOutput:
    mode: AOnlySelfDecodeMode
    main_loss: float
    self_loss: float
    total_loss: float
    per_section_loss: dict[str, float]
    grad_z_norm: dict[str, float]
    grad_A_from_self_decode_norm: float
    grad_A_total_norm: float
    has_model_b: bool
    optimizer_contains_model_b: bool
    use_projector: bool
    use_role_embedding: bool
    extra_trainable_params_except_A: int
    expected_forward_count_per_batch: int
    actual_forward_count_per_batch: int
    finite: bool


FirstPassFn = Callable[[nn.Module, object], FirstPassOutput | Mapping[str, object]]


def causal_lm_loss(logits: Tensor, labels: Tensor) -> Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
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




def compute_tensor_grad_norm(grads) -> tuple[float, bool]:
    total = 0.0
    finite = True
    for grad in grads:
        if grad is None:
            continue
        finite = finite and bool(torch.isfinite(grad).all().item())
        total += float(grad.detach().float().pow(2).sum().item())
    return total**0.5, finite

def tokenize_text(tokenizer, text: str, max_length: int | None = None) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids[:max_length] if max_length is not None else ids


def section_prefix(section: str) -> str:
    return f"\nTarget {section}:\n"


def explain_prompt_text(record: Mapping[str, str], section: str, tokenizer, max_q: int) -> str:
    q_ids = tokenize_text(tokenizer, str(record["question"]), max_q)
    question = tokenizer.decode(q_ids, skip_special_tokens=False)
    instruction = DEFAULT_EXPLAIN_PROMPTS.get(
        section, f"Reconstruct the Heima {section} thought from the continuous latent."
    )
    return f"Instruction:\n{instruction}\nDo not use the image.\n\nQuestion/context:\n{question}\n\nLatent:\n"


def _eos(tokenizer) -> str:
    return getattr(tokenizer, "eos_token", None) or ""


def build_self_decode_features(
    *,
    model_a: nn.Module,
    tokenizer,
    records: Sequence[Mapping[str, str]],
    section: str,
    z: Tensor | None,
    max_q: int,
    max_target: int,
    include_latent: bool = True,
) -> SelfDecodeFeatures:
    """Build second-pass inputs_embeds and CE labels for one CoT section.

    Layout per row:
    explain_prompt + question/context tokens, optional continuous latent slot,
    section prefix tokens, target text_cot_i tokens.
    Labels are -100 except on target text tokens.
    """

    device = next(model_a.parameters()).device
    embed = model_a.get_input_embeddings()
    seqs: list[Tensor] = []
    labels_rows: list[Tensor] = []
    prompt_lengths: list[int] = []
    latent_positions: list[int | None] = []
    prefix_lengths: list[int] = []
    target_lengths: list[int] = []

    if include_latent and z is None:
        raise ValueError("include_latent=True requires a continuous z tensor")
    if z is not None and z.dim() != 2:
        raise ValueError(f"z must have shape [batch, hidden], got {tuple(z.shape)}")
    if include_latent and z is not None and z.size(0) != len(records):
        raise ValueError("z batch size must match records")

    for i, rec in enumerate(records):
        prompt_ids = tokenize_text(tokenizer, explain_prompt_text(rec, section, tokenizer, max_q))
        prefix_ids = tokenize_text(tokenizer, section_prefix(section))
        target_ids = tokenize_text(tokenizer, str(rec[section]) + _eos(tokenizer), max_target)
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long, device=device)
        target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)
        parts = [embed(prompt_tensor.unsqueeze(0)).squeeze(0)]
        prompt_len = len(prompt_ids)
        latent_pos = None
        if include_latent:
            latent_pos = prompt_len
            parts.append(z[i : i + 1])
        parts.extend([
            embed(prefix_tensor.unsqueeze(0)).squeeze(0),
            embed(target_tensor.unsqueeze(0)).squeeze(0),
        ])
        seq = torch.cat(parts, dim=0)
        labels = torch.full((seq.size(0),), -100, dtype=torch.long, device=device)
        target_start = prompt_len + (1 if include_latent else 0) + len(prefix_ids)
        labels[target_start : target_start + len(target_ids)] = target_tensor
        seqs.append(seq)
        labels_rows.append(labels)
        prompt_lengths.append(prompt_len)
        latent_positions.append(latent_pos)
        prefix_lengths.append(len(prefix_ids))
        target_lengths.append(len(target_ids))

    max_len = max(seq.size(0) for seq in seqs)
    hidden = seqs[0].size(-1)
    inputs_embeds = torch.zeros(len(seqs), max_len, hidden, dtype=seqs[0].dtype, device=device)
    attention_mask = torch.zeros(len(seqs), max_len, dtype=torch.long, device=device)
    labels = torch.full((len(seqs), max_len), -100, dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        inputs_embeds[i, : seq.size(0)] = seq
        attention_mask[i, : seq.size(0)] = 1
        labels[i, : labels_rows[i].size(0)] = labels_rows[i]

    return SelfDecodeFeatures(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        prompt_lengths=prompt_lengths,
        latent_positions=latent_positions,
        prefix_lengths=prefix_lengths,
        target_lengths=target_lengths,
    )


def self_decode_forward(
    *,
    model_a: nn.Module,
    tokenizer,
    records: Sequence[Mapping[str, str]],
    section: str,
    z: Tensor | None,
    max_q: int,
    max_target: int,
    include_latent: bool = True,
) -> tuple[Tensor, Tensor, Tensor, SelfDecodeFeatures]:
    features = build_self_decode_features(
        model_a=model_a,
        tokenizer=tokenizer,
        records=records,
        section=section,
        z=z,
        max_q=max_q,
        max_target=max_target,
        include_latent=include_latent,
    )
    out = model_a(
        inputs_embeds=features.inputs_embeds,
        attention_mask=features.attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    loss = causal_lm_loss(out.logits, features.labels)
    return loss, out.logits, features.labels, features


def _as_first_pass_output(obj: FirstPassOutput | Mapping[str, object]) -> FirstPassOutput:
    if isinstance(obj, FirstPassOutput):
        return obj
    main = obj.get("main_loss", obj.get("ntp_loss"))
    latents = obj.get("latents")
    if not isinstance(main, Tensor) or not isinstance(latents, Mapping):
        raise TypeError("first_pass_fn must return FirstPassOutput or mapping with main_loss/ntp_loss and latents")
    return FirstPassOutput(main_loss=main, latents=latents)  # type: ignore[arg-type]


def run_a_only_train_step(
    *,
    model_a: nn.Module,
    optimizer_a: torch.optim.Optimizer,
    records: Sequence[Mapping[str, str]],
    tokenizer,
    first_pass_fn: FirstPassFn,
    mode: AOnlySelfDecodeMode | str,
    lambda_self: float,
    sections: Sequence[str] = DEFAULT_SECTIONS,
    max_q: int = 160,
    max_target: int = 160,
    step_optimizer: bool = True,
) -> AOnlyStepOutput:
    """Run one Model-A-only Stage2 step with N+1 A forwards.

    Baseline still performs second-pass self-decode forwards for logging, but it
    does so under no_grad with detached z and keeps L_total equal to L_main.
    Self-decode mode keeps the z graph and backpropagates once through
    L_main + lambda_self * mean_i(L_cot_i).
    """

    stage_mode = AOnlySelfDecodeMode(mode)
    sections = tuple(sections)
    optimizer_a.zero_grad(set_to_none=True)
    first = _as_first_pass_output(first_pass_fn(model_a, records))
    for section in sections:
        if section not in first.latents:
            raise KeyError(f"missing first-pass latent for section {section!r}")
        if first.latents[section].requires_grad:
            first.latents[section].retain_grad()

    per_losses: dict[str, Tensor] = {}
    actual_forward_count = 1
    if stage_mode == AOnlySelfDecodeMode.A_ONLY_MAIN_BASELINE:
        with torch.no_grad():
            for section in sections:
                z = first.latents[section].detach()
                loss, _logits, _labels, _features = self_decode_forward(
                    model_a=model_a,
                    tokenizer=tokenizer,
                    records=records,
                    section=section,
                    z=z,
                    max_q=max_q,
                    max_target=max_target,
                    include_latent=True,
                )
                per_losses[section] = loss.detach()
                actual_forward_count += 1
        self_loss = torch.stack([v.to(first.main_loss.device) for v in per_losses.values()]).mean()
        grad_a_from_self = 0.0
        grad_z = {section: 0.0 for section in sections}
        total = first.main_loss
    else:
        for section in sections:
            loss, _logits, _labels, _features = self_decode_forward(
                model_a=model_a,
                tokenizer=tokenizer,
                records=records,
                section=section,
                z=first.latents[section],
                max_q=max_q,
                max_target=max_target,
                include_latent=True,
            )
            per_losses[section] = loss
            actual_forward_count += 1
        self_loss = torch.stack(list(per_losses.values())).mean()
        params = [param for param in model_a.parameters() if param.requires_grad]
        z_tensors = [first.latents[section] for section in sections]
        grads = torch.autograd.grad(
            self_loss,
            params + z_tensors,
            retain_graph=True,
            allow_unused=True,
        )
        grad_a_from_self, finite_self = compute_tensor_grad_norm(grads[: len(params)])
        grad_z = {
            section: compute_tensor_grad_norm([grad])[0]
            for section, grad in zip(sections, grads[len(params) :])
        }
        total = first.main_loss + float(lambda_self) * self_loss

    total.backward()
    grad_a_total, finite_total = compute_grad_norm(model_a.parameters())
    if step_optimizer:
        optimizer_a.step()

    finite_losses = torch.isfinite(first.main_loss.detach()) and torch.isfinite(self_loss.detach()) and torch.isfinite(total.detach())
    finite = bool(finite_losses.item()) and finite_total
    if stage_mode == AOnlySelfDecodeMode.A_ONLY_SELF_DECODE:
        finite = finite and finite_self

    return AOnlyStepOutput(
        mode=stage_mode,
        main_loss=float(first.main_loss.detach().cpu().item()),
        self_loss=float(self_loss.detach().cpu().item()),
        total_loss=float(total.detach().cpu().item()),
        per_section_loss={k: float(v.detach().cpu().item()) for k, v in per_losses.items()},
        grad_z_norm=grad_z,
        grad_A_from_self_decode_norm=grad_a_from_self,
        grad_A_total_norm=grad_a_total,
        has_model_b=False,
        optimizer_contains_model_b=False,
        use_projector=False,
        use_role_embedding=False,
        extra_trainable_params_except_A=0,
        expected_forward_count_per_batch=len(sections) + 1,
        actual_forward_count_per_batch=actual_forward_count,
        finite=finite,
    )


@torch.no_grad()
def evaluate_self_decode_interventions(
    *,
    model_a: nn.Module,
    tokenizer,
    records: Sequence[Mapping[str, str]],
    first_pass_fn: FirstPassFn,
    sections: Sequence[str] = DEFAULT_SECTIONS,
    max_q: int = 160,
    max_target: int = 160,
) -> dict[str, dict[str, float]]:
    first = _as_first_pass_output(first_pass_fn(model_a, records))
    metrics: dict[str, dict[str, float]] = {}
    for section in sections:
        z = first.latents[section]
        shuffle = torch.roll(z, shifts=1, dims=0) if z.size(0) > 1 else torch.zeros_like(z)
        zero = torch.zeros_like(z)
        correct_loss, _l, _lab, _f = self_decode_forward(
            model_a=model_a, tokenizer=tokenizer, records=records, section=section, z=z,
            max_q=max_q, max_target=max_target, include_latent=True,
        )
        shuffle_loss, _l, _lab, _f = self_decode_forward(
            model_a=model_a, tokenizer=tokenizer, records=records, section=section, z=shuffle,
            max_q=max_q, max_target=max_target, include_latent=True,
        )
        zero_loss, _l, _lab, _f = self_decode_forward(
            model_a=model_a, tokenizer=tokenizer, records=records, section=section, z=zero,
            max_q=max_q, max_target=max_target, include_latent=True,
        )
        q_loss, _l, _lab, _f = self_decode_forward(
            model_a=model_a, tokenizer=tokenizer, records=records, section=section, z=None,
            max_q=max_q, max_target=max_target, include_latent=False,
        )
        correct = float(correct_loss.item())
        shuffled = float(shuffle_loss.item())
        zeroed = float(zero_loss.item())
        q_only = float(q_loss.item())
        metrics[section] = {
            "correct": correct,
            "shuffle": shuffled,
            "zero": zeroed,
            "q_only": q_only,
            "shuffle_margin": shuffled - correct,
            "zero_margin": zeroed - correct,
            "qz_gain": q_only - correct,
        }
    return metrics
