"""Gradient-control helpers for Heima Stage2 interpreter supervision.

This module intentionally does not implement a new Heima trainer. It isolates
one experimental switch after official Stage1 interpreter training: whether the
frozen interpreter loss is detached from Model A's latent state or allowed to
supervise Model A through the latent input.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn


class Stage2Mode(str, Enum):
    """Allowed Stage2 comparison modes."""

    HEIMA_BASELINE = "heima_baseline"
    OURS_INTERP_SUPERVISION = "ours_interp_supervision"


@dataclass(frozen=True)
class Stage2StepOutput:
    """Scalar outputs and gradient audit values from one Stage2 step."""

    mode: Stage2Mode
    ntp_loss: float
    interp_loss: float
    total_loss: float
    grad_A_from_interp: float
    grad_B_from_interp: float
    teacher_B_frozen: bool


def freeze_teacher_interpreters(interpreters: Iterable[nn.Module]) -> None:
    """Freeze Stage1 interpreter checkpoints for Stage2 teacher use."""

    for model in interpreters:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)


def assert_teacher_interpreters_frozen(interpreters: Iterable[nn.Module]) -> None:
    """Raise if any teacher interpreter parameter is trainable."""

    trainable = []
    for model_idx, model in enumerate(interpreters):
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable.append(f"decoder[{model_idx}].{name}")
    if trainable:
        raise AssertionError("Stage2 teacher B must be frozen: " + ", ".join(trainable[:10]))


def compute_grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    """Compute the L2 norm of all finite gradients in a parameter iterable."""

    total = torch.zeros((), dtype=torch.float64)
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if not torch.isfinite(grad).all():
            return float("inf")
        total += grad.double().pow(2).sum().cpu()
    return float(total.sqrt().item())


def _stage2_latent_for_interp(z: Tensor, mode: Stage2Mode) -> Tensor:
    if mode == Stage2Mode.HEIMA_BASELINE:
        return z.detach()
    if mode == Stage2Mode.OURS_INTERP_SUPERVISION:
        return z
    raise ValueError(f"Unsupported Stage2 mode: {mode}")


def _decoder_parameters(interpreters: Iterable[nn.Module]) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for model in interpreters:
        params.extend(list(model.parameters()))
    return params


def run_stage2_train_step(
    *,
    model_a: nn.Module,
    interpreters: Mapping[str, nn.Module],
    optimizer_a: torch.optim.Optimizer,
    batch: Mapping[str, Tensor],
    mode: Stage2Mode | str,
    lambda_interp: float,
) -> Stage2StepOutput:
    """Run one strict Stage2 step with frozen B and auditable gradient flow.

    The model objects used in tests and dry-runs follow the same contract as the
    official Heima path at the point of loss composition:

    - ``model_a(batch)`` returns ``{"ntp_loss": loss, "latents": {section: z}}``.
    - each interpreter consumes ``(z, batch)`` and returns an interpreter CE loss.

    Production entrypoints adapt official Heima batches/models to this boundary;
    the only algorithmic switch is whether ``z`` is detached before B.
    """

    stage2_mode = Stage2Mode(mode)
    teacher_list = list(interpreters.values())
    freeze_teacher_interpreters(teacher_list)
    assert_teacher_interpreters_frozen(teacher_list)

    optimizer_a.zero_grad(set_to_none=True)

    out = model_a(batch)
    ntp_loss = out["ntp_loss"]
    latents: Mapping[str, Tensor] = out["latents"]

    interp_loss = torch.zeros((), device=ntp_loss.device, dtype=ntp_loss.dtype)
    for section, interpreter in interpreters.items():
        if section not in latents:
            raise KeyError(f"Missing latent for interpreter section {section!r}")
        z_for_b = _stage2_latent_for_interp(latents[section], stage2_mode)
        interp_loss = interp_loss + interpreter(z_for_b, batch)

    total_loss = ntp_loss
    if stage2_mode == Stage2Mode.OURS_INTERP_SUPERVISION:
        total_loss = total_loss + float(lambda_interp) * interp_loss

    optimizer_a.zero_grad(set_to_none=True)
    for param in _decoder_parameters(teacher_list):
        param.grad = None
    if interp_loss.requires_grad:
        interp_loss.backward(retain_graph=True)
        grad_a_from_interp = compute_grad_norm(model_a.parameters())
        grad_b_from_interp = compute_grad_norm(_decoder_parameters(teacher_list))
    else:
        # In the strict Heima baseline, z is detached and B is frozen. The
        # interpreter loss is still computed for logging/evaluation, but it has
        # no autograd path by construction.
        grad_a_from_interp = 0.0
        grad_b_from_interp = 0.0

    optimizer_a.zero_grad(set_to_none=True)
    for param in _decoder_parameters(teacher_list):
        param.grad = None
    total_loss.backward()
    optimizer_a.step()

    return Stage2StepOutput(
        mode=stage2_mode,
        ntp_loss=float(ntp_loss.detach().cpu().item()),
        interp_loss=float(interp_loss.detach().cpu().item()),
        total_loss=float(total_loss.detach().cpu().item()),
        grad_A_from_interp=grad_a_from_interp,
        grad_B_from_interp=grad_b_from_interp,
        teacher_B_frozen=True,
    )
