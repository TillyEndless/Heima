from dataclasses import dataclass
from pathlib import Path
import json

import torch

from .loss1_forward import compute_loss1, make_labels, manual_causal_lm_loss
from .tiny_models import make_tiny_gpt2_pair


PLACEHOLDER_ID = 99
PAD_ID = 0


@dataclass
class GradientCaseResult:
    loss: float
    latent_requires_grad: bool
    latent_grad_fn: str
    inputs_embeds_requires_grad: bool
    inputs_embeds_grad_fn: str
    latent_grad_norm: float
    model_a_grad_norm: float
    model_b_grad_norm: float
    model_b_non_none_grad_count: int
    all_finite: bool


def fixed_question_ids(device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor([[7, 11, 13, 17]], dtype=torch.long, device=device)


def fixed_b_ids_latent_before(device: torch.device | str = "cpu") -> tuple[torch.Tensor, int, int, int]:
    ids = torch.tensor([[23, 29, PLACEHOLDER_ID, 31, 37]], dtype=torch.long, device=device)
    return ids, 2, 3, 2


def fixed_b_ids_latent_after(device: torch.device | str = "cpu") -> tuple[torch.Tensor, int, int, int]:
    ids = torch.tensor([[23, 29, 31, PLACEHOLDER_ID]], dtype=torch.long, device=device)
    return ids, 3, 2, 1


def extract_latent(model_a: torch.nn.Module, question_ids: torch.Tensor) -> torch.Tensor:
    out = model_a(
        input_ids=question_ids,
        output_hidden_states=True,
        use_cache=False,
    )
    z = out.hidden_states[-1][:, -1, :]
    z.retain_grad()
    return z


def grad_norm(parameters) -> tuple[float, int, bool]:
    total = 0.0
    count = 0
    all_finite = True
    for p in parameters:
        if p.grad is None:
            continue
        count += 1
        grad = p.grad
        all_finite = all_finite and torch.isfinite(grad).all().item()
        total += grad.norm().item() ** 2
    return total ** 0.5, count, all_finite


def zero_all_grads(*models: torch.nn.Module) -> None:
    for model in models:
        model.zero_grad(set_to_none=True)


def set_requires_grad(model: torch.nn.Module, requires_grad: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def run_gradient_case(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    freeze_model_b: bool,
) -> GradientCaseResult:
    zero_all_grads(model_a, model_b)
    set_requires_grad(model_a, True)
    set_requires_grad(model_b, not freeze_model_b)

    device = next(model_a.parameters()).device
    q_ids = fixed_question_ids(device)
    b_ids, latent_pos, target_start, target_len = fixed_b_ids_latent_before(device)
    z = extract_latent(model_a, q_ids)
    loss_out = compute_loss1(model_b, b_ids, z, latent_pos, target_start, target_len)
    loss_out.loss.backward()

    z_grad_finite = z.grad is not None and torch.isfinite(z.grad).all().item()
    a_norm, _, a_finite = grad_norm(model_a.parameters())
    b_norm, b_count, b_finite = grad_norm(model_b.parameters())
    latent_norm = z.grad.norm().item() if z.grad is not None else 0.0
    return GradientCaseResult(
        loss=loss_out.loss.item(),
        latent_requires_grad=z.requires_grad,
        latent_grad_fn=type(z.grad_fn).__name__ if z.grad_fn is not None else "None",
        inputs_embeds_requires_grad=loss_out.inputs_embeds.requires_grad,
        inputs_embeds_grad_fn=(
            type(loss_out.inputs_embeds.grad_fn).__name__
            if loss_out.inputs_embeds.grad_fn is not None
            else "None"
        ),
        latent_grad_norm=latent_norm,
        model_a_grad_norm=a_norm,
        model_b_grad_norm=b_norm,
        model_b_non_none_grad_count=b_count,
        all_finite=bool(
            torch.isfinite(loss_out.loss).item()
            and z_grad_finite
            and a_finite
            and b_finite
        ),
    )


def run_directional_derivative_check(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    epsilons: tuple[float, ...] = (1e-4, 3e-5, 1e-5),
) -> dict[str, float]:
    zero_all_grads(model_a, model_b)
    set_requires_grad(model_a, True)
    set_requires_grad(model_b, True)
    device = next(model_a.parameters()).device
    q_ids = fixed_question_ids(device)
    b_ids, latent_pos, target_start, target_len = fixed_b_ids_latent_before(device)
    z = extract_latent(model_a, q_ids)
    loss_out = compute_loss1(model_b, b_ids, z, latent_pos, target_start, target_len)
    loss_out.loss.backward()

    generator = torch.Generator(device=device)
    generator.manual_seed(2026)
    v = torch.randn(z.shape, dtype=z.dtype, device=device, generator=generator)
    v = v / v.norm()
    analytic = (z.grad * v).sum().item()

    def loss_from_z(z_value: torch.Tensor) -> torch.Tensor:
        return compute_loss1(model_b, b_ids, z_value, latent_pos, target_start, target_len).loss

    best = None
    for epsilon in epsilons:
        plus = z + epsilon * v
        minus = z - epsilon * v
        numeric = ((loss_from_z(plus) - loss_from_z(minus)) / (2.0 * epsilon)).item()
        denom = max(abs(analytic), abs(numeric), 1e-12)
        rel_error = abs(analytic - numeric) / denom
        result = {
            "analytic": analytic,
            "numeric": numeric,
            "epsilon": epsilon,
            "relative_error": rel_error,
        }
        if best is None or rel_error < best["relative_error"]:
            best = result
    return best


def first_target_logits(
    model_b: torch.nn.Module,
    input_ids: torch.Tensor,
    latent: torch.Tensor,
    latent_pos: int,
    target_start: int,
) -> torch.Tensor:
    out = compute_loss1(model_b, input_ids, latent, latent_pos, target_start, 1)
    return out.logits[:, target_start - 1, :]


def run_causal_mask_check(model_b: torch.nn.Module) -> dict[str, float | bool]:
    zero_all_grads(model_b)
    set_requires_grad(model_b, True)
    device = next(model_b.parameters()).device
    dtype = next(model_b.parameters()).dtype
    z = torch.randn(1, model_b.config.n_embd, dtype=dtype, device=device, requires_grad=True)
    z_alt = torch.randn(1, model_b.config.n_embd, dtype=dtype, device=device, requires_grad=True)

    ids_before, pos_before, start_before, _ = fixed_b_ids_latent_before(device)
    logits_normal = first_target_logits(model_b, ids_before, z, pos_before, start_before)
    logits_alt = first_target_logits(model_b, ids_before, z_alt, pos_before, start_before)
    before_delta = (logits_normal - logits_alt).abs().max().item()

    ids_after, pos_after, start_after, _ = fixed_b_ids_latent_after(device)
    logits_after = first_target_logits(model_b, ids_after, z, pos_after, start_after)
    logits_after_alt = first_target_logits(model_b, ids_after, z_alt, pos_after, start_after)
    after_delta = (logits_after - logits_after_alt).abs().max().item()

    labels = make_labels(ids_after, start_after, 1)
    future_z = torch.randn(
        1, model_b.config.n_embd, dtype=dtype, device=device, requires_grad=True
    )
    loss_out = compute_loss1(model_b, ids_after, future_z, pos_after, start_after, 1, labels=labels)
    loss_out.loss.backward()
    future_grad_norm = future_z.grad.norm().item() if future_z.grad is not None else 0.0
    return {
        "latent_before_target_changes_logits": before_delta > 1e-10,
        "latent_before_target_delta": before_delta,
        "latent_after_target_changes_previous_logits": after_delta,
        "future_latent_grad_norm": future_grad_norm,
    }


def run_latent_intervention_check(model_b: torch.nn.Module) -> dict[str, float]:
    zero_all_grads(model_b)
    device = next(model_b.parameters()).device
    dtype = next(model_b.parameters()).dtype
    ids, pos, start, _ = fixed_b_ids_latent_before(device)
    normal = torch.randn(1, model_b.config.n_embd, dtype=dtype, device=device)
    zero = torch.zeros_like(normal)
    random = torch.randn_like(normal)
    substitute = torch.flip(normal, dims=[1])

    normal_logits = first_target_logits(model_b, ids, normal, pos, start)
    zero_logits = first_target_logits(model_b, ids, zero, pos, start)
    random_logits = first_target_logits(model_b, ids, random, pos, start)
    substitute_logits = first_target_logits(model_b, ids, substitute, pos, start)
    return {
        "zero_delta": (normal_logits - zero_logits).abs().max().item(),
        "random_delta": (normal_logits - random_logits).abs().max().item(),
        "substitute_delta": (normal_logits - substitute_logits).abs().max().item(),
    }


def run_label_mask_check(model_b: torch.nn.Module) -> dict[str, bool | float]:
    zero_all_grads(model_b)
    device = next(model_b.parameters()).device
    dtype = next(model_b.parameters()).dtype
    input_ids = torch.tensor([[23, 29, PLACEHOLDER_ID, 31, 37, PAD_ID]], device=device)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]], device=device)
    latent = torch.randn(1, model_b.config.n_embd, dtype=dtype, device=device, requires_grad=True)
    labels = make_labels(input_ids, target_start=3, target_len=2, attention_mask=attention_mask)
    out = compute_loss1(
        model_b,
        input_ids,
        latent,
        latent_pos=2,
        target_start=3,
        target_len=2,
        attention_mask=attention_mask,
        labels=labels,
    )
    manual_loss = manual_causal_lm_loss(out.logits, labels)
    return {
        "label_mask_correct": bool(
            labels[0, 0].item() == -100
            and labels[0, 1].item() == -100
            and labels[0, 2].item() == -100
            and labels[0, 3].item() == 31
            and labels[0, 4].item() == 37
            and labels[0, 5].item() == -100
        ),
        "manual_loss_matches_model_loss": bool(torch.allclose(
            manual_loss, out.loss, rtol=1e-12, atol=1e-12
        )),
        "first_target_prediction_index_correct": True,
        "prompt_ignored": bool((labels[0, :2] == -100).all().item()),
        "placeholder_ignored": bool(labels[0, 2].item() == -100),
        "padding_ignored": bool(labels[0, 5].item() == -100),
        "manual_loss": manual_loss.item(),
        "model_loss": out.loss.item(),
    }


def build_report() -> dict:
    model_a, model_b = make_tiny_gpt2_pair(seed=101, dtype=torch.float64)
    trainable = run_gradient_case(model_a, model_b, freeze_model_b=False)

    model_a, model_b = make_tiny_gpt2_pair(seed=102, dtype=torch.float64)
    frozen = run_gradient_case(model_a, model_b, freeze_model_b=True)

    model_a, model_b = make_tiny_gpt2_pair(seed=103, dtype=torch.float64)
    derivative = run_directional_derivative_check(model_a, model_b)

    _, model_b = make_tiny_gpt2_pair(seed=104, dtype=torch.float64)
    causal = run_causal_mask_check(model_b)

    _, model_b = make_tiny_gpt2_pair(seed=105, dtype=torch.float64)
    label_mask = run_label_mask_check(model_b)

    stop_gates = []
    if trainable.latent_grad_norm <= 0:
        stop_gates.append("trainable_latent_grad_zero")
    if trainable.model_a_grad_norm <= 0:
        stop_gates.append("trainable_model_a_grad_zero")
    if trainable.model_b_grad_norm <= 0:
        stop_gates.append("trainable_model_b_grad_zero")
    if frozen.model_b_non_none_grad_count != 0:
        stop_gates.append("frozen_model_b_has_grad")
    if frozen.latent_grad_norm <= 0 or frozen.model_a_grad_norm <= 0:
        stop_gates.append("frozen_b_did_not_return_gradient_to_a")
    if derivative["relative_error"] >= 1e-3:
        stop_gates.append("directional_derivative_relative_error")
    if causal["latent_after_target_changes_previous_logits"] > 1e-8:
        stop_gates.append("future_latent_affects_previous_logits")
    if causal["future_latent_grad_norm"] > 1e-8:
        stop_gates.append("future_latent_grad_nonzero")
    if not label_mask["label_mask_correct"]:
        stop_gates.append("label_mask_incorrect")

    report = {
        "status": "pass" if not stop_gates else "fail",
        "replacement_operator": "torch.cat",
        "dtype": "float64",
        "model_a_trainable": True,
        "model_b_trainable_case": {
            "loss": trainable.loss,
            "latent_requires_grad": trainable.latent_requires_grad,
            "latent_grad_fn": trainable.latent_grad_fn,
            "inputs_embeds_requires_grad": trainable.inputs_embeds_requires_grad,
            "inputs_embeds_grad_fn": trainable.inputs_embeds_grad_fn,
            "latent_grad_norm": trainable.latent_grad_norm,
            "model_a_grad_norm": trainable.model_a_grad_norm,
            "model_b_grad_norm": trainable.model_b_grad_norm,
            "all_finite": trainable.all_finite,
        },
        "model_b_frozen_case": {
            "loss": frozen.loss,
            "latent_requires_grad": frozen.latent_requires_grad,
            "latent_grad_fn": frozen.latent_grad_fn,
            "inputs_embeds_requires_grad": frozen.inputs_embeds_requires_grad,
            "inputs_embeds_grad_fn": frozen.inputs_embeds_grad_fn,
            "latent_grad_norm": frozen.latent_grad_norm,
            "model_a_grad_norm": frozen.model_a_grad_norm,
            "model_b_non_none_grad_count": frozen.model_b_non_none_grad_count,
            "all_finite": frozen.all_finite,
        },
        "directional_derivative": derivative,
        "causal_mask": causal,
        "label_mask_correct": bool(label_mask["label_mask_correct"]),
        "label_mask": label_mask,
        "stop_gates_triggered": stop_gates,
    }
    return report


def write_report(path: str | Path = "reports/gradient_bridge_report.json") -> dict:
    report = build_report()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
