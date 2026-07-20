import torch

from src.gradient_bridge.gradient_audit import run_gradient_case
from src.gradient_bridge.tiny_models import make_tiny_gpt2_pair


def test_model_b_trainable_returns_gradients_to_a_and_b():
    model_a, model_b = make_tiny_gpt2_pair(seed=11, dtype=torch.float64)

    result = run_gradient_case(model_a, model_b, freeze_model_b=False)

    assert result.loss > 0
    assert result.latent_requires_grad
    assert result.latent_grad_norm > 0
    assert result.model_a_grad_norm > 0
    assert result.model_b_grad_norm > 0
    assert result.model_b_non_none_grad_count > 0
    assert result.all_finite


def test_model_b_frozen_still_returns_gradients_to_a():
    model_a, model_b = make_tiny_gpt2_pair(seed=12, dtype=torch.float64)

    result = run_gradient_case(model_a, model_b, freeze_model_b=True)

    assert result.loss > 0
    assert result.latent_requires_grad
    assert result.latent_grad_norm > 0
    assert result.model_a_grad_norm > 0
    assert result.model_b_grad_norm == 0
    assert result.model_b_non_none_grad_count == 0
    assert result.all_finite


def test_model_a_and_model_b_do_not_share_parameter_objects():
    model_a, model_b = make_tiny_gpt2_pair(seed=13, dtype=torch.float64)

    ids_a = {id(p) for p in model_a.parameters()}
    ids_b = {id(p) for p in model_b.parameters()}

    assert ids_a.isdisjoint(ids_b)
