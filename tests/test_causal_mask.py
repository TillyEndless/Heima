import torch

from src.gradient_bridge.gradient_audit import (
    run_causal_mask_check,
    run_latent_intervention_check,
)
from src.gradient_bridge.tiny_models import make_tiny_gpt2_pair


def test_latent_before_target_changes_target_logits_but_future_latent_does_not():
    _, model_b = make_tiny_gpt2_pair(seed=31, dtype=torch.float64)

    result = run_causal_mask_check(model_b)

    assert result["latent_before_target_changes_logits"]
    assert result["latent_after_target_changes_previous_logits"] <= 1e-8
    assert result["future_latent_grad_norm"] <= 1e-8


def test_latent_interventions_change_first_target_logits():
    _, model_b = make_tiny_gpt2_pair(seed=32, dtype=torch.float64)

    result = run_latent_intervention_check(model_b)

    assert result["zero_delta"] > 1e-10
    assert result["random_delta"] > 1e-10
    assert result["substitute_delta"] > 1e-10
