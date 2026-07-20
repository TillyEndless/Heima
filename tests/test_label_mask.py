import torch

from src.gradient_bridge.gradient_audit import run_label_mask_check
from src.gradient_bridge.tiny_models import make_tiny_gpt2_pair


def test_label_mask_ignores_prompt_placeholder_and_padding():
    _, model_b = make_tiny_gpt2_pair(seed=41, dtype=torch.float64)

    result = run_label_mask_check(model_b)

    assert result["label_mask_correct"]
    assert result["manual_loss_matches_model_loss"]
    assert result["first_target_prediction_index_correct"]
    assert result["prompt_ignored"]
    assert result["placeholder_ignored"]
    assert result["padding_ignored"]
