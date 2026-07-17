from __future__ import annotations

import torch
import torch.nn.functional as F

from src.htext.heima_reuse import heima_ce_loss, hf_shifted_ce_loss


def test_loss_value_and_gradient_parity():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, -100, 3], [-100, 4, 5, 6, -100]])
    actual = heima_ce_loss(logits, labels)
    expected = hf_shifted_ce_loss(logits, labels)
    rel = abs(actual.item() - expected.item()) / max(abs(expected.item()), 1e-12)
    assert rel < 1e-6
    actual.backward(retain_graph=True)
    actual_grad = logits.grad.detach().clone()
    logits.grad.zero_()
    expected.backward()
    denom = logits.grad.detach().norm().item()
    grad_rel = (actual_grad - logits.grad).norm().item() / max(denom, 1e-12)
    assert grad_rel < 1e-5


def test_loss_rejects_empty_target():
    logits = torch.randn(1, 3, 5)
    labels = torch.full((1, 3), -100)
    try:
        heima_ce_loss(logits, labels)
    except ValueError as exc:
        assert "non-ignored" in str(exc)
    else:
        raise AssertionError("empty target should raise")

