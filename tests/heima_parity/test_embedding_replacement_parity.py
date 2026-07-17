from __future__ import annotations

import torch

from src.htext.heima_reuse import official_embedding_replacement


def _reference_replacement(token_embeds, thinking_token, mask):
    out = token_embeds.clone()
    out[mask] = thinking_token.reshape(-1, token_embeds.shape[-1])
    return out


def test_embedding_replacement_forward_and_gradient_parity():
    torch.manual_seed(0)
    token_embeds = torch.randn(2, 4, 3, requires_grad=True)
    latent = torch.randn(2, 1, 3, requires_grad=True)
    mask = torch.tensor([[False, True, False, False], [False, False, True, False]])
    actual = official_embedding_replacement(token_embeds, latent, mask)
    expected = _reference_replacement(token_embeds, latent, mask)
    assert torch.max(torch.abs(actual - expected)).item() < 1e-6
    grad = torch.randn_like(actual)
    actual.backward(grad, retain_graph=True)
    actual_token_grad = token_embeds.grad.detach().clone()
    actual_latent_grad = latent.grad.detach().clone()
    token_embeds.grad.zero_()
    latent.grad.zero_()
    expected.backward(grad)
    assert torch.max(torch.abs(actual_token_grad - token_embeds.grad)).item() < 1e-6
    assert torch.max(torch.abs(actual_latent_grad - latent.grad)).item() < 1e-6

