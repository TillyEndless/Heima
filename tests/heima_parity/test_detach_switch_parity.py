from __future__ import annotations

import torch

from src.htext.heima_reuse import prepare_latent_for_decoder


def test_detach_switch_is_only_gradient_difference():
    torch.manual_seed(0)
    z = torch.randn(2, 3, requires_grad=True)
    weight = torch.randn(3, 1, requires_grad=True)

    detached = prepare_latent_for_decoder(z, True)
    loss_detached = (detached @ weight).sum()
    grad_detached = torch.autograd.grad(loss_detached, z, allow_unused=True)[0]
    assert grad_detached is None

    attached = prepare_latent_for_decoder(z, False)
    loss_attached = (attached @ weight).sum()
    grad_attached = torch.autograd.grad(loss_attached, z)[0]
    assert grad_attached.norm().item() > 0

