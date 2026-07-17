from __future__ import annotations

import torch

from src.htext.modeling import LatentProjector


def test_projector_shape_and_grad():
    projector = LatentProjector(hidden_size=4)
    z = torch.randn(2, 4, requires_grad=True)
    out = projector(z)
    assert out.shape == z.shape
    out.pow(2).sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert projector.linear.weight.grad is not None

