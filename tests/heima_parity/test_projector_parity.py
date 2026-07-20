from __future__ import annotations

import torch

from src.htext.heima_reuse import OfficialCompatibleAbstractProjection, official_projector_spec


def test_projector_official_shape_and_grad():
    torch.manual_seed(0)
    projector = OfficialCompatibleAbstractProjection(4, 6)
    z = torch.randn(3, 4, requires_grad=True)
    out = projector(z)
    assert out.shape == (3, 6)
    out.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    spec = official_projector_spec(4, 6)
    assert spec["layer_order"] == ["Linear", "ReLU(inplace=True)", "Linear", "Dropout(p=0.0)"]
    assert spec["normalization"] is None

