from __future__ import annotations

import torch

from src.htext.heima_reuse import heima_shifted_thinking_mask


def test_hidden_extraction_parity_with_official_mask():
    torch.manual_seed(0)
    thinking_id = 7
    tokens = torch.tensor([[2, 3, thinking_id, 4], [5, thinking_id, 6, 0]])
    hidden = torch.randn(2, 4, 5)
    mask = heima_shifted_thinking_mask(tokens, thinking_id)
    official = hidden[mask].view(2, 1, 5)
    manual = torch.stack([hidden[0, 1], hidden[1, 0]], dim=0).unsqueeze(1)
    assert torch.max(torch.abs(official - manual)).item() < 1e-6

