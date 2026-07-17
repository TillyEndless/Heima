from __future__ import annotations

import torch

from src.htext.heima_reuse import direct_thinking_mask, heima_shifted_thinking_mask


def test_thinking_mask_matches_official_shift_formula():
    torch.manual_seed(0)
    thinking_id = 9
    tokens = torch.tensor([[1, 2, thinking_id, 4, 0], [5, thinking_id, 7, 8, 0]])
    expected = torch.tensor([[False, True, False, False, False], [True, False, False, False, False]])
    assert torch.equal(heima_shifted_thinking_mask(tokens, thinking_id), expected)
    assert torch.equal(direct_thinking_mask(tokens, thinking_id), tokens == thinking_id)

