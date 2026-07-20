from __future__ import annotations

import torch

from src.htext.heima_reuse import heima_shifted_thinking_mask


def test_heima_shifted_thinking_mask_selects_previous_position():
    thinking_id = 7
    tokens = torch.tensor([[11, 12, thinking_id, 13], [21, thinking_id, 22, 23]])
    mask = heima_shifted_thinking_mask(tokens, thinking_id)
    assert mask.tolist() == [[False, True, False, False], [True, False, False, False]]

