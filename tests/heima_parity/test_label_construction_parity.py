from __future__ import annotations

import torch

from src.htext.modeling import build_h0_labels, build_h1_labels


def test_label_construction_main_and_loss1_boundaries():
    h0 = build_h0_labels(
        total_len=7,
        question_len=2,
        num_thinking_tokens=1,
        answer_prefix_len=2,
        answer_ids=torch.tensor([8, 9]),
        thinking_id=5,
    )
    assert h0.tolist() == [-100, -100, 5, -100, -100, 8, 9]
    h1 = build_h1_labels(total_len=6, target_start=3, target_ids=torch.tensor([10, 11, 12]))
    assert h1.tolist() == [-100, -100, -100, 10, 11, 12]
    assert (h0 != -100).sum().item() == 3
    assert (h1 != -100).sum().item() == 3

