from __future__ import annotations

import torch

from src.htext.modeling import build_h0_labels, build_h1_labels


def test_h0_labels_supervise_thinking_and_answer_only():
    labels = build_h0_labels(
        total_len=8,
        question_len=3,
        num_thinking_tokens=2,
        answer_prefix_len=1,
        answer_ids=torch.tensor([9, 10]),
        thinking_id=7,
    )
    assert labels.tolist() == [-100, -100, -100, 7, 7, -100, 9, 10]


def test_h1_labels_supervise_target_only():
    labels = build_h1_labels(
        total_len=7,
        target_start=4,
        target_ids=torch.tensor([11, 12, 13]),
    )
    assert labels.tolist() == [-100, -100, -100, -100, 11, 12, 13]

