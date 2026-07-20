import torch

from src.g1.latent_reasoner import build_main_labels


def test_main_labels_mask_question_latent_prefix_and_padding():
    labels = build_main_labels(
        total_len=8,
        question_len=3,
        latent_len=1,
        answer_prefix_len=2,
        answer_ids=torch.tensor([41, 42]),
        pad_to=10,
    )

    assert labels.tolist() == [-100, -100, -100, -100, -100, -100, 41, 42, -100, -100]

