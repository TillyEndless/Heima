import torch

from src.g1.whole_cot_decoder import build_loss1_labels


def test_loss1_labels_mask_prompt_latent_and_padding():
    labels = build_loss1_labels(
        total_len=7,
        prompt_len=3,
        latent_len=1,
        cot_ids=torch.tensor([51, 52, 53]),
        pad_to=9,
    )

    assert labels.tolist() == [-100, -100, -100, -100, 51, 52, 53, -100, -100]

