import torch

from src.g1.latent_reasoner import assert_parameter_independence
from src.g1.whole_cot_decoder import replace_latent_with_cat


def test_replace_latent_with_cat_keeps_gradient_path():
    token_embeds = torch.randn(2, 5, 3)
    latent = torch.randn(2, 3, requires_grad=True)

    injected = replace_latent_with_cat(token_embeds, latent, latent_pos=2)
    injected[:, 2, :].sum().backward()

    assert latent.grad is not None
    assert torch.all(latent.grad == 1)


def test_distinct_parameter_objects_for_real_modules():
    a = torch.nn.Linear(3, 3)
    b = torch.nn.Linear(3, 3)
    assert_parameter_independence(a, b)
