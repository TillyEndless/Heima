import torch

from src.g1.evaluator import make_intervention_latents


def test_intervention_latents_are_distinct_and_shuffled():
    z = torch.arange(12, dtype=torch.float32).view(3, 4)
    interventions = make_intervention_latents(z)

    assert torch.equal(interventions["normal"], z)
    assert torch.equal(interventions["zero"], torch.zeros_like(z))
    assert interventions["random"].shape == z.shape
    assert torch.equal(interventions["shuffled"], z[[1, 2, 0]])
    assert not torch.equal(interventions["random"], z)

