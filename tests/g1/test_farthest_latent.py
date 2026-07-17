import torch

from src.g1.latent_retrieval import farthest_indices


def test_farthest_latent_excludes_self():
    z = torch.eye(4)
    far = farthest_indices(z)
    assert len(far) == 4
    assert all(i != j for i, j in enumerate(far))

