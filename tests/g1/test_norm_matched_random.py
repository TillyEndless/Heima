import torch

from src.g1.sample_specific_audit import make_conditions


def test_norm_matched_random_matches_row_norms():
    z = torch.arange(1, 13, dtype=torch.float32).view(3, 4)
    conditions, metadata = make_conditions(z)
    assert torch.allclose(
        conditions["norm_matched_random"].norm(dim=1),
        z.norm(dim=1),
        rtol=1e-5,
        atol=1e-5,
    )
    assert metadata["cyclic_has_fixed_point"] is False
