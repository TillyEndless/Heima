from src.g1.token_category_metrics import cyclic_derangement


def test_cyclic_derangement_has_no_fixed_points():
    perm = cyclic_derangement(32)
    assert sorted(perm) == list(range(32))
    assert all(i != j for i, j in enumerate(perm))

