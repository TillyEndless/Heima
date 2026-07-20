import torch

from src.gradient_bridge.embedding_injection import inject_single_latent


def test_inject_single_latent_uses_cat_and_preserves_gradient():
    token_embeds = torch.randn(1, 5, 4, dtype=torch.float64)
    latent = torch.randn(1, 4, dtype=torch.float64, requires_grad=True)

    injected = inject_single_latent(token_embeds, latent, latent_pos=2)

    assert injected.shape == token_embeds.shape
    assert torch.allclose(injected[:, :2], token_embeds[:, :2])
    assert torch.allclose(injected[:, 2], latent)
    assert torch.allclose(injected[:, 3:], token_embeds[:, 3:])
    assert injected.requires_grad
    assert injected.grad_fn is not None

    injected[:, 2].sum().backward()
    assert latent.grad is not None
    assert torch.all(latent.grad == 1)


def test_inject_single_latent_rejects_bad_shapes():
    token_embeds = torch.randn(1, 5, 4, dtype=torch.float64)
    latent = torch.randn(1, 1, 4, dtype=torch.float64)

    try:
        inject_single_latent(token_embeds, latent, latent_pos=2)
    except ValueError as exc:
        assert "latent" in str(exc)
    else:
        raise AssertionError("bad latent shape should fail")
