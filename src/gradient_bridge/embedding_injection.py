import torch


def inject_single_latent(
    token_embeds: torch.Tensor,
    latent: torch.Tensor,
    latent_pos: int,
) -> torch.Tensor:
    if token_embeds.ndim != 3:
        raise ValueError("token_embeds must have shape [batch, seq, hidden]")
    if latent.ndim != 2:
        raise ValueError("latent must have shape [batch, hidden]")
    if token_embeds.shape[0] != latent.shape[0]:
        raise ValueError("token_embeds and latent batch sizes must match")
    if token_embeds.shape[2] != latent.shape[1]:
        raise ValueError("token_embeds hidden size must match latent hidden size")
    if not 0 <= latent_pos < token_embeds.shape[1]:
        raise ValueError("latent_pos is out of range")

    return torch.cat(
        [
            token_embeds[:, :latent_pos, :],
            latent.unsqueeze(1),
            token_embeds[:, latent_pos + 1 :, :],
        ],
        dim=1,
    )

