from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_cosine(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1) @ F.normalize(x.float(), dim=-1).T


def farthest_indices(x: torch.Tensor) -> list[int]:
    sims = pairwise_cosine(x)
    sims.fill_diagonal_(float("inf"))
    return sims.argmin(dim=1).tolist()


def effective_rank(x: torch.Tensor) -> float:
    centered = x.float() - x.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    probs = singular / singular.sum().clamp_min(1e-12)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    return torch.exp(entropy).item()


def recall_at_k(scores: torch.Tensor, k: int) -> float:
    topk = scores.topk(k, dim=1).indices
    target = torch.arange(scores.size(0), device=scores.device).unsqueeze(1)
    return topk.eq(target).any(dim=1).float().mean().item()
