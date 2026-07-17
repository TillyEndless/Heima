from __future__ import annotations

import math

import torch


def grad_norm(parameters) -> tuple[float, bool]:
    total = 0.0
    finite = True
    for p in parameters:
        if p.grad is None:
            continue
        finite = finite and torch.isfinite(p.grad).all().item()
        total += p.grad.float().norm().item() ** 2
    return math.sqrt(total), bool(finite)


def tensor_norm(tensor: torch.Tensor | None) -> float:
    if tensor is None:
        return 0.0
    return tensor.float().norm().item()


def finite_ratio(tensors: list[torch.Tensor | None]) -> float:
    total = 0
    finite = 0
    for tensor in tensors:
        if tensor is None:
            continue
        total += tensor.numel()
        finite += torch.isfinite(tensor).sum().item()
    return 1.0 if total == 0 else finite / total


def cosine(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if a is None or b is None:
        return None
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    denom = af.norm() * bf.norm()
    if denom.item() == 0:
        return None
    return torch.dot(af, bf).div(denom).item()
