from __future__ import annotations

import torch
from torch import nn

from src.heima_stage2 import Stage2Mode, assert_teacher_interpreters_frozen, run_stage2_train_step


class ToyA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4, bias=False)
        self.head = nn.Linear(4, 1, bias=False)

    def forward(self, batch):
        z = self.encoder(batch["x"])
        pred = self.head(z)
        ntp_loss = (pred - batch["y"]).pow(2).mean()
        return {"ntp_loss": ntp_loss, "latents": {"reasoning": z}}


class ToyFrozenB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 1, bias=False)

    def forward(self, z, batch):
        pred = self.proj(z)
        return (pred - batch["cot_y"]).pow(2).mean()


def batch():
    torch.manual_seed(7)
    return {
        "x": torch.randn(3, 4),
        "y": torch.randn(3, 1),
        "cot_y": torch.randn(3, 1),
    }


def make_models():
    torch.manual_seed(11)
    model_a = ToyA()
    model_b = ToyFrozenB()
    opt = torch.optim.SGD(model_a.parameters(), lr=0.01)
    return model_a, model_b, opt


def test_freeze_b_teacher() -> None:
    model_a, model_b, opt = make_models()
    out = run_stage2_train_step(
        model_a=model_a,
        interpreters={"reasoning": model_b},
        optimizer_a=opt,
        batch=batch(),
        mode=Stage2Mode.HEIMA_BASELINE,
        lambda_interp=1.0,
    )
    assert out.teacher_B_frozen
    assert_teacher_interpreters_frozen([model_b])
    assert all(not p.requires_grad for p in model_b.parameters())
    b_param_ids = {id(bp) for bp in model_b.parameters()}
    assert all(id(p) not in b_param_ids for group in opt.param_groups for p in group["params"])


def test_grad_a_from_interp_zero_for_heima_baseline() -> None:
    model_a, model_b, opt = make_models()
    out = run_stage2_train_step(
        model_a=model_a,
        interpreters={"reasoning": model_b},
        optimizer_a=opt,
        batch=batch(),
        mode=Stage2Mode.HEIMA_BASELINE,
        lambda_interp=1.0,
    )
    assert out.grad_A_from_interp == 0.0
    assert out.grad_B_from_interp == 0.0


def test_grad_a_from_interp_nonzero_for_ours() -> None:
    model_a, model_b, opt = make_models()
    out = run_stage2_train_step(
        model_a=model_a,
        interpreters={"reasoning": model_b},
        optimizer_a=opt,
        batch=batch(),
        mode=Stage2Mode.OURS_INTERP_SUPERVISION,
        lambda_interp=0.5,
    )
    assert out.grad_A_from_interp > 0.0
    assert torch.isfinite(torch.tensor(out.grad_A_from_interp))
    assert out.grad_B_from_interp == 0.0
