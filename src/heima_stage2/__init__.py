"""Strict Heima Stage1/Stage2 comparison helpers."""

from .interp_supervision import (
    Stage2Mode,
    Stage2StepOutput,
    assert_teacher_interpreters_frozen,
    compute_grad_norm,
    freeze_teacher_interpreters,
    run_stage2_train_step,
)

__all__ = [
    "Stage2Mode",
    "Stage2StepOutput",
    "assert_teacher_interpreters_frozen",
    "compute_grad_norm",
    "freeze_teacher_interpreters",
    "run_stage2_train_step",
    "Loss2Features",
    "compute_loss2_grad_norm",
    "loss2_forward",
    "pool_target_hidden",
]

from .loss2_alignment import (
    Loss2Features,
    compute_grad_norm as compute_loss2_grad_norm,
    loss2_forward,
    pool_target_hidden,
)
