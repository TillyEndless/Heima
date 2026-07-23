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
    "AOnlySelfDecodeMode",
    "AOnlyStepOutput",
    "FirstPassOutput",
    "SelfDecodeFeatures",
    "build_self_decode_features",
    "evaluate_self_decode_interventions",
    "run_a_only_train_step",
    "self_decode_forward",
]

from .model_a_only_self_decode import (
    AOnlySelfDecodeMode,
    AOnlyStepOutput,
    FirstPassOutput,
    SelfDecodeFeatures,
    build_self_decode_features,
    evaluate_self_decode_interventions,
    run_a_only_train_step,
    self_decode_forward,
)
