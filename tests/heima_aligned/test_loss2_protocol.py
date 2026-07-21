from src.heima_aligned.protocol import mode_plan


def test_loss2_modes_have_frozen_teacher():
    for mode in ["ours_warm_b_fixed_loss1_loss2", "ours_warm_b_joint_loss1_loss2", "ours_cold_b_joint_loss1_loss2", "main_loss1_loss2"]:
        plan = mode_plan(mode)
        assert plan
        assert all(step.get("loss2") is True for step in plan)
        assert all(step.get("teacher_frozen") is True for step in plan)


def test_compute_matched_loss1_only_disables_loss2():
    plan = mode_plan("main_loss1_only")
    assert all(step.get("loss1") is True for step in plan)
    assert all(step.get("loss2") is False for step in plan)
    assert all(step.get("lambda_loss2") == 0.0 for step in plan)


def test_warm_fixed_loss2_keeps_b_frozen_but_differentiable():
    plan = mode_plan("ours_warm_b_fixed_loss1_loss2")
    assert all(step.get("train_a") is True for step in plan)
    assert all(step.get("train_b") is False for step in plan)
    assert all(step.get("b_frozen_differentiable") is True for step in plan)
