from src.heima_aligned.protocol import mode_plan


def test_aonly_self_loss1_has_no_external_b_during_training():
    plan = mode_plan("aonly_self_loss1")
    train_steps = [p for p in plan if p["stage"].startswith("ours_")]
    assert train_steps
    assert all(p.get("external_b") is False for p in train_steps)
    assert all(p.get("self_decoder") is True for p in train_steps)
    assert all(p.get("loss1") is True for p in train_steps)


def test_aonly_adds_eval_only_interpreters_for_fair_comparison():
    plan = mode_plan("aonly_self_loss1")
    eval_b = [p for p in plan if p.get("eval_only_interpreter")]
    assert [p["section"] for p in eval_b] == ["summary", "caption", "reasoning"]
    assert all(p.get("train_a") is False and p.get("train_b") is True for p in eval_b)


def test_aonly_compute_matched_disables_loss1():
    plan = mode_plan("aonly_compute_matched_main_only")
    assert all(p.get("lambda_loss1") == 0.0 for p in plan)
    assert all(p.get("loss1") is False for p in plan)
