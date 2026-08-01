import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "qwen7b_stage1_repair.py"
spec = importlib.util.spec_from_file_location("repair", SCRIPT)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def test_answer_near_limit_kept_in_strict_mode():
    plan = repair.plan_stage1_lengths(20, 32, 64, max_q=64, max_latent=64, max_answer=64, max_seq=116, strict=True)
    assert plan["keep"]
    assert not plan["answer_truncated"]
    assert plan["sequence_length"] == 116


def test_raw_k_over_max_latent_filtered_in_strict_mode():
    plan = repair.plan_stage1_lengths(20, 65, 20, max_q=64, max_latent=64, max_answer=64, max_seq=200, strict=True)
    assert not plan["keep"]
    assert "raw_K_exceeds_max_latent" in plan["filter_reason"]
    assert plan["latent_cap_hit"]


def test_question_too_long_filtered_in_strict_mode():
    plan = repair.plan_stage1_lengths(65, 16, 20, max_q=64, max_latent=64, max_answer=64, max_seq=200, strict=True)
    assert not plan["keep"]
    assert "question_too_long" in plan["filter_reason"]


def test_cot_too_long_filtered_via_latent_budget():
    plan = repair.plan_stage1_lengths(20, 100, 20, max_q=64, max_latent=64, max_answer=64, max_seq=200, strict=True)
    assert not plan["keep"]
    assert "raw_K_exceeds_max_latent" in plan["filter_reason"]


def test_boundary_exactly_at_limit_is_valid():
    plan = repair.plan_stage1_lengths(10, 10, 10, max_q=10, max_latent=10, max_answer=10, max_seq=30, strict=True)
    assert plan["keep"]
    assert plan["sequence_length"] == 30


def test_non_strict_caps_latent_but_does_not_silently_claim_clean():
    plan = repair.plan_stage1_lengths(20, 80, 20, max_q=64, max_latent=64, max_answer=64, max_seq=200, strict=False)
    assert plan["keep"]
    assert plan["latent_cap_hit"]
    assert plan["used_K"] == 64
