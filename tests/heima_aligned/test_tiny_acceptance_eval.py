from src.heima_aligned.tiny_acceptance_eval import (
    build_intervention_eval_plan,
    build_reasoning_token_diagnostics,
    build_tiny_acceptance_eval_manifest,
    build_warm_b_interface,
    generation_exact_match,
)


def test_reasoning_token_diagnostics_marks_semantic_tokens():
    diag = build_reasoning_token_diagnostics(
        "The chart shows South Sudan peaked at 379.85% in 2016, so the answer is 2016.",
        "2016",
    )
    assert diag.counts()["numeric_tokens"] >= 2
    assert diag.counts()["entity_tokens"] >= 2
    assert diag.counts()["answer_tokens"] >= 1
    assert diag.counts()["content_tokens"] > diag.counts()["answer_tokens"]


def test_generation_exact_match_normalizes_lightly():
    assert generation_exact_match("  2016. ", "2016")
    assert not generation_exact_match("2017", "2016")


def test_intervention_plan_keeps_prompt_and_target_fixed():
    plan = build_intervention_eval_plan("reasoning")
    assert plan["keep_fixed"] == ["question", "decoder_prompt", "teacher_target", "attention_mask", "labels"]
    assert set(plan["interventions"]) == {"q_only", "correct", "shuffle", "zero"}
    assert plan["vary_only"] == "injected_latent"


def test_warm_b_interface_compares_cold_and_warm_joint():
    iface = build_warm_b_interface()
    assert iface["stage_after"] == "freeze_A_train_B"
    assert iface["comparisons"] == ["cold_b_joint", "warm_b_joint"]
    assert "Model A checkpoint" in iface["invariant_between_comparisons"]


def test_eval_manifest_is_mechanism_not_benchmark():
    manifest = build_tiny_acceptance_eval_manifest({"reasoning": "Use 3 and 4 to get 7.", "answer": "7"})
    assert manifest["not_a_benchmark_reproduction"] is True
    assert "reasoning_token_level_evaluation" in manifest
    assert "warm_b_evaluation_interface" in manifest
