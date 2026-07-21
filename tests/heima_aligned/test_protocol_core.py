from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch

from src.heima_aligned.protocol import (
    HeimaOfficialAbstractProjection,
    HeimaRecord,
    THINKING_TOKENS,
    build_decoder_labels,
    build_decoder_sample,
    build_encoder_sample,
    build_main_labels,
    build_stage_response,
    decoder_prompt,
    extract_predictor_hidden,
    mode_plan,
    replace_embeddings_after_lookup,
    split_hash,
)


def rec() -> HeimaRecord:
    return HeimaRecord("1", "x.jpg", "What is shown?", "summ", "cap", "reason", "ans")


def test_full_schema_and_hash():
    r = rec()
    assert r.question and r.summary and r.caption and r.reasoning and r.answer
    assert len(split_hash([r])) == 64


def test_progressive_stage_exact_sequence():
    r = rec()
    s0 = build_stage_response(r, "stage_0_explicit")
    s1 = build_stage_response(r, "stage_1_summary")
    s2 = build_stage_response(r, "stage_2_caption")
    s3 = build_stage_response(r, "stage_3_reasoning")
    assert "summ" in s0 and "cap" in s0 and "reason" in s0
    assert THINKING_TOKENS["summary"] in s1 and "cap" in s1 and "reason" in s1
    assert THINKING_TOKENS["summary"] in s2 and THINKING_TOKENS["caption"] in s2 and "reason" in s2
    assert all(THINKING_TOKENS[x] in s3 for x in THINKING_TOKENS)


def test_recover_exact_sequence():
    r = rec()
    recover = build_encoder_sample(r, "stage_4_recover")
    assert recover["replaced_sections"] == ["summary", "caption", "reasoning"]
    assert "ans" in recover["response"]


def test_marker_text_answer_label_mask():
    labels = build_main_labels([10, 11, 12, 13], prompt_len=1)
    assert labels == [-100, 11, 12, 13]
    dec = build_decoder_labels([1, 2, 3], prompt_len=2, train_on_input=True)
    assert dec == [1, 2, 3]
    dec2 = build_decoder_labels([1, 2, 3], prompt_len=2, train_on_input=False)
    assert dec2 == [-100, -100, 3]


def test_no_future_section_leakage_in_decoder_prompt():
    r = rec()
    p = decoder_prompt(r, "summary")
    assert THINKING_TOKENS["summary"] in p
    assert "cap" not in p and "reason" not in p


def test_official_predictor_hidden_extraction():
    input_ids = torch.tensor([[4, 99, 5], [7, 8, 99]])
    h = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    state = extract_predictor_hidden(input_ids, h, 99)
    assert state.thinking_positions.tolist() == [1, 2]
    assert state.selected_positions.tolist() == [0, 1]
    assert torch.equal(state.hidden[0], h[0, 0])
    assert torch.equal(state.hidden[1], h[1, 1])


def test_projector_parity_shape_and_order():
    p = HeimaOfficialAbstractProjection(3, 5)
    names = [m.__class__.__name__ for m in p.net]
    assert names == ["Linear", "ReLU", "Linear", "Dropout"]
    assert p.net[0].bias is not None and p.net[2].bias is not None
    assert p(torch.randn(2, 3)).shape == (2, 5)


def test_embedding_replacement_after_lookup_and_grad():
    base = torch.zeros(2, 4, 3, requires_grad=True)
    z = torch.randn(2, 1, 3, requires_grad=True)
    mask = torch.tensor([[False, True, False, False], [False, False, True, False]])
    out = replace_embeddings_after_lookup(base, z, mask)
    assert torch.equal(out[0, 1], z[0, 0])
    loss = out.sum()
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_three_interpreter_independence_spec():
    plan = mode_plan("heima_scaled_baseline")
    stages = [x["stage"] for x in plan]
    assert "train_interpreter_summary" in stages
    assert "train_interpreter_caption" in stages
    assert "train_interpreter_reasoning" in stages


def test_frozen_a_baseline_and_warm_modes_plan():
    base = mode_plan("heima_scaled_baseline")
    assert all(not x.get("train_a", True) for x in base if x["stage"].startswith("train_interpreter"))
    fixed = mode_plan("ours_warm_b_fixed")
    assert all(x.get("b_frozen_differentiable") for x in fixed)
    joint = mode_plan("ours_warm_b_joint")
    assert all(x.get("train_a") and x.get("train_b") for x in joint)
    cold = mode_plan("ours_cold_b_joint")
    assert all(x.get("cold_b") for x in cold)
    main = mode_plan("compute_matched_main_only")
    assert all(x.get("lambda_loss1") == 0.0 for x in main)


def test_one_command_dry_run_and_resume(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    cfg = root / "configs/heima_aligned/ab_loss1_qwen_vl3b.yaml"
    run_id = "pytest_protocol"
    cmd = [
        "bash", str(root / "scripts/heima_aligned/run_all.sh"),
        "--config", str(cfg), "--mode", "heima_scaled_baseline", "--run-id", run_id,
        "--output-root", str(tmp_path), "--dry-run", "--smoke",
    ]
    subprocess.run(cmd, check=True, cwd=root)
    subprocess.run(cmd + ["--resume"], check=True, cwd=root)
    out = tmp_path / f"heima_aligned_{run_id}"
    assert (out / "summary.json").exists()
    assert (out / "stages/explicit_cot_sft/COMPLETED").exists()


def test_no_overwrite_existing_output(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    cfg = root / "configs/heima_aligned/ab_loss1_qwen_vl3b.yaml"
    cmd = ["bash", str(root / "scripts/heima_aligned/run_all.sh"), "--config", str(cfg), "--mode", "heima_scaled_baseline", "--run-id", "x", "--output-root", str(tmp_path), "--dry-run", "--smoke"]
    subprocess.run(cmd, check=True, cwd=root)
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(cmd, check=True, cwd=root)


def test_paper_and_causal_evaluator_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    cfg = root / "configs/heima_aligned/ab_loss1_qwen_vl3b.yaml"
    cmd = ["bash", str(root / "scripts/heima_aligned/run_eval.sh"), "--config", str(cfg), "--mode", "heima_scaled_baseline", "--run-id", "eval", "--output-root", str(tmp_path), "--dry-run", "--smoke"]
    subprocess.run(cmd, check=True, cwd=root)
    summary = json.loads((tmp_path / "heima_aligned_eval/summary.json").read_text())
    stages = [x.get("stage") for x in summary["completed"]]
    assert "eval_encoder" in stages and "eval_decoder" in stages and "eval_causal" in stages
