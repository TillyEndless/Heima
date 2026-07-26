import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gpt2_matrix", ROOT / "scripts" / "run_gpt2_cot_latent_adjacent_matrix.py")
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_masked_loss_empty_mask_is_finite_zero():
    logits = torch.randn(2, 4, 8)
    labels = torch.full((2, 4), -100)
    mask = torch.zeros((2, 4), dtype=torch.bool)
    loss = mod.masked_loss(logits, labels, mask)
    assert torch.isfinite(loss)
    assert float(loss) == 0.0


def test_declared_groups_do_not_use_forbidden_components():
    for path in (ROOT / "configs" / "gpt2_cot_latent_adjacent_matrix").glob("*.yaml"):
        text = path.read_text()
        assert "has_model_b: false" in text
        assert "projector: false" in text
        assert "role_embedding: false" in text
        assert "cumulative_latent: false" in text
        assert "loss2: false" in text


def test_example_schema_is_question_cot_answer():
    ex = mod.Example(sample_id="x", question="q", cot="reason", answer="a")
    assert ex.question == "q"
    assert ex.cot == "reason"
    assert ex.answer == "a"


if __name__ == "__main__":
    test_masked_loss_empty_mask_is_finite_zero()
    test_declared_groups_do_not_use_forbidden_components()
    test_example_schema_is_question_cot_answer()
