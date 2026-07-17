import torch

from src.gradient_bridge.gradient_audit import run_directional_derivative_check
from src.gradient_bridge.tiny_models import make_tiny_gpt2_pair


def test_directional_derivative_matches_autograd():
    model_a, model_b = make_tiny_gpt2_pair(seed=21, dtype=torch.float64)

    result = run_directional_derivative_check(
        model_a,
        model_b,
        epsilons=(1e-4, 3e-5, 1e-5),
    )

    assert result["relative_error"] < 1e-3
    assert result["epsilon"] in (1e-4, 3e-5, 1e-5)
    assert torch.isfinite(torch.tensor(result["analytic"], dtype=torch.float64))
    assert torch.isfinite(torch.tensor(result["numeric"], dtype=torch.float64))
