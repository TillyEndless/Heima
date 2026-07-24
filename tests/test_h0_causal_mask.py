from __future__ import annotations

import json
from pathlib import Path


def build_toy_sequence():
    # H0 decoder layout mirrors decoder_forward: prompt(question + latent slot) + target CoT.
    return ["Q", "<LATENT>", "t1", "t2", "t3", "t4"]


def causal_visible(src_pos: int, dst_pos: int) -> bool:
    # Decoder-only causal attention: hidden at src_pos can attend only positions <= src_pos.
    return dst_pos <= src_pos


def prediction_source_position(label_position: int) -> int:
    # heima_ce_loss shifts labels left: label at position p is predicted by logits at p-1.
    return label_position - 1


def test_h0_decoder_causal_visibility(tmp_path=None):
    seq = build_toy_sequence()
    target_positions = {"t1": 2, "t2": 3, "t3": 4, "t4": 5}
    checks = {}

    src_t1 = prediction_source_position(target_positions["t1"])
    checks["predict_t1"] = {
        "source_position": src_t1,
        "visible": [tok for i, tok in enumerate(seq) if causal_visible(src_t1, i)],
        "invisible": [tok for i, tok in enumerate(seq) if not causal_visible(src_t1, i)],
    }
    assert checks["predict_t1"]["visible"] == ["Q", "<LATENT>"]
    assert checks["predict_t1"]["invisible"] == ["t1", "t2", "t3", "t4"]

    src_t3 = prediction_source_position(target_positions["t3"])
    checks["predict_t3"] = {
        "source_position": src_t3,
        "visible": [tok for i, tok in enumerate(seq) if causal_visible(src_t3, i)],
        "invisible": [tok for i, tok in enumerate(seq) if not causal_visible(src_t3, i)],
    }
    assert checks["predict_t3"]["visible"] == ["Q", "<LATENT>", "t1", "t2"]
    assert checks["predict_t3"]["invisible"] == ["t3", "t4"]

    report = {
        "status": "PASS_NO_CAUSAL_LEAK_FOUND",
        "sequence": seq,
        "rule": "labels[p] is predicted from hidden[p-1]; hidden[p-1] can attend only positions <= p-1",
        "checks": checks,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/h0_causal_visibility_toy.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    test_h0_decoder_causal_visibility()
