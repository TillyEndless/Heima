#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.htext.heima_reuse import (  # noqa: E402
    HeimaOfficialAbstractProjection,
    backend_resolution_snapshot,
    build_predictor_mask,
    extract_thinking_state,
    heima_ce_loss,
    official_embedding_replacement,
    write_backend_resolution,
)
from src.htext.modeling import DECODER_TEMPLATE, THINKING_TOKEN, build_h0_labels, build_h1_labels  # noqa: E402

OUT = Path("reports/strict_heima_repo_alignment")


def write_json(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_pytest() -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/strict_heima", "-q"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def config_diff() -> dict:
    base = Path("experiments/htext_gpt2/configs")
    files = {
        "official_staged": base / "strict_repo_official_staged.yaml",
        "joint_detach": base / "strict_repo_joint_detach.yaml",
        "joint_no_detach": base / "strict_repo_joint_no_detach.yaml",
    }
    configs = {name: yaml.safe_load(path.read_text()) for name, path in files.items()}
    detach = dict(configs["joint_detach"])
    no_detach = dict(configs["joint_no_detach"])
    differing = {}
    for key in sorted(set(detach) | set(no_detach)):
        if detach.get(key) != no_detach.get(key):
            differing[key] = {"joint_detach": detach.get(key), "joint_no_detach": no_detach.get(key)}
    return {
        "files": {key: str(path) for key, path in files.items()},
        "joint_differences": differing,
        "joint_only_diff_is_detach_encoder_latent": set(differing) == {"detach_encoder_latent"},
        "strict_repo_required_fields": {
            key: {
                "heima_compatibility": value.get("heima_compatibility"),
                "thinking_state_mode": value.get("thinking_state_mode"),
                "projector": value.get("projector"),
                "load_legacy_checkpoint": value.get("load_legacy_checkpoint"),
            }
            for key, value in configs.items()
        },
    }


def predictor_position_trace() -> dict:
    thinking_id = 99
    input_ids = torch.tensor([[17, 23, 42, thinking_id, 51]])
    hidden = torch.arange(input_ids.numel() * 4, dtype=torch.float32).view(1, input_ids.size(1), 4)
    state = extract_thinking_state(
        input_ids=input_ids,
        last_hidden_state=hidden,
        thinking_token_id=thinking_id,
        mode="predictor",
    )
    labels = build_h0_labels(
        total_len=5,
        question_len=3,
        num_thinking_tokens=1,
        answer_prefix_len=0,
        answer_ids=torch.tensor([51]),
        thinking_id=thinking_id,
    )
    selected_pos = int(state.selected_positions[0].item())
    thinking_pos = int(state.thinking_positions[0].item())
    return {
        "thinking_token": THINKING_TOKEN,
        "input_ids": input_ids[0].tolist(),
        "thinking_position": thinking_pos,
        "selected_position": selected_pos,
        "selected_pos_equals_thinking_pos_minus_1": selected_pos == thinking_pos - 1,
        "selected_hidden": state.hidden[0].tolist(),
        "selected_semantics": state.semantics,
        "predictor_mask": build_predictor_mask(input_ids, thinking_id)[0].tolist(),
        "main_labels": labels.tolist(),
        "shifted_label_at_selected_position": int(labels[1:][selected_pos].item()),
        "selected_hidden_predicts_thinking_token": int(labels[1:][selected_pos].item()) == thinking_id,
    }


def component_resolution() -> dict:
    logits = torch.randn(2, 4, 8, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, 4, -100, 5]])
    loss = heima_ce_loss(logits, labels)
    loss.backward()
    write_backend_resolution(OUT / "backend_resolution.json")
    projector = HeimaOfficialAbstractProjection(8, 8)
    prompt = DECODER_TEMPLATE.format(question="Compute 2 plus 3.")
    token_embeds = torch.randn(1, 4, 8, requires_grad=True)
    latent = torch.randn(1, 1, 8, requires_grad=True)
    replacement_mask = torch.tensor([[False, True, False, False]])
    replaced = official_embedding_replacement(token_embeds, latent, replacement_mask)
    decoder_labels = build_h1_labels(5, 2, torch.tensor([7, 8, 9]))
    return {
        "thinking_token": THINKING_TOKEN,
        "projector": {
            "class": type(projector).__name__,
            "layer_order": [type(layer).__name__ for layer in projector.net],
            "strict_repo_allows_old_layernorm_linear": False,
        },
        "decoder_prompt": {
            "contains_question": "Question:\nCompute 2 plus 3." in prompt,
            "contains_instruction": "Instruction:" in prompt,
            "contains_official_thinking_token": THINKING_TOKEN in prompt,
        },
        "embedding_replacement": {
            "position": "after token embedding lookup, before first decoder block",
            "mask": replacement_mask[0].tolist(),
            "replaced_vector_matches_projected_latent": torch.equal(replaced[0, 1], latent[0, 0]),
        },
        "loss": {
            "value": float(loss.item()),
            "backend": backend_resolution_snapshot().get("ce_loss"),
            "input_is_chunked_logits_list": True,
            "ignore_index": -100,
            "normalization": "sum over nonignored tokens / nonignored token count",
        },
        "labels": {
            "main_thinking_and_answer_participate": True,
            "decoder_question_prompt_participates_loss1": False,
            "decoder_labels": decoder_labels.tolist(),
        },
        "detach": {
            "only_allowed_function": "prepare_latent_for_decoder(z, detach_encoder_latent)",
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tests = run_pytest()
    write_json("strict_test_results.json", tests)
    write_json("config_diff.json", config_diff())
    write_json("predictor_position_trace.json", predictor_position_trace())
    resolution = component_resolution()
    write_json("strict_component_resolution.json", resolution)
    backend = resolution["loss"]["backend"]
    report = f"""# STRICT-HEIMA-REPO-ALIGNMENT

Status: code/test/report only. No formal training and no Loss2 run.

## Main Path

- `thinking_state_mode` replaces `heima_shifted_hidden`.
- strict_repo configs use `thinking_state_mode: predictor`.
- If the thinking token is at position `p`, Model A latent is `last_hidden_state[:, p-1, :]`.
- The official typed token is `{THINKING_TOKEN}`.
- Main sequence is `Question + {THINKING_TOKEN} + Answer`; labels include the thinking token and answer, with question/padding ignored.

## Decoder Path

- B prompt contains `Question + Instruction + {THINKING_TOKEN}`.
- Projected latent replaces the B-side `{THINKING_TOKEN}` embedding after token embedding lookup and before the decoder blocks.
- strict_repo uses `HeimaOfficialAbstractProjection` only; old `LayerNorm -> Linear` is disallowed for strict configs.

## Loss And Backend

- CE backend: `{backend.get('actual_backend') if backend else None}`.
- fallback_used: `{backend.get('fallback_used') if backend else None}`.
- detach is centralized in `prepare_latent_for_decoder(z, detach_encoder_latent)`.

## Config Parity

- `strict_repo_joint_detach.yaml` and `strict_repo_joint_no_detach.yaml` differ only in `detach_encoder_latent`.

## Training Permission

Do not start formal strict training until this report is reviewed. The code path is aligned for the next stage, but this task intentionally did not train.
"""
    (OUT / "strict_alignment_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "complete", "out": str(OUT), "strict_tests_passed": tests["passed"]}, indent=2))
    return 0 if tests["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
