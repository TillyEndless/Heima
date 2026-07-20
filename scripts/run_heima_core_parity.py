#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.htext.heima_reuse import (  # noqa: E402
    OfficialCompatibleAbstractProjection,
    backend_resolution_snapshot,
    direct_thinking_mask,
    heima_ce_loss,
    heima_shifted_thinking_mask,
    hf_shifted_ce_loss,
    official_embedding_replacement,
    official_projector_spec,
    prepare_latent_for_decoder,
    write_backend_resolution,
)
from src.htext.modeling import (  # noqa: E402
    LATENT_TOKEN,
    build_h0_labels,
    build_h1_labels,
    setup_special_tokens,
    tokenize_text,
)
from src.htext.schema_adapter import OFFICIAL_REASONING_TOKEN, decoder_prompt, minimal_fixture  # noqa: E402


OUT = Path("reports/heima_core_parity")


def write_json(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_pytest() -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/heima_parity", "-q"],
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


def make_thinking_trace() -> dict:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = "/mnt/nas/share2/home/zxl/models/openai-community-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_safetensors=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, use_safetensors=True)
    setup_special_tokens(tokenizer, model)
    model.eval()
    record = {"question": "Compute (2 + 3) * 4.", "answer": "20", "cot": "First add 2 and 3: 2 + 3 = 5. Then multiply 5 by 4: 5 * 4 = 20."}
    thinking_id = tokenizer.convert_tokens_to_ids(OFFICIAL_REASONING_TOKEN)
    q_ids = tokenize_text(tokenizer, record["question"], 80)
    answer_ids = tokenize_text(tokenizer, record["answer"] + tokenizer.eos_token, 16)
    prefix_ids = tokenize_text(tokenizer, "\nAnswer: ")
    input_ids = torch.tensor([q_ids + [thinking_id] + prefix_ids + answer_ids])
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    direct = direct_thinking_mask(input_ids, thinking_id)
    shifted = heima_shifted_thinking_mask(input_ids, thinking_id)
    direct_idx = torch.where(direct)[1].tolist()
    shifted_idx = torch.where(shifted)[1].tolist()
    return {
        "official_shift_source": "heima/main_python/2-training...py:1503-1532 masks tokens[:, 1:] == thinking_id, selecting the hidden state that predicts the thinking token under next-token CE.",
        "hf_gpt2_semantics": f"HF GPT-2 hidden_states[-1][position] is the contextual representation after consuming token at position. The direct {OFFICIAL_REASONING_TOKEN} hidden is position t; the official shifted mask selects t-1, the predictor position for token t.",
        "equivalence_judgment": f"Not equivalent if the intended latent is the contextual hidden of {OFFICIAL_REASONING_TOKEN} itself. Equivalent only if the intended Heima latent is the predictor hidden immediately before {OFFICIAL_REASONING_TOKEN}, matching the official next-token-shift implementation.",
        "recommendation": "Use thinking_state_mode: predictor for strict repository alignment; keep thinking_state_mode: token only as an ablation.",
        "input_token_ids": input_ids[0].tolist(),
        "input_token_strings": [tokenizer.decode([int(x)]) for x in input_ids[0]],
        "direct_thinking_mask": direct[0].tolist(),
        "official_shifted_mask": shifted[0].tolist(),
        "direct_selected_indices": direct_idx,
        "shifted_selected_indices": shifted_idx,
        "direct_selected_token_strings": [tokenizer.decode([int(input_ids[0, i])]) for i in direct_idx],
        "shifted_selected_token_strings": [tokenizer.decode([int(input_ids[0, i])]) for i in shifted_idx],
        "direct_hidden_shape": list(out.hidden_states[-1][direct].shape),
        "shifted_hidden_shape": list(out.hidden_states[-1][shifted].shape),
        "off_by_one_exists": bool(direct_idx and shifted_idx and direct_idx[0] != shifted_idx[0]),
    }


def projector_report() -> dict:
    htext = {
        "class": "src.htext.modeling.LatentProjector",
        "layer_order": ["LayerNorm", "Linear"],
        "bias": True,
        "normalization": "LayerNorm",
        "initialization": "Linear identity weight, zero bias",
    }
    official = official_projector_spec(4096, 4096)
    return {
        "official": official,
        "htext_current": htext,
        "mismatch": True,
        "action_taken": "Added OfficialCompatibleAbstractProjection; did not delete LatentProjector to preserve old checkpoints.",
    }


def embedding_report() -> dict:
    token_embeds = torch.randn(2, 4, 3, requires_grad=True)
    latent = torch.randn(2, 1, 3, requires_grad=True)
    mask = torch.tensor([[False, True, False, False], [False, False, True, False]])
    out = official_embedding_replacement(token_embeds, latent, mask)
    out.sum().backward()
    return {
        "official_source": "torchtune_pkg/torchtune/torchtune/modules/transformer.py:669-701",
        "replacement_position": "after tok_embeddings(tokens), before transformer layers",
        "position_embedding_note": "Llama/Torchtune path uses RoPE, no additive learned token-position embedding at this replacement point. HF GPT-2 adapter replacement through inputs_embeds occurs before GPT-2 adds learned position embeddings.",
        "mask_shape": list(mask.shape),
        "multiple_thinking_tokens": "official tensor path flattens all true mask positions and requires count == batch*num_replace_tokens; list path handles variable counts per batch",
        "in_place_or_out_of_place": "official mutates the local h tensor view in-place; adapter returns cloned out-of-place tensor for safer parity testing",
        "grad_to_latent_nonzero": latent.grad.detach().abs().sum().item() > 0,
        "grad_to_token_embedding_nonzero": token_embeds.grad.detach().abs().sum().item() > 0,
    }


def label_loss_trace() -> dict:
    fixture = minimal_fixture()
    token_map = {OFFICIAL_REASONING_TOKEN: 101, LATENT_TOKEN: 102}
    question_ids = torch.tensor([11, 12, 13])
    answer_ids = torch.tensor([21, 22])
    prefix_len = 1
    main_ids = torch.tensor([11, 12, 13, token_map[OFFICIAL_REASONING_TOKEN], 31, 21, 22])
    main_labels = build_h0_labels(
        total_len=main_ids.numel(),
        question_len=3,
        num_thinking_tokens=1,
        answer_prefix_len=prefix_len,
        answer_ids=answer_ids,
        thinking_id=token_map[OFFICIAL_REASONING_TOKEN],
    )
    prompt_ids = torch.tensor([41, 42, token_map[LATENT_TOKEN], 43])
    target_ids = torch.tensor([51, 52, 53])
    loss1_ids = torch.cat([prompt_ids, target_ids])
    loss1_labels = build_h1_labels(loss1_ids.numel(), prompt_ids.numel(), target_ids)
    logits = torch.randn(2, 5, 7, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, -100, 3], [-100, 4, 5, 6, -100]])
    official_loss = heima_ce_loss(logits, labels)
    hf_loss = hf_shifted_ce_loss(logits, labels)
    return {
        "fixture": fixture.__dict__,
        "main": {
            "input_ids": main_ids.tolist(),
            "labels": main_labels.tolist(),
            "ignore_positions": torch.where(main_labels == -100)[0].tolist(),
            "shifted_labels": main_labels[1:].tolist(),
            "participating_token_roles": ["thinking_token", "answer"],
            "non_ignored_token_count": int((main_labels != -100).sum().item()),
            "loss_denominator": int((main_labels[1:] != -100).sum().item()),
        },
        "loss1": {
            "input_ids": loss1_ids.tolist(),
            "labels": loss1_labels.tolist(),
            "ignore_positions": torch.where(loss1_labels == -100)[0].tolist(),
            "shifted_labels": loss1_labels[1:].tolist(),
            "participating_token_roles": ["cot_target"],
            "decoder_question_prompt_participates": False,
            "non_ignored_token_count": int((loss1_labels != -100).sum().item()),
            "loss_denominator": int((loss1_labels[1:] != -100).sum().item()),
        },
        "loss": {
            "official_torchtune_loss": float(official_loss.item()),
            "hf_shifted_ce_loss": float(hf_loss.item()),
            "absolute_error": abs(float(official_loss.item()) - float(hf_loss.item())),
            "ignore_index": -100,
            "reduction": "sum over nonignored tokens / nonignored token count",
        },
        "train_on_input_note": "Official decoder dataset sets train_on_input=True, but HText Loss1 masks prompt/question by construction. This is a remaining semantic mismatch unless explicitly configured.",
    }


def remaining_components() -> dict:
    return {
        "hf_gpt2_model_adapter": "HF GPT-2 cannot directly call Torchtune Llama/Vision builders or transformer thinking_token forward API.",
        "data_builder": "Official LLaVA-CoT section/image builders are not directly reused; HText uses text-only fixture/schema adapter.",
        "training_recipe": "Torchtune FSDP/LoRA/checkpointer recipe is not directly reused.",
        "progressive_pipeline": "Not reused by task constraint.",
        "official_parameters": "Llama-3.2V-11B-cot and Llama-3.1-8B-Instruct checkpoints are not used in lightweight GPT-2 run.",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parity = run_pytest()
    write_json("parity_test_results.json", parity)
    trace = make_thinking_trace()
    write_json("thinking_position_trace.json", trace)
    write_json("projector_spec.json", projector_report())
    write_json("embedding_replacement_spec.json", embedding_report())
    write_json("label_loss_trace.json", label_loss_trace())
    write_backend_resolution(OUT / "backend_resolution.json")
    write_json("remaining_non_reused_components.json", remaining_components())
    backend = backend_resolution_snapshot()
    report = f"""# HEIMA-CORE-PARITY-MIGRATION

Status: migration/parity only. No formal training, Loss2, K expansion, or model change was run.

## Direct Official Imports

- CEWithChunkedOutputLoss: actual backend `{backend.get('ce_loss', {}).get('actual_backend')}`, fallback_used={backend.get('ce_loss', {}).get('fallback_used')}.

## Copied / Mirrored Official Logic

- Shifted thinking-token mask mirrors Heima training lines 1503-1532.
- Embedding replacement mirrors Torchtune transformer lines 669-701.
- Official-compatible projector class mirrors transformer lines 404-416.

## HF Compatibility Adapters

- GPT-2 uses `inputs_embeds` replacement because HF GPT-2 has no `thinking_token` forward API.
- GPT-2 model adapters are represented by `HeimaEncoderInterface` and `HeimaDecoderInterface` wrappers.

## Key Findings

- Shifted hidden is not the same as the contextual hidden at `{OFFICIAL_REASONING_TOKEN}` under HF GPT-2. It selects the previous position that predicts `{OFFICIAL_REASONING_TOKEN}`. Strict repo mode uses `thinking_state_mode: predictor`; token mode is an ablation.
- Current `LayerNorm -> Linear` projector does not match official `Linear -> ReLU -> Linear -> Dropout`.
- Loss value and gradient parity against HF shifted CE is covered by parity tests.
- detach=true/false is centralized in `prepare_latent_for_decoder(z, detach_encoder_latent)`.

## Next Training Permission

Do not enter next-stage training yet. Strict repository alignment should use official shifted predictor hidden for the text-only GPT-2 backend.
"""
    (OUT / "heima_core_parity_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "complete", "out": str(OUT), "parity_passed": parity["passed"]}, indent=2))
    return 0 if parity["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
