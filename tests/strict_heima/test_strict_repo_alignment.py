from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from transformers import GPT2Config, GPT2LMHeadModel

from src.htext.heima_reuse import (
    HeimaOfficialAbstractProjection,
    backend_resolution_snapshot,
    build_predictor_mask,
    direct_thinking_mask,
    extract_thinking_state,
    heima_ce_loss,
    heima_shifted_thinking_mask,
    official_embedding_replacement,
    prepare_latent_for_decoder,
)
from src.htext.modeling import THINKING_TOKEN, _decoder_prompt_ids, h0_forward, h1_forward, setup_special_tokens
from src.htext.trainer import _build_projector


class TinyTokenizer:
    def __init__(self):
        words = [
            THINKING_TOKEN,
            "Compute",
            "2",
            "plus",
            "3.",
            "3",
            "5",
            "2",
            "equals",
            "5.",
            "Question:",
            "Answer:",
            "Instruction:",
            "Explain",
            "the",
            "reasoning",
            "information",
            "encoded",
            "in",
            "latent",
            "state.",
            "Latent:",
            "Reasoning:",
            "full",
            "for",
            "this",
            "question.",
            "5<|eos|>",
            "5.<|eos|>",
        ]
        self.vocab = {"<|pad|>": 0, "<|eos|>": 1}
        for word in words:
            self.vocab.setdefault(word, len(self.vocab))
        self.pad_token = "<|pad|>"
        self.eos_token = "<|eos|>"
        self.pad_token_id = 0
        self.eos_token_id = 1

    def add_special_tokens(self, spec):
        added = 0
        for token in spec.get("additional_special_tokens", []):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def __len__(self):
        return len(self.vocab)

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        for part in text.replace("\n", " ").split():
            if part not in self.vocab:
                self.vocab[part] = len(self.vocab)
            ids.append(self.vocab[part])
        return {"input_ids": ids}


def _tiny_models():
    tokenizer = TinyTokenizer()
    cfg = GPT2Config(
        vocab_size=128,
        n_positions=64,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=1,
        eos_token_id=1,
    )
    model_a = GPT2LMHeadModel(cfg)
    model_b = GPT2LMHeadModel(cfg)
    setup_special_tokens(tokenizer, model_a, model_b)
    return tokenizer, model_a, model_b


def test_predictor_selected_pos_is_thinking_pos_minus_one_and_matches_repo_mask():
    thinking_id = 7
    input_ids = torch.tensor([[3, 4, thinking_id, 8], [5, 6, 9, thinking_id]])
    hidden = torch.randn(2, 4, 5)
    state = extract_thinking_state(
        input_ids=input_ids,
        last_hidden_state=hidden,
        thinking_token_id=thinking_id,
        mode="predictor",
    )
    torch.testing.assert_close(state.selected_positions, state.thinking_positions - 1, rtol=0, atol=0)
    torch.testing.assert_close(build_predictor_mask(input_ids, thinking_id), heima_shifted_thinking_mask(input_ids, thinking_id), rtol=0, atol=0)
    torch.testing.assert_close(state.hidden, torch.stack([hidden[0, 1], hidden[1, 2]], dim=0), rtol=0, atol=0)


def test_selected_hidden_is_position_that_predicts_thinking_token():
    thinking_id = 11
    input_ids = torch.tensor([[4, 5, thinking_id, 6]])
    labels = input_ids.clone()
    state = extract_thinking_state(
        input_ids=input_ids,
        last_hidden_state=torch.randn(1, 4, 3),
        thinking_token_id=thinking_id,
        mode="predictor",
    )
    predictor_pos = int(state.selected_positions[0].item())
    assert int(labels[:, 1:][0, predictor_pos].item()) == thinking_id


def test_strict_projector_class_is_heima_official():
    projector = _build_projector({"heima_compatibility": "strict_repo", "projector": "heima_official"}, 8, strict_repo=True)
    assert isinstance(projector, HeimaOfficialAbstractProjection)
    assert [type(layer).__name__ for layer in projector.net] == ["Linear", "ReLU", "Linear", "Dropout"]


def test_decoder_prompt_contains_question_and_official_thinking_token():
    tokenizer = TinyTokenizer()
    prompt_ids, pos = _decoder_prompt_ids(tokenizer, "Compute 2 plus 3.", mode="qz")
    assert pos is not None
    assert tokenizer.convert_tokens_to_ids(THINKING_TOKEN) in prompt_ids
    decoded_ids = set(prompt_ids)
    assert tokenizer.vocab["Question:"] in decoded_ids


def test_b_embedding_replacement_occurs_at_thinking_token_position():
    token_embeds = torch.randn(1, 5, 4, requires_grad=True)
    latent = torch.randn(1, 1, 4, requires_grad=True)
    mask = torch.tensor([[False, False, True, False, False]])
    replaced = official_embedding_replacement(token_embeds, latent, mask)
    torch.testing.assert_close(replaced[0, 2], latent[0, 0], rtol=0, atol=0)
    torch.testing.assert_close(replaced[0, 1], token_embeds[0, 1], rtol=0, atol=0)


def test_official_ce_loss_backend_executes_when_repo_source_exists():
    source = Path("torchtune_pkg/torchtune/torchtune/modules/loss/ce_chunked_output_loss.py")
    if not source.exists():
        pytest.skip("official torchtune source tree is not present in this workspace")
    logits = torch.randn(2, 4, 9, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, 4, -100, 5]])
    loss = heima_ce_loss(logits, labels)
    loss.backward()
    resolution = backend_resolution_snapshot()["ce_loss"]
    assert resolution["fallback_used"] is False
    assert resolution["actual_backend"].endswith("CEWithChunkedOutputLoss")
    assert logits.grad is not None


def test_joint_configs_only_differ_by_detach():
    base = Path("experiments/htext_gpt2/configs")
    detach = yaml.safe_load((base / "strict_repo_joint_detach.yaml").read_text())
    no_detach = yaml.safe_load((base / "strict_repo_joint_no_detach.yaml").read_text())
    detach.pop("detach_encoder_latent")
    no_detach.pop("detach_encoder_latent")
    assert detach == no_detach


def test_detach_true_blocks_loss1_to_model_a_and_false_allows_it():
    torch.manual_seed(0)
    records = [{"question": "Compute 2 plus 3.", "answer": "5", "cot": "2 plus 3 equals 5."}]
    for detach in (True, False):
        tokenizer, model_a, model_b = _tiny_models()
        projector = HeimaOfficialAbstractProjection(16, 16)
        main = h0_forward(model_a, tokenizer, records, 16, 8, 1, "predictor")
        z = prepare_latent_for_decoder(main.thinking_hidden, detach)
        loss1 = h1_forward(model_b, tokenizer, records, z, projector, 16, mode="qz")
        loss1.loss.backward()
        a_grad = sum(float(p.grad.detach().abs().sum().item()) for p in model_a.parameters() if p.grad is not None)
        if detach:
            assert a_grad == 0.0
        else:
            assert a_grad > 0.0
