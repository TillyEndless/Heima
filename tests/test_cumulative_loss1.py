from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import torch
import torch.nn as nn

from scripts.run_data_small_vlm_official_sections import (
    SECTIONS,
    THINKING_TOKENS,
    decoder_forward,
    decoder_prompt,
    prefix_sections,
    prepare_stage_latents,
)


class TinyTokenizer:
    def __init__(self):
        self.eos_token = "<EOS>"
        self.pad_token = "<PAD>"
        self.vocab = {self.pad_token: 0, self.eos_token: 1}
        for token in THINKING_TOKENS.values():
            self.vocab[token] = len(self.vocab)
        self.pad_token_id = self.vocab[self.pad_token]

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        for token in THINKING_TOKENS.values():
            text = text.replace(token, f" {token} ")
        text = text.replace(self.eos_token, f" {self.eos_token} ")
        ids = []
        for piece in text.replace("\n", " ").split():
            if piece not in self.vocab:
                self.vocab[piece] = len(self.vocab)
            ids.append(self.vocab[piece])
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv[int(i)] for i in ids)

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]


class TinyLM(nn.Module):
    def __init__(self, vocab_size=512, hidden=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.mix = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, vocab_size)
        self.config = SimpleNamespace(hidden_size=hidden, use_cache=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, use_cache=False):
        assert use_cache is False
        if inputs_embeds is None:
            x = self.embed(input_ids)
        else:
            assert input_ids is None
            x = inputs_embeds
        if attention_mask is None:
            attention_mask = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
        hidden = torch.tanh(self.mix(torch.cumsum(x * attention_mask.unsqueeze(-1).to(x.dtype), dim=1)))
        return SimpleNamespace(logits=self.head(hidden))


def args(**kwargs):
    base = dict(max_q=64, max_target=64, loss1_latent_context_mode="local", cumulative_grad_mode="all_prefix", train_latent_marker_ntp=False)
    base.update(kwargs)
    return Namespace(**base)


def records():
    return [
        {"question": "What is shown?", "summary": "Look at the image.", "caption": "A green plant.", "reasoning": "The leaves imply A.", "answer": "A"},
        {"question": "Which colony?", "summary": "Find the highlighted area.", "caption": "A map is shown.", "reasoning": "The area is New York.", "answer": "New York"},
    ]


def make():
    torch.manual_seed(0)
    tok = TinyTokenizer()
    model = TinyLM()
    projectors = {s: nn.Linear(8, 8) for s in SECTIONS}
    z = {s: torch.randn(2, 8, requires_grad=True) for s in SECTIONS}
    return tok, model, projectors, z


def test_prefix_sections_no_future_latents():
    assert prefix_sections("summary", "local") == ("summary",)
    assert prefix_sections("caption", "local") == ("caption",)
    assert prefix_sections("reasoning", "local") == ("reasoning",)
    assert prefix_sections("summary", "causal_cumulative") == ("summary",)
    assert prefix_sections("caption", "causal_cumulative") == ("summary", "caption")
    assert prefix_sections("reasoning", "causal_cumulative") == ("summary", "caption", "reasoning")


def test_prompt_slots_are_causal_and_q_only_has_no_slots():
    tok = TinyTokenizer()
    rec = records()[0]
    p_summary = decoder_prompt(rec, "summary", tok, args(), context_mode="causal_cumulative")
    p_caption = decoder_prompt(rec, "caption", tok, args(), context_mode="causal_cumulative")
    p_reasoning = decoder_prompt(rec, "reasoning", tok, args(), context_mode="causal_cumulative")
    assert p_summary.count(THINKING_TOKENS["summary"]) == 1
    assert THINKING_TOKENS["caption"] not in p_summary
    assert p_caption.index(THINKING_TOKENS["summary"]) < p_caption.index(THINKING_TOKENS["caption"])
    assert THINKING_TOKENS["reasoning"] not in p_caption
    assert p_reasoning.index(THINKING_TOKENS["summary"]) < p_reasoning.index(THINKING_TOKENS["caption"]) < p_reasoning.index(THINKING_TOKENS["reasoning"])
    assert "<THINKING_OF_" not in decoder_prompt(rec, "reasoning", tok, args(), q_only=True, context_mode="causal_cumulative")


def test_local_parity_with_single_projector_and_projector_dict():
    tok, model, projectors, z = make()
    recs = records()
    a = args(loss1_latent_context_mode="local")
    old_loss, old_logits, old_labels = decoder_forward(model, projectors["caption"], tok, "caption", recs, z["caption"], a, context_mode="local")
    new_loss, new_logits, new_labels = decoder_forward(model, projectors, tok, "caption", recs, {"caption": z["caption"]}, a)
    assert torch.allclose(old_loss, new_loss)
    assert torch.allclose(old_logits, new_logits)
    assert torch.equal(old_labels, new_labels)


def test_cumulative_labels_only_target_and_padding_batch():
    tok, model, projectors, z = make()
    a = args(loss1_latent_context_mode="causal_cumulative")
    loss, _logits, labels = decoder_forward(model, projectors, tok, "reasoning", records(), z, a)
    assert torch.isfinite(loss)
    target_count = sum(len(tok(r["reasoning"] + tok.eos_token)["input_ids"][: a.max_target]) for r in records())
    assert int((labels != -100).sum().item()) == target_count
    for token in THINKING_TOKENS.values():
        token_id = tok.convert_tokens_to_ids(token)
        assert not torch.any(labels == token_id)


def test_grad_modes_local_all_prefix_current_only_and_detach():
    tok, model, projectors, z = make()
    recs = records()
    local = args(loss1_latent_context_mode="local")
    loss, *_ = decoder_forward(model, projectors, tok, "caption", recs, prepare_stage_latents(z, "caption", local, detach_encoder_latent=False), local)
    grads = torch.autograd.grad(loss, [z[s] for s in SECTIONS], retain_graph=True, allow_unused=True)
    assert grads[0] is None
    assert grads[1] is not None and grads[1].norm() > 0
    assert grads[2] is None

    cum = args(loss1_latent_context_mode="causal_cumulative", cumulative_grad_mode="all_prefix")
    loss, *_ = decoder_forward(model, projectors, tok, "reasoning", recs, prepare_stage_latents(z, "reasoning", cum, detach_encoder_latent=False), cum)
    grads = torch.autograd.grad(loss, [z[s] for s in SECTIONS], retain_graph=True, allow_unused=True)
    assert all(g is not None and g.norm() > 0 for g in grads)

    current = args(loss1_latent_context_mode="causal_cumulative", cumulative_grad_mode="current_only")
    loss, *_ = decoder_forward(model, projectors, tok, "reasoning", recs, prepare_stage_latents(z, "reasoning", current, detach_encoder_latent=False), current)
    grads = torch.autograd.grad(loss, [z[s] for s in SECTIONS], retain_graph=True, allow_unused=True)
    assert grads[0] is None and grads[1] is None
    assert grads[2] is not None and grads[2].norm() > 0

    loss, *_ = decoder_forward(model, projectors, tok, "reasoning", recs, prepare_stage_latents(z, "reasoning", cum, detach_encoder_latent=True), cum)
    grads = torch.autograd.grad(loss, [z[s] for s in SECTIONS], retain_graph=True, allow_unused=True)
    assert all(g is None for g in grads)


def test_shuffle_variants_preserve_shape_and_norm():
    _tok, _model, _projectors, z = make()
    rolled = {s: torch.roll(z[s], shifts=1, dims=0) for s in SECTIONS}
    for s in SECTIONS:
        assert rolled[s].shape == z[s].shape
        assert torch.allclose(rolled[s].float().norm(dim=-1).sort().values, z[s].float().norm(dim=-1).sort().values)
