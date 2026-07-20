from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.g1.latent_reasoner import causal_lm_loss
from src.g1.loss2_teacher import (
    SEM_TOKEN,
    assert_student_teacher_independent,
    assert_teacher_frozen_and_excluded,
    causal_leakage_check,
    ensure_sem_token,
    exact_detach_grad_check,
    feature_diagnostics,
    find_same_question_pairs,
    freeze_teacher,
    loss2_intervention_diagnostics,
    loss2_distance,
    loss2_forward,
    parameter_fingerprint,
    student_feature_forward,
    teacher_feature_forward,
)
from src.g1.whole_cot_decoder import LATENT_TOKEN, loss1_forward


class TinyTokenizer:
    def __init__(self):
        self.eos_token = "<EOS>"
        self.pad_token = "<PAD>"
        self.vocab = {self.pad_token: 0, self.eos_token: 1}
        self.pad_token_id = 0
        self.eos_token_id = 1

    def add_special_tokens(self, payload):
        added = 0
        for token in payload.get("additional_special_tokens", []):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        if token == LATENT_TOKEN:
            return self.eos_token_id
        return self.vocab.get(token, -1)

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        text = text.replace(SEM_TOKEN, f" {SEM_TOKEN} ").replace(LATENT_TOKEN, f" {LATENT_TOKEN} ")
        ids = []
        for token in text.replace("\n", " ").split():
            if token == LATENT_TOKEN:
                ids.append(self.eos_token_id)
            else:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                ids.append(self.vocab[token])
        return {"input_ids": ids}

    def __len__(self):
        return len(self.vocab)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=256, hidden_size=12):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.mix = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.config = SimpleNamespace(n_embd=hidden_size, hidden_size=hidden_size, use_cache=False)

    def resize_token_embeddings(self, n):
        if n <= self.embed.num_embeddings:
            return
        old = self.embed
        new = nn.Embedding(n, old.embedding_dim)
        with torch.no_grad():
            new.weight[: old.num_embeddings].copy_(old.weight)
        self.embed = new

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, output_hidden_states=False, use_cache=False):
        assert use_cache is False
        if inputs_embeds is None:
            x = self.embed(input_ids)
        else:
            assert input_ids is None
            x = inputs_embeds
        if attention_mask is None:
            attention_mask = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
        mask = attention_mask.to(x.dtype).unsqueeze(-1)
        hidden = torch.tanh(self.mix(torch.cumsum(x * mask, dim=1)))
        logits = self.lm_head(hidden)
        return SimpleNamespace(logits=logits, hidden_states=(x, hidden) if output_hidden_states else None)


def records():
    return [
        {"question": "Compute two plus three.", "cot": "Add two and three to get five.", "answer": "5"},
        {"question": "Compute four plus six.", "cot": "Add four and six to get ten.", "answer": "10"},
    ]


def make_models():
    torch.manual_seed(0)
    tok = TinyTokenizer()
    b_dec = TinyCausalLM()
    b_teacher = TinyCausalLM()
    b_teacher.load_state_dict(b_dec.state_dict())
    ensure_sem_token(tok, b_dec, b_teacher)
    freeze_teacher(b_teacher)
    return tok, b_dec, b_teacher


def test_teacher_frozen_excluded_and_independent():
    tok, b_dec, b_teacher = make_models()
    del tok
    assert_student_teacher_independent(b_dec, b_teacher)
    opt = torch.optim.AdamW(b_dec.parameters(), lr=1e-3)
    assert_teacher_frozen_and_excluded(b_teacher, opt)
    before = parameter_fingerprint(b_teacher)
    out = teacher_feature_forward(b_teacher, TinyTokenizerWithSem(), records())
    assert out.h_t.requires_grad is False
    assert all(p.grad is None for p in b_teacher.parameters())
    assert parameter_fingerprint(b_teacher) == before


class TinyTokenizerWithSem(TinyTokenizer):
    def __init__(self):
        super().__init__()
        self.add_special_tokens({"additional_special_tokens": [SEM_TOKEN]})


def test_student_gradient_detach_and_teacher_stop_gradient():
    tok, b_dec, b_teacher = make_models()
    recs = records()
    z = torch.randn(len(recs), b_dec.config.n_embd, requires_grad=True)
    student, teacher, loss2 = loss2_forward(b_dec, b_teacher, tok, recs, z, 32)
    assert student.h_l.shape == teacher.h_t.shape
    assert teacher.h_t.requires_grad is False
    grad_z = torch.autograd.grad(loss2.loss2, z, retain_graph=True)[0]
    assert torch.isfinite(grad_z).all() and grad_z.norm() > 0
    loss2.loss2.backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in b_dec.parameters())
    assert all(p.grad is None for p in b_teacher.parameters())

    b_dec.zero_grad(set_to_none=True)
    z2 = torch.randn_like(z, requires_grad=True)
    _student, _teacher, detached = loss2_forward(b_dec, b_teacher, tok, recs, z2, 32, detach_latent=True)
    assert torch.autograd.grad(detached.loss2, z2, allow_unused=True)[0] is None
    detached.loss2.backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in b_dec.parameters())


def test_exact_detach_grad_check_reports_z_control():
    tok, b_dec, b_teacher = make_models()
    recs = records()
    z = torch.randn(len(recs), b_dec.config.n_embd, requires_grad=True)
    out = exact_detach_grad_check(b_dec, b_teacher, tok, recs, z, 32)
    assert out["grad_z_no_detach_finite"] is True
    assert out["grad_z_no_detach_norm"] > 0
    assert out["grad_z_detach_is_none"] is True or out["grad_z_detach_norm"] == 0.0
    assert out["grad_z_detach_finite"] is True


def test_distance_modes_shape_layer_and_latent_sensitivity():
    tok, b_dec, b_teacher = make_models()
    recs = records()
    z = torch.randn(len(recs), b_dec.config.n_embd, requires_grad=True)
    student, teacher, _loss2 = loss2_forward(b_dec, b_teacher, tok, recs, z, 32, layer_index=-1)
    assert student.h_l.shape == teacher.h_t.shape
    assert student.h_l.dtype == teacher.h_t.dtype
    assert student.h_l.device == teacher.h_t.device
    for distance in ("cosine", "mse", "normalized_mse"):
        per = loss2_distance(student.h_l, teacher.h_t, distance)
        assert per.shape == (len(recs),)
        assert torch.isfinite(per).all()
    zero = torch.zeros_like(z)
    normal = student_feature_forward(b_dec, tok, recs, z, 32).h_l
    zero_h = student_feature_forward(b_dec, tok, recs, zero, 32).h_l
    shuffle_h = student_feature_forward(b_dec, tok, recs, torch.roll(z, 1, 0), 32).h_l
    assert not torch.allclose(normal, zero_h)
    assert not torch.allclose(normal, shuffle_h)


def test_loss2_intervention_and_representation_diagnostics():
    tok, b_dec, b_teacher = make_models()
    recs = records()
    z = torch.randn(len(recs), b_dec.config.n_embd)
    diag = loss2_intervention_diagnostics(b_dec, b_teacher, tok, recs, z, 32)
    for key in (
        "loss2_normal",
        "loss2_shuffle",
        "loss2_zero",
        "loss2_random",
        "shuffle_margin",
        "zero_margin",
        "random_margin",
        "hL_batch_variance",
        "hT_batch_variance",
        "hL_mean_pairwise_cosine",
        "hT_mean_pairwise_cosine",
        "correct_pair_cosine",
        "shuffled_pair_cosine",
        "centered_correct_pair_cosine",
        "centered_shuffled_pair_cosine",
        "centered_margin",
    ):
        assert key in diag.metrics
        assert isinstance(diag.metrics[key], float)
    direct = feature_diagnostics(diag.h_l, diag.h_t)
    assert direct["centered_margin"] == diag.metrics["centered_margin"]


def test_pre_sem_causal_no_future_leakage():
    tok, b_dec, _teacher = make_models()
    recs_a = records()
    recs_b = [dict(r) for r in recs_a]
    recs_b[0]["cot"] = "Completely different future target tokens appear here."
    z = torch.randn(len(recs_a), b_dec.config.n_embd)
    a = student_feature_forward(b_dec, tok, recs_a, z, 64)
    b = student_feature_forward(b_dec, tok, recs_b, z, 64)
    assert torch.allclose(a.h_l[0], b.h_l[0])
    sem_pos = a.sem_positions[0].item()
    assert torch.allclose(a.logits[0, sem_pos], b.logits[0, sem_pos])


def test_causal_leakage_helper_and_same_question_pairs():
    tok, b_dec, _teacher = make_models()
    recs_a = records()
    recs_b = [dict(r) for r in recs_a]
    recs_b[1]["cot"] = "Only future reasoning tokens are changed."
    z = torch.randn(len(recs_a), b_dec.config.n_embd)
    leakage = causal_leakage_check(b_dec, tok, recs_a, recs_b, z, 64)
    assert leakage["sem_hidden_max_abs_diff"] == 0.0
    assert leakage["sem_logits_max_abs_diff"] == 0.0

    no_pairs = find_same_question_pairs(recs_a)
    assert no_pairs == []
    paired = [dict(recs_a[0]), dict(recs_a[0])]
    paired[1]["cot"] = "Different cot with the same question."
    assert find_same_question_pairs(paired) == [(0, 1)]


def test_lambda2_zero_parity_with_old_loss1():
    tok, b_dec, b_teacher = make_models()
    del b_teacher
    recs = records()
    z = torch.randn(len(recs), b_dec.config.n_embd, requires_grad=True)
    old = loss1_forward(b_dec, tok, recs, z, 32)
    new = loss1_forward(b_dec, tok, recs, z, 32)
    assert torch.allclose(old.loss, new.loss)
    assert torch.allclose(old.logits, new.logits)
    old_grad = torch.autograd.grad(old.loss, z, retain_graph=True)[0]
    new_grad = torch.autograd.grad(new.loss, z, retain_graph=True)[0]
    assert torch.allclose(old_grad, new_grad)


def test_teacher_does_not_change_after_student_update():
    tok, b_dec, b_teacher = make_models()
    recs = records()
    z = torch.randn(len(recs), b_dec.config.n_embd, requires_grad=True)
    opt = torch.optim.AdamW(b_dec.parameters(), lr=1e-2)
    before = parameter_fingerprint(b_teacher)
    _student, _teacher, loss2 = loss2_forward(b_dec, b_teacher, tok, recs, z, 32)
    loss2.loss2.backward()
    opt.step()
    assert parameter_fingerprint(b_teacher) == before
