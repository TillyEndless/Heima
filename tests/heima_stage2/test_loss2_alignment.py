from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from src.heima_stage2.loss2_alignment import compute_grad_norm, loss2_forward


THINK = {"summary": "<S>", "caption": "<C>", "reasoning": "<R>"}
SECTIONS = ("summary", "caption", "reasoning")


class TinyTokenizer:
    eos_token = "<eos>"
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.vocab = {self.pad_token: 0, self.eos_token: 1}
        for tok in THINK.values():
            self.vocab[tok] = len(self.vocab)

    def __call__(self, text: str, add_special_tokens: bool = False):
        ids = []
        for token in text.replace("\n", " ").split():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            ids.append(self.vocab[token])
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens: bool = False):
        inv = {v: k for k, v in self.vocab.items()}
        toks = [inv[int(i)] for i in ids]
        if skip_special_tokens:
            toks = [t for t in toks if not t.startswith("<")]
        return " ".join(toks)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab[token]


class TinyB(nn.Module):
    def __init__(self, vocab_size: int = 256, hidden: int = 10) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.mix = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, vocab_size)
        self.config = SimpleNamespace(hidden_size=hidden)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, output_hidden_states=True, use_cache=False):
        if inputs_embeds is None:
            x = self.embed(input_ids)
        else:
            x = inputs_embeds
        if attention_mask is None:
            attention_mask = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
        hidden = torch.tanh(self.mix((x * attention_mask.unsqueeze(-1)).cumsum(dim=1)))
        logits = self.head(hidden)
        return SimpleNamespace(logits=logits, hidden_states=(hidden,))


class TinyA(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 6) -> None:
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden)

    def forward(self, x):
        return self.encoder(x)


def freeze(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def records():
    return [
        {"question": "q one", "summary": "sum one", "caption": "cap one", "reasoning": "reason one"},
        {"question": "q two", "summary": "sum two", "caption": "cap two", "reasoning": "reason two"},
    ]


def make_case():
    torch.manual_seed(5)
    tokenizer = TinyTokenizer()
    model_a = TinyA()
    model_b = TinyB()
    projector = nn.Linear(6, 10)
    freeze(model_b)
    freeze(projector)
    x = torch.randn(2, 4)
    z = model_a(x)
    z.retain_grad()
    return tokenizer, model_a, model_b, projector, z


def run_loss2(pool="mean", detach_latent=False):
    tokenizer, model_a, model_b, projector, z = make_case()
    out = loss2_forward(
        model_b=model_b,
        projector=projector,
        tokenizer=tokenizer,
        records=records(),
        section="reasoning",
        sections=SECTIONS,
        z=z,
        max_q=20,
        max_target=20,
        thinking_token=THINK["reasoning"],
        pool=pool,
        detach_latent=detach_latent,
    )
    return tokenizer, model_a, model_b, projector, z, out


def test_loss2_hidden_shapes_match_and_finite() -> None:
    _tok, _a, _b, _p, _z, out = run_loss2()
    assert out.latent_shape == out.text_shape == (2, 10)
    assert torch.isfinite(out.loss)


def test_h_text_detached() -> None:
    _tok, _a, _b, _p, _z, out = run_loss2()
    assert out.h_text_detached is True
    assert out.h_text.requires_grad is False
    assert out.h_latent.requires_grad is True


def test_b_parameters_frozen_and_optimizer_excludes_b() -> None:
    _tok, model_a, model_b, _p, _z, _out = run_loss2()
    opt = torch.optim.SGD(model_a.parameters(), lr=0.01)
    b_ids = {id(p) for p in model_b.parameters()}
    assert all(not p.requires_grad for p in model_b.parameters())
    assert all(id(p) not in b_ids for group in opt.param_groups for p in group["params"])


def test_baseline_grad_a_from_loss2_zero() -> None:
    _tok, model_a, model_b, projector, _z, out = run_loss2(detach_latent=True)
    if out.loss.requires_grad:
        out.loss.backward()
    grad_a, _ = compute_grad_norm(model_a.parameters())
    grad_b, _ = compute_grad_norm(model_b.parameters())
    grad_p, _ = compute_grad_norm(projector.parameters())
    assert grad_a == 0.0
    assert grad_b == 0.0
    assert grad_p == 0.0


def test_ours_loss1_loss2_grad_a_from_loss2_positive_b_zero() -> None:
    _tok, model_a, model_b, projector, z, out = run_loss2(detach_latent=False)
    grad_z = torch.autograd.grad(out.loss, z, retain_graph=True)[0]
    out.loss.backward()
    grad_a, finite_a = compute_grad_norm(model_a.parameters())
    grad_b, finite_b = compute_grad_norm(model_b.parameters())
    grad_p, finite_p = compute_grad_norm(projector.parameters())
    assert torch.isfinite(grad_z).all() and grad_z.norm() > 0
    assert grad_a > 0.0 and finite_a
    assert grad_b == 0.0 and finite_b
    assert grad_p == 0.0 and finite_p


def test_loss2_does_not_compare_a_hidden_to_b_hidden() -> None:
    _tok, _model_a, _model_b, _projector, z, out = run_loss2()
    assert out.h_latent.shape[-1] == 10
    assert out.h_text.shape[-1] == 10
    assert z.shape[-1] == 6
    assert out.h_latent.shape[-1] != z.shape[-1]


def test_pool_last_and_mean_both_run() -> None:
    for pool in ("mean", "last"):
        _tok, _a, _b, _p, _z, out = run_loss2(pool=pool)
        assert out.pool == pool
        assert torch.isfinite(out.loss)


def test_one_batch_smoke_total_loss_finite() -> None:
    _tok, model_a, model_b, projector, z, out = run_loss2()
    main = z.pow(2).mean()
    loss1 = z.abs().mean()
    total = main + 0.1 * loss1 + 0.05 * out.loss
    total.backward()
    grad_a, finite = compute_grad_norm(model_a.parameters())
    grad_b, _ = compute_grad_norm(model_b.parameters())
    grad_p, _ = compute_grad_norm(projector.parameters())
    assert torch.isfinite(total)
    assert grad_a > 0.0 and finite
    assert grad_b == 0.0
    assert grad_p == 0.0
