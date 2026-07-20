from __future__ import annotations

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.htext.modeling import LATENT_TOKEN, THINKING_TOKEN, LatentProjector, h0_forward, h1_forward, setup_special_tokens


class TinyTokenizer:
    def __init__(self):
        words = [
            THINKING_TOKEN,
            LATENT_TOKEN,
            "Compute",
            "2",
            "plus",
            "3.",
            "3",
            "5",
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


def test_joint_loss1_reaches_model_a_thinking_hidden():
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
    projector = LatentProjector(16)
    records = [{"question": "Compute 2 plus 3.", "answer": "5", "cot": "2 plus 3 equals 5."}]
    main = h0_forward(model_a, tokenizer, records, 16, 8, 1)
    loss1 = h1_forward(model_b, tokenizer, records, main.thinking_hidden, projector, 16)
    grad_z = torch.autograd.grad(loss1.loss, main.thinking_hidden, retain_graph=True)[0]
    total = main.loss + 0.1 * loss1.loss
    total.backward()
    a_grad = sum(
        float(p.grad.detach().abs().sum().item())
        for p in model_a.parameters()
        if p.grad is not None
    )
    assert main.thinking_hidden.requires_grad
    assert torch.isfinite(grad_z).all()
    assert grad_z.norm().item() > 0
    assert a_grad > 0
