import math
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_gpt2_latent_ablation import (  # noqa: E402
    Example,
    THINK_TOKEN,
    compute_losses,
    make_self_decode_batch,
    replace_slot_embeddings,
)


class TinyTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1, THINK_TOKEN: 2}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"

    def encode(self, text, add_special_tokens=False):
        ids = []
        text = text.replace("\n", " \n ")
        for tok in text.split():
            if THINK_TOKEN in tok and tok != THINK_TOKEN:
                before, _, after = tok.partition(THINK_TOKEN)
                parts = [p for p in (before, THINK_TOKEN, after) if p]
            else:
                parts = [tok]
            for part in parts:
                if part not in self.vocab:
                    self.vocab[part] = len(self.vocab)
                ids.append(self.vocab[part])
        return ids


class TinyOutput:
    def __init__(self, logits, hidden_states=None):
        self.logits = logits
        self.hidden_states = hidden_states


class TinyLM(nn.Module):
    def __init__(self, vocab_size=512, hidden_size=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, output_hidden_states=False):
        embeds = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        hidden = embeds + 0.01 * torch.cumsum(embeds, dim=1)
        logits = self.lm_head(hidden)
        return TinyOutput(logits, (hidden,) if output_hidden_states else None)


def records():
    return [
        Example("What happens?", "The object falls because gravity acts on it.", "It falls."),
        Example("Why is it wet?", "Rain contacted the surface before the photo.", "Because of rain."),
    ]


def test_latent_slot_embedding_replacement_correct():
    tok = TinyTokenizer()
    model = TinyLM()
    batch = make_self_decode_batch(tok, records(), max_length=80, device=torch.device("cpu"))
    embeds = model.get_input_embeddings()(batch["input_ids"])
    z = torch.randn(embeds.size(0), embeds.size(-1), requires_grad=True)
    replaced = replace_slot_embeddings(embeds, z, batch["positions"], detach=False)
    for row, pos in enumerate(batch["positions"].tolist()):
        assert pos >= 0
        assert torch.allclose(replaced[row, pos], z[row])


def test_detach_makes_grad_z_zero():
    positions = torch.tensor([1])
    input_embeddings = torch.zeros(1, 3, 4)
    z = torch.ones(1, 4, requires_grad=True)
    replaced = replace_slot_embeddings(input_embeddings, z, positions, detach=True)
    loss = replaced.sum()
    assert not loss.requires_grad
    assert z.grad is None


def test_no_detach_makes_grad_z_positive():
    positions = torch.tensor([1])
    input_embeddings = torch.zeros(1, 3, 4)
    z = torch.ones(1, 4, requires_grad=True)
    replaced = replace_slot_embeddings(input_embeddings, z, positions, detach=False)
    loss = replaced.sum()
    loss.backward()
    assert z.grad is not None
    assert float(z.grad.abs().sum()) > 0


def test_latent_token_loss_gives_lm_head_gradient():
    tok = TinyTokenizer()
    model = TinyLM()
    out = compute_losses(model, tok, records(), "G1", 0.1, 0.05, 80, torch.device("cpu"))
    out.total_loss.backward()
    assert model.lm_head.weight.grad is not None
    assert float(model.lm_head.weight.grad.abs().sum()) > 0


def test_g0_g1_g2_g3_loss_shapes_are_scalar_and_finite():
    tok = TinyTokenizer()
    for group in ("G0", "G1", "G2", "G3"):
        model = TinyLM()
        out = compute_losses(model, tok, records(), group, 0.1, 0.05, 80, torch.device("cpu"))
        assert out.total_loss.shape == torch.Size([])
        assert math.isfinite(float(out.total_loss.detach()))
        out.total_loss.backward()
