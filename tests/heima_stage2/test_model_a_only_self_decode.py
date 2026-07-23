from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from src.heima_stage2.model_a_only_self_decode import (
    AOnlySelfDecodeMode,
    FirstPassOutput,
    build_self_decode_features,
    causal_lm_loss,
    run_a_only_train_step,
)


class TinyTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def __init__(self) -> None:
        self.vocab = {"<pad>": 0, "<eos>": 1}

    def __call__(self, text: str, add_special_tokens: bool = False):
        ids = []
        for token in text.replace("\n", " \n ").split():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            ids.append(self.vocab[token])
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens: bool = False):
        inv = {v: k for k, v in self.vocab.items()}
        toks = [inv.get(int(i), "<unk>") for i in ids]
        if skip_special_tokens:
            toks = [t for t in toks if not t.startswith("<")]
        return " ".join(toks)


class ToyA(nn.Module):
    def __init__(self, vocab_size: int = 256, hidden: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.block = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, vocab_size)
        self.forward_calls = 0

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, output_hidden_states=True, use_cache=False):
        self.forward_calls += 1
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        token_hidden = torch.tanh(self.block(inputs_embeds))
        hidden = token_hidden.cumsum(dim=1)
        logits = self.head(hidden)
        return SimpleNamespace(logits=logits, hidden_states=(hidden,))


def records():
    return [
        {
            "question": "what color is the left bar",
            "summary": "find the relevant bar",
            "caption": "the chart has colored bars",
            "reasoning": "the left bar is blue",
            "answer": "blue",
        },
        {
            "question": "what is the larger value",
            "summary": "compare the two values",
            "caption": "two values are shown",
            "reasoning": "the right value is larger",
            "answer": "right",
        },
    ]


def make_first_pass(tokenizer: TinyTokenizer):
    def first_pass(model: ToyA, batch_records):
        rows, labels, positions = [], [], {"summary": [], "caption": [], "reasoning": []}
        for rec in batch_records:
            prompt = tokenizer(rec["question"] + " <sum> <cap> <reas> answer:")["input_ids"]
            answer = tokenizer(rec["answer"] + " " + tokenizer.eos_token)["input_ids"]
            ids = prompt + answer
            label = [-100] * len(prompt) + answer
            rows.append(ids)
            labels.append(label)
            positions["summary"].append(len(prompt) - 4)
            positions["caption"].append(len(prompt) - 3)
            positions["reasoning"].append(len(prompt) - 2)
        max_len = max(len(x) for x in rows)
        input_ids = torch.full((len(rows), max_len), tokenizer.pad_token_id, dtype=torch.long)
        label_tensor = torch.full((len(rows), max_len), -100, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for i, ids in enumerate(rows):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            label_tensor[i, : len(labels[i])] = torch.tensor(labels[i])
            attention[i, : len(ids)] = 1
        out = model(input_ids=input_ids, attention_mask=attention, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[-1]
        batch_idx = torch.arange(len(rows))
        latents = {}
        for section, pos in positions.items():
            pos_tensor = torch.tensor(pos, dtype=torch.long)
            z = hidden[batch_idx, pos_tensor]
            if z.requires_grad:
                z.retain_grad()
            latents[section] = z
        return FirstPassOutput(main_loss=causal_lm_loss(out.logits, label_tensor), latents=latents)

    return first_pass


def test_inputs_embeds_concat_shape_and_label_mask() -> None:
    tokenizer = TinyTokenizer()
    model = ToyA()
    batch = records()
    z = torch.randn(len(batch), 8, requires_grad=True)
    features = build_self_decode_features(
        model_a=model,
        tokenizer=tokenizer,
        records=batch,
        section="summary",
        z=z,
        max_q=20,
        max_target=20,
        include_latent=True,
    )
    assert features.inputs_embeds.shape[0] == len(batch)
    assert features.inputs_embeds.shape[-1] == 8
    assert features.labels.shape == features.attention_mask.shape
    for row, latent_pos in enumerate(features.latent_positions):
        assert latent_pos is not None
        target_count = int((features.labels[row] != -100).sum().item())
        assert target_count == features.target_lengths[row]
        prefix_end = features.prompt_lengths[row] + 1 + features.prefix_lengths[row]
        assert torch.all(features.labels[row, :prefix_end] == -100)
        assert features.attention_mask[row, latent_pos].item() == 1


def test_no_model_b_projector_or_role_embedding_audit_flags() -> None:
    tokenizer = TinyTokenizer()
    model = ToyA()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    out = run_a_only_train_step(
        model_a=model,
        optimizer_a=opt,
        records=records(),
        tokenizer=tokenizer,
        first_pass_fn=make_first_pass(tokenizer),
        mode=AOnlySelfDecodeMode.A_ONLY_SELF_DECODE,
        lambda_self=0.05,
        sections=("summary", "caption", "reasoning"),
    )
    assert out.has_model_b is False
    assert out.optimizer_contains_model_b is False
    assert out.use_projector is False
    assert out.use_role_embedding is False
    assert out.extra_trainable_params_except_A == 0


def test_n3_batch_uses_four_model_a_forwards() -> None:
    tokenizer = TinyTokenizer()
    model = ToyA()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    out = run_a_only_train_step(
        model_a=model,
        optimizer_a=opt,
        records=records(),
        tokenizer=tokenizer,
        first_pass_fn=make_first_pass(tokenizer),
        mode="a_only_self_decode",
        lambda_self=0.05,
        sections=("summary", "caption", "reasoning"),
    )
    assert out.expected_forward_count_per_batch == 4
    assert out.actual_forward_count_per_batch == 4
    assert model.forward_calls == 4


def test_self_decode_mode_sends_gradients_to_z_and_a() -> None:
    tokenizer = TinyTokenizer()
    model = ToyA()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    out = run_a_only_train_step(
        model_a=model,
        optimizer_a=opt,
        records=records(),
        tokenizer=tokenizer,
        first_pass_fn=make_first_pass(tokenizer),
        mode=AOnlySelfDecodeMode.A_ONLY_SELF_DECODE,
        lambda_self=0.05,
        sections=("summary", "caption", "reasoning"),
        step_optimizer=False,
    )
    assert out.grad_A_from_self_decode_norm > 0.0
    assert all(out.grad_z_norm[s] > 0.0 for s in ("summary", "caption", "reasoning"))


def test_baseline_mode_keeps_self_decode_gradient_zero() -> None:
    tokenizer = TinyTokenizer()
    model = ToyA()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    out = run_a_only_train_step(
        model_a=model,
        optimizer_a=opt,
        records=records(),
        tokenizer=tokenizer,
        first_pass_fn=make_first_pass(tokenizer),
        mode=AOnlySelfDecodeMode.A_ONLY_MAIN_BASELINE,
        lambda_self=0.05,
        sections=("summary", "caption", "reasoning"),
        step_optimizer=False,
    )
    assert out.grad_A_from_self_decode_norm == 0.0
    assert all(v == 0.0 for v in out.grad_z_norm.values())
    assert out.total_loss == out.main_loss
    assert model.forward_calls == 4


def test_one_batch_smoke_loss_finite() -> None:
    tokenizer = TinyTokenizer()
    model = ToyA()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    out = run_a_only_train_step(
        model_a=model,
        optimizer_a=opt,
        records=records(),
        tokenizer=tokenizer,
        first_pass_fn=make_first_pass(tokenizer),
        mode=AOnlySelfDecodeMode.A_ONLY_SELF_DECODE,
        lambda_self=0.05,
        sections=("summary", "caption", "reasoning"),
    )
    assert out.finite
    assert torch.isfinite(torch.tensor([out.main_loss, out.self_loss, out.total_loss])).all()


def test_latent_intervention_eval_has_required_conditions() -> None:
    from src.heima_stage2.model_a_only_self_decode import evaluate_self_decode_interventions

    tokenizer = TinyTokenizer()
    model = ToyA()
    metrics = evaluate_self_decode_interventions(
        model_a=model,
        tokenizer=tokenizer,
        records=records(),
        first_pass_fn=make_first_pass(tokenizer),
        sections=("summary", "caption", "reasoning"),
        max_q=20,
        max_target=20,
    )
    assert set(metrics) == {"summary", "caption", "reasoning"}
    for section_metrics in metrics.values():
        assert set(section_metrics) == {
            "correct",
            "shuffle",
            "zero",
            "q_only",
            "shuffle_margin",
            "zero_margin",
            "qz_gain",
        }
        assert all(torch.isfinite(torch.tensor(v)) for v in section_metrics.values())
