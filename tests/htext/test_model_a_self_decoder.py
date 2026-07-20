from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.htext.self_decoder import (
    SelfDecodeLatentInterface,
    build_self_decoder_batch,
    forward_model_a_text_only_self_decoder,
    prepare_self_decoder_latent,
)


class TinyTokenizer:
    def __init__(self):
        self.pad_token = "<PAD>"
        self.eos_token = "<EOS>"
        self.specials = ["<PAD>", "<EOS>", "<THINKING_OF_SUMMARY>", "<THINKING_OF_CAPTION>", "<THINKING_OF_REASONING>"]
        self.vocab = {tok: idx for idx, tok in enumerate(self.specials)}
        self.pad_token_id = self.vocab[self.pad_token]
        self.eos_token_id = self.vocab[self.eos_token]

    def _id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
        return self.vocab[token]

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        spaced = text
        for tok in self.specials:
            spaced = spaced.replace(tok, f" {tok} ")
        tokens = spaced.replace("\n", " ").split()
        return {"input_ids": [self._id(tok) for tok in tokens]}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._id(token)

    def __len__(self) -> int:
        return len(self.vocab)


class TinyCausalSelfModel(nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_size: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.mix = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(use_cache=False, hidden_size=hidden_size)
        self.last_kwargs = None

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        del kwargs
        self.last_kwargs = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
            "use_cache": use_cache,
            "return_dict": return_dict,
        }
        if inputs_embeds is None:
            assert input_ids is not None
            x = self.embed(input_ids)
        else:
            assert input_ids is None
            x = inputs_embeds
        if attention_mask is None:
            attention_mask = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
        mask = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
        causal_sum = torch.cumsum(x * mask, dim=1)
        denom = torch.cumsum(mask, dim=1).clamp_min(1.0)
        hidden = torch.tanh(self.mix(causal_sum / denom))
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        if output_hidden_states:
            hidden_states = (x, hidden)
        else:
            hidden_states = None
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=hidden_states)


@pytest.fixture()
def tokenizer():
    return TinyTokenizer()


@pytest.fixture()
def records():
    return [
        {"question": "what color is chart a", "summary": "chart a is blue", "caption": "blue bar", "reasoning": "the bar is blue"},
        {"question": "what color is the longer chart b item", "summary": "chart b is red", "caption": "red longer bar", "reasoning": "the red bar is longer"},
    ]


def make_batch(tokenizer, records, *, stage="summary", label_mode="text_only"):
    return build_self_decoder_batch(
        tokenizer=tokenizer,
        records=records,
        stage=stage,
        thinking_token=f"<THINKING_OF_{stage.upper()}>",
        target_key=stage,
        label_mode=label_mode,
        device=torch.device("cpu"),
    )


def test_text_only_forward_and_label_loss_parity(tokenizer, records):
    torch.manual_seed(0)
    batch = make_batch(tokenizer, records)
    model = TinyCausalSelfModel(vocab_size=256)
    model.eval()
    with torch.no_grad():
        input_path = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            output_hidden_states=True,
            use_cache=False,
        )
        embeds = model.get_input_embeddings()(batch.input_ids)
        embed_path = model(
            inputs_embeds=embeds,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            output_hidden_states=True,
            use_cache=False,
        )
    assert input_path.logits.shape == embed_path.logits.shape
    assert torch.allclose(input_path.logits, embed_path.logits, atol=0, rtol=0)
    assert torch.allclose(input_path.loss, embed_path.loss, atol=0, rtol=0)


def test_injected_latent_sensitivity_and_grad_to_z(tokenizer, records):
    torch.manual_seed(1)
    batch = make_batch(tokenizer, records)
    model = TinyCausalSelfModel(vocab_size=256)
    z1 = torch.randn(len(records), 16, requires_grad=True)
    z2 = (z1.detach() + 0.5).requires_grad_(True)
    out1 = forward_model_a_text_only_self_decoder(
        model_a=model,
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        labels=batch.labels,
        replacement_mask=batch.replacement_mask,
        injected_latent=z1,
    )
    out2 = forward_model_a_text_only_self_decoder(
        model_a=model,
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        labels=batch.labels,
        replacement_mask=batch.replacement_mask,
        injected_latent=z2,
    )
    rows = torch.arange(len(records))
    assert torch.allclose(out1.inputs_embeds[rows, batch.latent_slot_positions], z1)
    assert not torch.allclose(out1.inputs_embeds[rows, batch.latent_slot_positions], out2.inputs_embeds[rows, batch.latent_slot_positions])
    assert not torch.allclose(out1.logits, out2.logits)
    grad_z = torch.autograd.grad(out1.loss, z1, retain_graph=True)[0]
    assert torch.isfinite(grad_z).all()
    assert grad_z.norm() > 0


def test_cross_forward_grad_and_detach_control(tokenizer, records):
    torch.manual_seed(2)
    batch = make_batch(tokenizer, records)
    model = TinyCausalSelfModel(vocab_size=256)
    producer = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        labels=batch.labels,
        output_hidden_states=True,
        use_cache=False,
    )
    z = producer.hidden_states[-1][:, 0, :]
    out = forward_model_a_text_only_self_decoder(
        model_a=model,
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        labels=batch.labels,
        replacement_mask=batch.replacement_mask,
        injected_latent=prepare_self_decoder_latent(z, detach_latent=False),
    )
    grad_z = torch.autograd.grad(out.loss, z, retain_graph=True)[0]
    cross = torch.autograd.grad(z, model.get_input_embeddings().weight, grad_outputs=grad_z, retain_graph=True)[0]
    assert torch.isfinite(cross).all()
    assert cross.norm() > 0

    detached = forward_model_a_text_only_self_decoder(
        model_a=model,
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        labels=batch.labels,
        replacement_mask=batch.replacement_mask,
        injected_latent=prepare_self_decoder_latent(z, detach_latent=True),
    )
    assert torch.autograd.grad(detached.loss, z, allow_unused=True, retain_graph=True)[0] is None
    detached.loss.backward()
    assert model.get_input_embeddings().weight.grad is not None
    assert torch.isfinite(model.get_input_embeddings().weight.grad).all()
    assert model.get_input_embeddings().weight.grad.norm() > 0


def test_same_parameter_identity_and_no_image_assertion(tokenizer, records):
    batch = make_batch(tokenizer, records)
    model = TinyCausalSelfModel(vocab_size=256)
    producer_model = model
    self_model = model
    assert producer_model is self_model
    assert id(producer_model.get_input_embeddings().weight) == id(self_model.get_input_embeddings().weight)
    out = forward_model_a_text_only_self_decoder(
        model_a=self_model,
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        labels=batch.labels,
        replacement_mask=batch.replacement_mask,
        injected_latent=torch.randn(len(records), 16),
    )
    assert out.audit["passed_input_ids"] is False
    assert out.audit["pixel_values"] is None
    assert model.last_kwargs["input_ids"] is None
    assert model.last_kwargs["pixel_values"] is None
    assert model.last_kwargs["image_grid_thw"] is None
    assert model.last_kwargs["pixel_values_videos"] is None
    assert model.last_kwargs["video_grid_thw"] is None
    assert model.last_kwargs["use_cache"] is False


def test_padded_batch_label_masks_and_replacement_positions(tokenizer, records):
    batch = make_batch(tokenizer, records, stage="reasoning", label_mode="latent_and_text")
    assert batch.input_ids.shape[0] == 2
    assert batch.input_ids.shape == batch.labels.shape == batch.replacement_mask.shape
    assert batch.replacement_mask.sum().item() == 2
    assert batch.latent_slot_positions[0].item() != batch.latent_slot_positions[1].item()
    for row in range(2):
        slot = batch.latent_slot_positions[row]
        target_start = batch.target_start_positions[row]
        assert batch.labels[row, :target_start].eq(-100).logical_not().sum().item() == 1
        assert batch.labels[row, slot].item() == batch.token_id
        assert (batch.labels[row, target_start : target_start + batch.target_token_counts[row]] != -100).all()

    text_only = make_batch(tokenizer, records, stage="reasoning", label_mode="text_only")
    for row in range(2):
        assert text_only.labels[row, text_only.latent_slot_positions[row]].item() == -100
        assert (text_only.labels[row, : text_only.target_start_positions[row]] != -100).sum().item() == 0


def test_causal_no_future_leakage(tokenizer, records):
    torch.manual_seed(3)
    batch_a = make_batch(tokenizer, records)
    changed = [dict(records[0]), dict(records[1])]
    changed[0]["summary"] = "chart a is green and future tokens differ"
    batch_b = make_batch(tokenizer, changed)
    model = TinyCausalSelfModel(vocab_size=256)
    z = torch.randn(len(records), 16)
    out_a = forward_model_a_text_only_self_decoder(
        model_a=model,
        input_ids=batch_a.input_ids,
        attention_mask=batch_a.attention_mask,
        labels=batch_a.labels,
        replacement_mask=batch_a.replacement_mask,
        injected_latent=z,
    )
    out_b = forward_model_a_text_only_self_decoder(
        model_a=model,
        input_ids=batch_b.input_ids,
        attention_mask=batch_b.attention_mask,
        labels=batch_b.labels,
        replacement_mask=batch_b.replacement_mask,
        injected_latent=z,
    )
    first_target_predictor = batch_a.target_start_positions[0].item() - 1
    assert torch.allclose(out_a.logits[0, first_target_predictor], out_b.logits[0, first_target_predictor])


def test_lambda_zero_main_only_equivalence(tokenizer, records):
    torch.manual_seed(4)
    batch = make_batch(tokenizer, records)
    model = TinyCausalSelfModel(vocab_size=256)
    main_a = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        labels=batch.labels,
        output_hidden_states=True,
        use_cache=False,
    )
    lambda_self = 0.0
    z = torch.randn(len(records), 16, requires_grad=True)
    self_out = forward_model_a_text_only_self_decoder(
        model_a=model,
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        labels=batch.labels,
        replacement_mask=batch.replacement_mask,
        injected_latent=z,
    )
    total = main_a.loss + lambda_self * self_out.loss
    main_b = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        labels=batch.labels,
        output_hidden_states=True,
        use_cache=False,
    )
    assert torch.allclose(total, main_b.loss)
    assert torch.allclose(main_a.logits, main_b.logits)


def test_role_and_adapter_interface(tokenizer):
    torch.manual_seed(5)
    model = TinyCausalSelfModel(vocab_size=256)
    stages = ("summary", "caption", "reasoning")
    stage_token_ids = {stage: tokenizer.convert_tokens_to_ids(f"<THINKING_OF_{stage.upper()}>") for stage in stages}
    interface = SelfDecodeLatentInterface(
        16,
        stages,
        adapter_type="identity",
        role_mode="typed",
        token_embedding_weight=model.get_input_embeddings().weight,
        stage_token_ids=stage_token_ids,
    )
    z = torch.randn(2, 16)
    injected_summary, audit = interface("summary", z)
    injected_caption, _ = interface("caption", z)
    assert audit["role_mode"] == "typed"
    assert not torch.allclose(injected_summary[0], injected_summary[1])
    assert not torch.allclose(injected_summary, injected_caption)

    no_role = SelfDecodeLatentInterface(16, stages, adapter_type="identity", role_mode="none")
    nr_summary, _ = no_role("summary", z)
    nr_caption, _ = no_role("caption", z)
    assert torch.allclose(nr_summary, z)
    assert torch.allclose(nr_summary, nr_caption)


def test_sequential_batched_parity(tokenizer, records):
    torch.manual_seed(6)
    model_seq = TinyCausalSelfModel(vocab_size=256)
    model_batched = TinyCausalSelfModel(vocab_size=256)
    model_batched.load_state_dict(model_seq.state_dict())
    batch = make_batch(tokenizer, records)
    z = torch.randn(len(records), 16)

    seq_loss = torch.zeros(())
    for idx in range(len(records)):
        single = make_batch(tokenizer, [records[idx]])
        out = forward_model_a_text_only_self_decoder(
            model_a=model_seq,
            input_ids=single.input_ids,
            attention_mask=single.attention_mask,
            labels=single.labels,
            replacement_mask=single.replacement_mask,
            injected_latent=z[idx : idx + 1],
        )
        seq_loss = seq_loss + out.loss / len(records)
    seq_loss.backward()

    out_batched = forward_model_a_text_only_self_decoder(
        model_a=model_batched,
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        labels=batch.labels,
        replacement_mask=batch.replacement_mask,
        injected_latent=z,
    )
    out_batched.loss.backward()

    assert torch.allclose(seq_loss, out_batched.loss, atol=1e-6, rtol=1e-6)
    for p_seq, p_batched in zip(model_seq.parameters(), model_batched.parameters()):
        assert torch.allclose(p_seq.grad, p_batched.grad, atol=1e-6, rtol=1e-6)
