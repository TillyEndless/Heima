from __future__ import annotations

import torch

from .latent_reasoner import main_forward
from .whole_cot_decoder import loss1_forward


def make_intervention_latents(z: torch.Tensor) -> dict[str, torch.Tensor]:
    random = torch.randn_like(z)
    if z.numel() > 1:
        random = random * z.std().clamp_min(1e-6) + z.mean()
    order = torch.roll(torch.arange(z.size(0), device=z.device), shifts=-1)
    return {
        "normal": z,
        "zero": torch.zeros_like(z),
        "random": random,
        "shuffled": z[order],
    }


def answer_em_from_logits(logits, labels) -> float:
    mask = labels != -100
    if mask.sum().item() == 0:
        return 0.0
    pred = logits.argmax(dim=-1)
    shifted_pred = pred[:, :-1]
    shifted_labels = labels[:, 1:]
    shifted_mask = shifted_labels != -100
    if shifted_mask.sum().item() == 0:
        return 0.0
    per_row = []
    for i in range(labels.size(0)):
        row_mask = shifted_mask[i]
        if row_mask.sum().item() == 0:
            continue
        per_row.append(torch.equal(shifted_pred[i][row_mask], shifted_labels[i][row_mask]))
    return sum(bool(x) for x in per_row) / max(len(per_row), 1)


def evaluate_main(model_a, tokenizer, records, config) -> dict:
    model_a.eval()
    out = main_forward(
        model_a,
        tokenizer,
        records,
        config["max_question_tokens"],
        config["max_answer_tokens"],
    )
    return {
        "answer_nll": out.loss.item(),
        "answer_em": answer_em_from_logits(out.logits, out.labels),
    }


def evaluate_interventions(model_a, model_b, tokenizer, records, config) -> dict:
    model_a.eval()
    if model_b is not None:
        model_b.eval()
    with torch.enable_grad():
        main = main_forward(
            model_a,
            tokenizer,
            records,
            config["max_question_tokens"],
            config["max_answer_tokens"],
        )
        interventions = make_intervention_latents(main.z)
        answer = {}
        decode = {}
        for name, z_value in interventions.items():
            main_i = main_forward(
                model_a,
                tokenizer,
                records,
                config["max_question_tokens"],
                config["max_answer_tokens"],
                latent_override=z_value,
            )
            answer[name] = {
                "nll": main_i.loss.item(),
                "em": answer_em_from_logits(main_i.logits, main_i.labels),
            }
            if model_b is not None:
                loss1 = loss1_forward(
                    model_b,
                    tokenizer,
                    records,
                    main.z,
                    config["max_cot_tokens"],
                    latent_override=z_value,
                )
                decode[name] = {"nll": loss1.loss.item()}
        first_diff = None
        first_diff_shuffle = None
        if model_b is not None and "normal" in decode:
            normal = loss1_forward(model_b, tokenizer, records, main.z, config["max_cot_tokens"])
            zero = loss1_forward(
                model_b,
                tokenizer,
                records,
                main.z,
                config["max_cot_tokens"],
                latent_override=torch.zeros_like(main.z),
            )
            shuffled = loss1_forward(
                model_b,
                tokenizer,
                records,
                main.z,
                config["max_cot_tokens"],
                latent_override=torch.roll(main.z, shifts=-1, dims=0),
            )
            target_positions = normal.labels.ne(-100)
            first = target_positions.float().argmax(dim=1)[0].item()
            first_diff = (
                normal.logits[0, first - 1] - zero.logits[0, first - 1]
            ).abs().max().item()
            first_diff_shuffle = (
                normal.logits[0, first - 1] - shuffled.logits[0, first - 1]
            ).abs().max().item()
        return {
            "answer_intervention": answer,
            "decode_intervention": decode,
            "first_cot_token_logit_difference": first_diff,
            "first_cot_token_logit_diff_zero": first_diff,
            "first_cot_token_logit_diff_shuffle": first_diff_shuffle,
        }


def generate_cot_sample(
    model_b,
    tokenizer,
    z,
    latent_override: torch.Tensor | None = None,
    max_new_tokens: int = 48,
) -> str | None:
    if model_b is None:
        return None
    from .whole_cot_decoder import DECODER_PROMPT, prompt_ids_and_latent_pos, replace_latent_with_cat

    model_b.eval()
    device = next(model_b.parameters()).device
    prompt_ids, latent_pos = prompt_ids_and_latent_pos(tokenizer)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    token_embeds = model_b.get_input_embeddings()(input_ids)
    use_z = z[:1] if latent_override is None else latent_override[:1]
    inputs_embeds = replace_latent_with_cat(token_embeds, use_z, latent_pos)
    generated = model_b.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def sample_decodes(model_a, model_b, tokenizer, records, config, max_samples: int = 8) -> str:
    if model_b is None:
        return ""
    lines = []
    model_a.eval()
    model_b.eval()
    selected = records[:max_samples]
    with torch.enable_grad():
        main = main_forward(
            model_a,
            tokenizer,
            selected,
            config["max_question_tokens"],
            config["max_answer_tokens"],
        )
        shuffled = torch.roll(main.z, shifts=-1, dims=0)
        zero = torch.zeros_like(main.z)
        pred = main.logits.argmax(dim=-1)
        for idx, record in enumerate(selected, start=1):
            z_i = main.z[idx - 1 : idx]
            zero_i = zero[idx - 1 : idx]
            shuffled_i = shuffled[idx - 1 : idx]
            normal_text = generate_cot_sample(model_b, tokenizer, z_i) or ""
            zero_text = generate_cot_sample(model_b, tokenizer, z_i, zero_i) or ""
            shuffled_text = generate_cot_sample(model_b, tokenizer, z_i, shuffled_i) or ""
            pred = main.logits.argmax(dim=-1)
            label_mask = main.labels.ne(-100)
            pred_tokens = []
            for pos in torch.where(label_mask[idx - 1])[0].tolist():
                pred_tokens.append(int(pred[idx - 1, pos - 1].item()))
            predicted_answer = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
            lines.extend(
                [
                    f"Sample {idx}",
                    f"Question: {record['question']}",
                    f"Gold CoT: {record['cot']}",
                    f"Normal latent decoded CoT: {normal_text}",
                    f"Zero latent decoded CoT: {zero_text}",
                    f"Shuffled latent decoded CoT: {shuffled_text}",
                    f"Gold answer: {record['answer']}",
                    f"Predicted answer: {predicted_answer}",
                    "",
                ]
            )
    return "\n".join(lines)
