from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from .modeling import (
    LatentProjector,
    _decoder_prompt_ids,
    extract_thinking_hidden,
    h0_answer_from_hidden_forward,
    h0_forward,
    h1_forward,
)
from .trainer import _answer_em, _build_projector, _load_model_a_checkpoint, _load_tokenizer_and_model, _thinking_state_mode


NUM_RE = re.compile(r"-?\d+")


def write_json(path: str | Path, obj: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def derangement_indices(n: int, device) -> torch.Tensor:
    if n < 2:
        raise ValueError("derangement requires at least two examples")
    return torch.roll(torch.arange(n, device=device), shifts=1)


def hidden_geometry(z: torch.Tensor) -> dict:
    flat = z.detach().float().reshape(-1, z.size(-1))
    normed = F.normalize(flat, dim=-1)
    cosine = normed @ normed.T
    if cosine.size(0) > 1:
        mask = ~torch.eye(cosine.size(0), dtype=torch.bool, device=cosine.device)
        pair = cosine[mask]
        pair_mean = float(pair.mean().item())
        pair_std = float(pair.std(unbiased=False).item())
    else:
        pair_mean = 1.0
        pair_std = 0.0
    centered = flat - flat.mean(dim=0, keepdim=True)
    var = centered.var(dim=0, unbiased=False)
    singular = torch.linalg.svdvals(centered)
    power = singular.pow(2)
    probs = power / power.sum().clamp_min(1e-12)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    return {
        "pairwise_cosine_mean": pair_mean,
        "pairwise_cosine_std": pair_std,
        "effective_rank": float(torch.exp(entropy).item()),
        "within_dataset_variance_mean": float(var.mean().item()),
        "within_dataset_variance_sum": float(var.sum().item()),
    }


def token_nlls(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~valid, 0)
    nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        safe_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)
    return nll, valid


def _category_masks(tokenizer, labels: torch.Tensor, records: list[dict]) -> dict[str, torch.Tensor]:
    shifted = labels[:, 1:].contiguous()
    masks = {
        "numeric": torch.zeros_like(shifted, dtype=torch.bool),
        "operator": torch.zeros_like(shifted, dtype=torch.bool),
        "intermediate": torch.zeros_like(shifted, dtype=torch.bool),
    }
    for i, record in enumerate(records):
        inter = set(record.get("intermediate_results", []))
        for j in torch.where(shifted[i] != -100)[0].tolist():
            token_text = tokenizer.decode([int(shifted[i, j].item())])
            stripped = token_text.strip()
            if any(ch.isdigit() for ch in token_text):
                masks["numeric"][i, j] = True
            if any(op in token_text for op in ["+", "-", "*", "/", "="]):
                masks["operator"][i, j] = True
            if stripped and any(stripped in value or value in stripped for value in inter):
                masks["intermediate"][i, j] = True
    return masks


def _mean_masked(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    if mask.sum().item() == 0:
        return None
    return float(values[mask].mean().item())


def cot_nll_breakdown(tokenizer, records: list[dict], logits: torch.Tensor, labels: torch.Tensor) -> dict:
    nll, valid = token_nlls(logits, labels)
    out = {"full": _mean_masked(nll, valid)}
    first_positions = torch.zeros_like(valid)
    first4 = torch.zeros_like(valid)
    first8 = torch.zeros_like(valid)
    for i in range(valid.size(0)):
        pos = torch.where(valid[i])[0]
        if pos.numel():
            first_positions[i, pos[:1]] = True
            first4[i, pos[:4]] = True
            first8[i, pos[:8]] = True
    out["first_token"] = _mean_masked(nll, first_positions)
    out["first_4_tokens"] = _mean_masked(nll, first4)
    out["first_8_tokens"] = _mean_masked(nll, first8)
    for name, mask in _category_masks(tokenizer, labels, records).items():
        out[f"{name}_tokens"] = _mean_masked(nll, mask & valid)
    return out


def logits_kl(normal_logits: torch.Tensor, other_logits: torch.Tensor, labels: torch.Tensor) -> dict:
    shifted_labels = labels[:, 1:]
    valid = shifted_labels != -100
    p = F.log_softmax(normal_logits[:, :-1, :], dim=-1)
    q = F.log_softmax(other_logits[:, :-1, :], dim=-1)
    kl = (p.exp() * (p - q)).sum(dim=-1)
    first = torch.zeros_like(valid)
    for i in range(valid.size(0)):
        pos = torch.where(valid[i])[0]
        if pos.numel():
            first[i, pos[0]] = True
    return {
        "first_token_logits_kl": _mean_masked(kl, first),
        "average_logits_kl": _mean_masked(kl, valid),
    }


def evaluate_h0_checkpoint(config: dict, checkpoint_dir: str | Path, records: list[dict]) -> dict:
    tokenizer, model_a, _, _ = _load_tokenizer_and_model(config, with_b=False)
    _load_model_a_checkpoint(checkpoint_dir, model_a)
    model_a.eval()
    chunks = []
    hidden_chunks = []
    positions = None
    with torch.no_grad():
        for start in range(0, len(records), config["eval_batch_size"]):
            batch = records[start : start + config["eval_batch_size"]]
            out = h0_forward(
                model_a,
                tokenizer,
                batch,
                config["max_question_tokens"],
                config["max_answer_tokens"],
                config["num_thinking_tokens"],
                _thinking_state_mode(config),
            )
            chunks.append(out)
            hidden_chunks.append(out.thinking_hidden.detach().cpu())
            positions = out.thinking_positions.detach().cpu()
    loss = sum(float(out.loss.item()) * out.labels.size(0) for out in chunks) / len(records)
    em = sum(_answer_em(out.logits, out.labels) * out.labels.size(0) for out in chunks) / len(records)
    from .modeling import THINKING_TOKEN

    thinking_id = tokenizer.convert_tokens_to_ids(THINKING_TOKEN)
    correct = 0
    total = 0
    for out in chunks:
        pred = out.logits[:, :-1].argmax(dim=-1)
        shifted_labels = out.labels[:, 1:]
        mask = shifted_labels == thinking_id
        correct += int((pred[mask] == thinking_id).sum().item())
        total += int(mask.sum().item())
    z = torch.cat(hidden_chunks, dim=0)
    answer_interventions = evaluate_h0_answer_interventions(config, checkpoint_dir, records[: config["intervention_samples"]])
    return {
        "main_nll": loss,
        "main_token_em": em,
        "answer_nll": answer_interventions["normal"]["answer_nll"],
        "answer_em": answer_interventions["normal"]["answer_em"],
        "thinking_token_accuracy": correct / max(total, 1),
        "thinking_positions_first_batch": positions.tolist() if positions is not None else None,
        "thinking_hidden_geometry": hidden_geometry(z),
        "answer_interventions": answer_interventions,
    }


def evaluate_h0_answer_interventions(config: dict, checkpoint_dir: str | Path, records: list[dict]) -> dict:
    tokenizer, model_a, _, _ = _load_tokenizer_and_model(config, with_b=False)
    _load_model_a_checkpoint(checkpoint_dir, model_a)
    model_a.eval()
    with torch.no_grad():
        z, _ = extract_thinking_hidden(
            model_a,
            tokenizer,
            records,
            config["max_question_tokens"],
            config["num_thinking_tokens"],
            _thinking_state_mode(config),
        )
        z0 = z[:, 0, :]
        variants = {
            "normal": z0,
            "shuffled": z0[derangement_indices(z0.size(0), z0.device)],
            "zero": torch.zeros_like(z0),
        }
        out = {}
        for name, value in variants.items():
            pred = h0_answer_from_hidden_forward(
                model_a,
                tokenizer,
                records,
                value,
                config["max_question_tokens"],
                config["max_answer_tokens"],
            )
            out[name] = {"answer_nll": float(pred.loss.item()), "answer_em": _answer_em(pred.logits, pred.labels)}
        return out


def _load_h1_models(config: dict, h0_checkpoint_dir: str | Path, h1_checkpoint_path: str | Path):
    tokenizer, model_a, model_b, device = _load_tokenizer_and_model(config, with_b=True)
    _load_model_a_checkpoint(h0_checkpoint_dir, model_a)
    for p in model_a.parameters():
        p.requires_grad_(False)
    projector = _build_projector(config, model_a.config.n_embd).to(device)
    ckpt = torch.load(h1_checkpoint_path, map_location=device)
    model_b.load_state_dict(ckpt["model_b"], strict=True)
    projector.load_state_dict(ckpt["projector"], strict=True)
    model_a.eval()
    model_b.eval()
    projector.eval()
    return tokenizer, model_a, model_b, projector


def make_latent_variants(z0: torch.Tensor) -> dict[str, torch.Tensor]:
    norm = z0.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    random = torch.randn_like(z0)
    random = F.normalize(random, dim=-1) * norm
    normed = F.normalize(z0, dim=-1)
    cosine = normed @ normed.T
    cosine.fill_diagonal_(math.inf)
    farthest = cosine.argmin(dim=1)
    mean = z0.mean(dim=0, keepdim=True).expand_as(z0)
    return {
        "normal": z0,
        "shuffled": z0[derangement_indices(z0.size(0), z0.device)],
        "farthest": z0[farthest],
        "zero": torch.zeros_like(z0),
        "norm_matched_random": random,
        "global_mean": mean,
    }


def evaluate_h1_interventions(config: dict, h0_checkpoint_dir: str | Path, h1_checkpoint_path: str | Path, records: list[dict], mode: str) -> dict:
    tokenizer, model_a, model_b, projector = _load_h1_models(config, h0_checkpoint_dir, h1_checkpoint_path)
    with torch.no_grad():
        z, _ = extract_thinking_hidden(
            model_a,
            tokenizer,
            records,
            config["max_question_tokens"],
            config["num_thinking_tokens"],
            _thinking_state_mode(config),
        )
        z0 = z[:, 0, :]
        variants = make_latent_variants(z0)
        if torch.equal(variants["normal"], variants["shuffled"]):
            raise RuntimeError("normal and shuffled latents are identical")
        normal_out = None
        out = {}
        for name, value in variants.items():
            pred = h1_forward(
                model_b,
                tokenizer,
                records,
                z,
                projector,
                config["max_cot_tokens"],
                latent_override=value,
                mode=mode,
            )
            breakdown = cot_nll_breakdown(tokenizer, records, pred.logits, pred.labels)
            item = {"nll": breakdown}
            if name == "normal":
                normal_out = pred
            else:
                item["logits_kl_from_normal"] = logits_kl(normal_out.logits, pred.logits, pred.labels) if normal_out is not None else None
            out[name] = item
        return out


def generate_h1_samples(config: dict, h0_checkpoint_dir: str | Path, h1_checkpoint_path: str | Path, records: list[dict], mode: str, max_new_tokens: int = 80) -> list[dict]:
    tokenizer, model_a, model_b, projector = _load_h1_models(config, h0_checkpoint_dir, h1_checkpoint_path)
    device = next(model_b.parameters()).device
    samples = []
    with torch.no_grad():
        z, _ = extract_thinking_hidden(
            model_a,
            tokenizer,
            records,
            config["max_question_tokens"],
            config["num_thinking_tokens"],
            _thinking_state_mode(config),
        )
        if len(records) < 2:
            variants = {"normal": z[:, 0, :], "shuffled": torch.zeros_like(z[:, 0, :])}
        else:
            variants = make_latent_variants(z[:, 0, :])
        for idx, record in enumerate(records):
            operation_type = record.get("operation_type", record.get("metadata", {}).get("operation_type", "unknown"))
            row = {"id": record["id"], "question": record["question"], "gold_cot": record["cot"], "answer": record["answer"], "operation_type": operation_type, "generations": {}}
            for name in ["normal", "shuffled"]:
                prompt_ids, latent_pos = _decoder_prompt_ids(tokenizer, record["question"], mode)
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                if mode == "q":
                    generated = model_b.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                    )
                    text = tokenizer.decode(generated[0, input_ids.size(1) :], skip_special_tokens=True)
                else:
                    embeds = model_b.get_input_embeddings()(input_ids)
                    projected = projector(variants[name][idx : idx + 1])
                    assert latent_pos is not None
                    inputs_embeds = torch.cat([embeds[:, :latent_pos, :], projected.unsqueeze(1), embeds[:, latent_pos + 1 :, :]], dim=1)
                    generated = model_b.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=torch.ones_like(input_ids),
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                    )
                    text = tokenizer.decode(generated[0], skip_special_tokens=True)
                row["generations"][name] = {
                    "text": text,
                    "number_match": record["answer"] in NUM_RE.findall(text),
                    "intermediate_result_match": any(value in NUM_RE.findall(text) for value in record.get("intermediate_results", [])),
                    "expression_match": any(op in text for op in ["+", "-", "*", "/", "="]),
                    "operation_type_match": operation_type_match(operation_type, text),
                }
            samples.append(row)
    return samples


def operation_type_match(operation_type: str, text: str) -> bool:
    has_mul = "*" in text or "multiply" in text.lower() or "product" in text.lower()
    has_add = "+" in text or "add" in text.lower() or "sum" in text.lower()
    has_sub = "-" in text or "subtract" in text.lower() or "minus" in text.lower()
    has_div = "/" in text or "divide" in text.lower() or "quotient" in text.lower()
    if operation_type == "add_mul":
        return has_add and has_mul
    if operation_type == "mul_sub":
        return has_mul and has_sub
    if operation_type == "sub_div":
        return has_sub and has_div
    if operation_type == "mixed_add_mul_sub":
        return has_add and has_mul and has_sub
    return False
