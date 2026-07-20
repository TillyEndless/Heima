from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from .latent_reasoner import causal_lm_loss, tokenize_text
from .whole_cot_decoder import LATENT_TOKEN, replace_latent_with_cat


SEM_TOKEN = "<SEM>"
Loss2Distance = Literal["cosine", "mse", "normalized_mse"]
Loss2Aggregate = Literal["mean", "sum"]
Loss2FeatureMode = Literal["pre_sem", "post_cot"]
TeacherContextMode = Literal["cumulative", "section_only"]


@dataclass(frozen=True)
class StudentFeatureOutput:
    h_l: torch.Tensor
    loss1: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    inputs_embeds: torch.Tensor
    sem_positions: torch.Tensor
    latent_pos: int


@dataclass(frozen=True)
class TeacherFeatureOutput:
    h_t: torch.Tensor
    sem_positions: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True)
class Loss2Output:
    loss2: torch.Tensor
    h_l: torch.Tensor
    h_t: torch.Tensor
    per_sample: torch.Tensor
    distance: str
    aggregate: str


@dataclass(frozen=True)
class Loss2DiagnosticOutput:
    metrics: dict
    h_l: torch.Tensor
    h_t: torch.Tensor


def ensure_sem_token(tokenizer, *models) -> int:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    before = len(tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": [SEM_TOKEN]})
    sem_id = tokenizer.convert_tokens_to_ids(SEM_TOKEN)
    if sem_id is None or sem_id < 0:
        raise ValueError("failed to register <SEM>")
    if len(tokenizer) != before:
        for model in models:
            model.resize_token_embeddings(len(tokenizer))
    return sem_id


def freeze_teacher(model_b_teacher) -> None:
    model_b_teacher.eval()
    for p in model_b_teacher.parameters():
        p.requires_grad_(False)


def parameter_fingerprint(model) -> str:
    h = hashlib.sha256()
    for _, p in model.state_dict().items():
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def assert_teacher_frozen_and_excluded(model_b_teacher, optimizer=None) -> None:
    if any(p.requires_grad for p in model_b_teacher.parameters()):
        raise AssertionError("teacher has trainable parameters")
    if optimizer is not None:
        teacher_ids = {id(p) for p in model_b_teacher.parameters()}
        for group in optimizer.param_groups:
            if any(id(p) in teacher_ids for p in group["params"]):
                raise AssertionError("teacher parameter is in optimizer")


def assert_student_teacher_independent(model_b_dec, model_b_teacher) -> None:
    dec_ids = {id(p) for p in model_b_dec.parameters()}
    teacher_ids = {id(p) for p in model_b_teacher.parameters()}
    if not dec_ids.isdisjoint(teacher_ids):
        raise AssertionError("B_dec and B_teacher share Parameter objects")


def _pad_rows(tokenizer, rows: list[list[int]], label_rows: list[list[int]] | None, device):
    max_len = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100) if label_rows is not None else None
    for i, row in enumerate(rows):
        n = len(row)
        input_ids[i, :n] = torch.tensor(row, dtype=torch.long, device=device)
        attention_mask[i, :n] = 1
        if labels is not None:
            labels[i, : len(label_rows[i])] = torch.tensor(label_rows[i], dtype=torch.long, device=device)
    return input_ids, attention_mask, labels


def _one_position(input_ids: torch.Tensor, token_id: int, name: str) -> torch.Tensor:
    mask = input_ids.eq(token_id)
    count = mask.sum(dim=1)
    if not torch.all(count == 1):
        raise ValueError(f"expected exactly one {name} per sample, got {count.tolist()}")
    return mask.long().argmax(dim=1)


def _one_position_before(input_ids: torch.Tensor, token_id: int, before_positions: torch.Tensor, name: str) -> torch.Tensor:
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1)
    mask = input_ids.eq(token_id) & positions.lt(before_positions.view(-1, 1))
    count = mask.sum(dim=1)
    if not torch.all(count == 1):
        raise ValueError(f"expected exactly one {name} before boundary per sample, got {count.tolist()}")
    return mask.long().argmax(dim=1)


def student_sem_prompt(record: dict) -> str:
    return (
        f"Question:\n{record['question']}\n\n"
        "Instruction:\nDecode the complete reasoning encoded in the latent state.\n\n"
        "Latent:\n"
        f"{LATENT_TOKEN}\n\n"
        "Semantic:\n"
        f"{SEM_TOKEN}\n\n"
        "Reasoning:\n"
    )


def teacher_sem_prompt(record: dict, *, context_mode: TeacherContextMode) -> str:
    if context_mode == "cumulative":
        cot_text = record["cot"]
    elif context_mode == "section_only":
        cot_text = record["cot"]
    else:
        raise ValueError(context_mode)
    return f"Question:\n{record['question']}\n\nExplicit reasoning:\n{cot_text}\n\n{SEM_TOKEN}"


def student_feature_forward(
    model_b_dec,
    tokenizer,
    records: list[dict],
    z: torch.Tensor,
    max_cot_tokens: int,
    *,
    layer_index: int = -1,
    latent_override: torch.Tensor | None = None,
    feature_mode: Loss2FeatureMode = "pre_sem",
) -> StudentFeatureOutput:
    if feature_mode not in ("pre_sem", "post_cot"):
        raise ValueError(feature_mode)
    device = next(model_b_dec.parameters()).device
    latent_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
    sem_id = tokenizer.convert_tokens_to_ids(SEM_TOKEN)
    if sem_id is None or sem_id < 0:
        raise ValueError("<SEM> is not registered")
    use_z = z if latent_override is None else latent_override
    rows, label_rows = [], []
    for rec in records:
        prompt_ids = tokenize_text(tokenizer, student_sem_prompt(rec))
        cot_ids = tokenize_text(tokenizer, rec["cot"] + tokenizer.eos_token, max_cot_tokens)
        rows.append(prompt_ids + cot_ids)
        labels = [-100] * len(prompt_ids) + cot_ids
        label_rows.append(labels)
    input_ids, attention_mask, labels = _pad_rows(tokenizer, rows, label_rows, device)
    sem_positions = _one_position(input_ids, sem_id, SEM_TOKEN)
    latent_positions = _one_position_before(input_ids, latent_id, sem_positions, LATENT_TOKEN)
    if not torch.all(sem_positions > latent_positions):
        raise ValueError("<SEM> must appear after latent slot")
    if int((labels != -100).sum().item()) == 0:
        raise RuntimeError("zero non-ignored Loss1 labels")
    if not torch.all(latent_positions == latent_positions[0]):
        raise ValueError("current replace_latent_with_cat requires fixed latent position")
    latent_pos = int(latent_positions[0].item())
    token_embeds = model_b_dec.get_input_embeddings()(input_ids)
    inputs_embeds = replace_latent_with_cat(token_embeds, use_z, latent_pos)
    out = model_b_dec(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states[layer_index]
    batch = torch.arange(input_ids.shape[0], device=device)
    if feature_mode == "pre_sem":
        feature_positions = sem_positions
    else:
        feature_positions = attention_mask.long().sum(dim=1) - 1
    h_l = hidden[batch, feature_positions, :]
    return StudentFeatureOutput(
        h_l=h_l,
        loss1=causal_lm_loss(out.logits, labels),
        logits=out.logits,
        labels=labels,
        inputs_embeds=inputs_embeds,
        sem_positions=sem_positions,
        latent_pos=latent_pos,
    )


@torch.no_grad()
def teacher_feature_forward(
    model_b_teacher,
    tokenizer,
    records: list[dict],
    *,
    layer_index: int = -1,
    context_mode: TeacherContextMode = "cumulative",
) -> TeacherFeatureOutput:
    device = next(model_b_teacher.parameters()).device
    sem_id = tokenizer.convert_tokens_to_ids(SEM_TOKEN)
    rows = [tokenize_text(tokenizer, teacher_sem_prompt(rec, context_mode=context_mode)) for rec in records]
    input_ids, attention_mask, _labels = _pad_rows(tokenizer, rows, None, device)
    sem_positions = _one_position(input_ids, sem_id, SEM_TOKEN)
    out = model_b_teacher(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states[layer_index]
    batch = torch.arange(input_ids.shape[0], device=device)
    h_t = hidden[batch, sem_positions, :].detach()
    return TeacherFeatureOutput(h_t=h_t, sem_positions=sem_positions, logits=out.logits)


def loss2_distance(h_l: torch.Tensor, h_t: torch.Tensor, distance: Loss2Distance) -> torch.Tensor:
    if h_l.shape != h_t.shape:
        raise ValueError(f"student/teacher feature shape mismatch: {tuple(h_l.shape)} vs {tuple(h_t.shape)}")
    if distance == "cosine":
        return 1.0 - F.cosine_similarity(h_l.float(), h_t.float(), dim=-1)
    if distance == "mse":
        return (h_l.float() - h_t.float()).pow(2).mean(dim=-1)
    if distance == "normalized_mse":
        ln = F.normalize(h_l.float(), dim=-1)
        tn = F.normalize(h_t.float(), dim=-1)
        return (ln - tn).pow(2).mean(dim=-1)
    raise ValueError(distance)


def aggregate_loss2(per_sample: torch.Tensor, aggregate: Loss2Aggregate) -> torch.Tensor:
    if aggregate == "mean":
        return per_sample.mean()
    if aggregate == "sum":
        return per_sample.sum()
    raise ValueError(aggregate)


def loss2_forward(
    model_b_dec,
    model_b_teacher,
    tokenizer,
    records: list[dict],
    z: torch.Tensor,
    max_cot_tokens: int,
    *,
    distance: Loss2Distance = "cosine",
    aggregate: Loss2Aggregate = "mean",
    layer_index: int = -1,
    detach_latent: bool = False,
    teacher_context_mode: TeacherContextMode = "cumulative",
    feature_mode: Loss2FeatureMode = "pre_sem",
) -> tuple[StudentFeatureOutput, TeacherFeatureOutput, Loss2Output]:
    z_for_student = z.detach() if detach_latent else z
    student = student_feature_forward(
        model_b_dec,
        tokenizer,
        records,
        z_for_student,
        max_cot_tokens,
        layer_index=layer_index,
        feature_mode=feature_mode,
    )
    teacher = teacher_feature_forward(
        model_b_teacher,
        tokenizer,
        records,
        layer_index=layer_index,
        context_mode=teacher_context_mode,
    )
    per = loss2_distance(student.h_l, teacher.h_t, distance)
    return student, teacher, Loss2Output(
        loss2=aggregate_loss2(per, aggregate),
        h_l=student.h_l,
        h_t=teacher.h_t,
        per_sample=per,
        distance=distance,
        aggregate=aggregate,
    )


def mean_pairwise_cosine(x: torch.Tensor) -> float:
    if x.shape[0] < 2:
        return 1.0
    xn = F.normalize(x.float(), dim=-1)
    sim = xn @ xn.T
    mask = ~torch.eye(x.shape[0], dtype=torch.bool, device=x.device)
    return float(sim[mask].mean().item())


def paired_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def centered(x: torch.Tensor) -> torch.Tensor:
    return x.float() - x.float().mean(dim=0, keepdim=True)


def feature_diagnostics(h_l: torch.Tensor, h_t: torch.Tensor) -> dict:
    if h_l.shape != h_t.shape:
        raise ValueError("feature diagnostics require matching shapes")
    correct = paired_cosine(h_l, h_t)
    shuffled_h_t = torch.roll(h_t, shifts=1, dims=0) if h_t.shape[0] > 1 else h_t
    shuffled = paired_cosine(h_l, shuffled_h_t)
    h_l_c = centered(h_l)
    h_t_c = centered(h_t)
    shuffled_h_t_c = torch.roll(h_t_c, shifts=1, dims=0) if h_t_c.shape[0] > 1 else h_t_c
    centered_correct = paired_cosine(h_l_c, h_t_c)
    centered_shuffled = paired_cosine(h_l_c, shuffled_h_t_c)
    return {
        "hL_batch_variance": float(h_l.float().var(dim=0).mean().item()) if h_l.shape[0] > 1 else 0.0,
        "hT_batch_variance": float(h_t.float().var(dim=0).mean().item()) if h_t.shape[0] > 1 else 0.0,
        "hL_mean_pairwise_cosine": mean_pairwise_cosine(h_l),
        "hT_mean_pairwise_cosine": mean_pairwise_cosine(h_t),
        "correct_pair_cosine": float(correct.mean().item()),
        "shuffled_pair_cosine": float(shuffled.mean().item()),
        "centered_correct_pair_cosine": float(centered_correct.mean().item()),
        "centered_shuffled_pair_cosine": float(centered_shuffled.mean().item()),
        "centered_margin": float((centered_correct - centered_shuffled).mean().item()),
    }


@torch.no_grad()
def loss2_intervention_diagnostics(
    model_b_dec,
    model_b_teacher,
    tokenizer,
    records: list[dict],
    z: torch.Tensor,
    max_cot_tokens: int,
    *,
    distance: Loss2Distance = "cosine",
    aggregate: Loss2Aggregate = "mean",
    layer_index: int = -1,
    teacher_context_mode: TeacherContextMode = "cumulative",
    feature_mode: Loss2FeatureMode = "pre_sem",
) -> Loss2DiagnosticOutput:
    rand = torch.randn_like(z)
    rand = rand / rand.float().norm(dim=-1, keepdim=True).clamp_min(1e-6) * z.float().norm(dim=-1, keepdim=True).to(rand.dtype)
    variants = {
        "normal": z,
        "shuffle": torch.roll(z, shifts=1, dims=0) if z.shape[0] > 1 else torch.zeros_like(z),
        "zero": torch.zeros_like(z),
        "random": rand,
    }
    metrics = {}
    normal_h_l = None
    teacher_h_t = None
    for name, value in variants.items():
        student, teacher, l2 = loss2_forward(
            model_b_dec,
            model_b_teacher,
            tokenizer,
            records,
            value,
            max_cot_tokens,
            distance=distance,
            aggregate=aggregate,
            layer_index=layer_index,
            teacher_context_mode=teacher_context_mode,
            feature_mode=feature_mode,
        )
        metrics[f"loss2_{name}"] = float(l2.loss2.item())
        metrics[f"h_l_variance_{name}"] = float(student.h_l.float().var(dim=0).mean().item()) if student.h_l.shape[0] > 1 else 0.0
        if name == "normal":
            normal_h_l = student.h_l.detach()
            teacher_h_t = teacher.h_t.detach()
    metrics["shuffle_margin"] = metrics["loss2_shuffle"] - metrics["loss2_normal"]
    metrics["zero_margin"] = metrics["loss2_zero"] - metrics["loss2_normal"]
    metrics["random_margin"] = metrics["loss2_random"] - metrics["loss2_normal"]
    metrics.update(feature_diagnostics(normal_h_l, teacher_h_t))
    return Loss2DiagnosticOutput(metrics=metrics, h_l=normal_h_l, h_t=teacher_h_t)


def exact_detach_grad_check(
    model_b_dec,
    model_b_teacher,
    tokenizer,
    records: list[dict],
    z: torch.Tensor,
    max_cot_tokens: int,
    *,
    distance: Loss2Distance = "cosine",
    aggregate: Loss2Aggregate = "mean",
    layer_index: int = -1,
    teacher_context_mode: TeacherContextMode = "cumulative",
    feature_mode: Loss2FeatureMode = "pre_sem",
) -> dict:
    if not z.requires_grad:
        raise ValueError("exact detach check requires original producer z with requires_grad")
    _student, _teacher, no_detach = loss2_forward(
        model_b_dec,
        model_b_teacher,
        tokenizer,
        records,
        z,
        max_cot_tokens,
        distance=distance,
        aggregate=aggregate,
        layer_index=layer_index,
        detach_latent=False,
        teacher_context_mode=teacher_context_mode,
        feature_mode=feature_mode,
    )
    grad_no_detach = torch.autograd.grad(no_detach.loss2, z, retain_graph=True)[0]
    _student_d, _teacher_d, detached = loss2_forward(
        model_b_dec,
        model_b_teacher,
        tokenizer,
        records,
        z,
        max_cot_tokens,
        distance=distance,
        aggregate=aggregate,
        layer_index=layer_index,
        detach_latent=True,
        teacher_context_mode=teacher_context_mode,
        feature_mode=feature_mode,
    )
    grad_detach = torch.autograd.grad(detached.loss2, z, allow_unused=True, retain_graph=True)[0]
    return {
        "grad_z_no_detach_norm": float(grad_no_detach.float().norm().item()),
        "grad_z_no_detach_finite": bool(torch.isfinite(grad_no_detach).all().item()),
        "grad_z_detach_is_none": grad_detach is None,
        "grad_z_detach_norm": 0.0 if grad_detach is None else float(grad_detach.float().norm().item()),
        "grad_z_detach_finite": True if grad_detach is None else bool(torch.isfinite(grad_detach).all().item()),
    }


@torch.no_grad()
def causal_leakage_check(
    model_b_dec,
    tokenizer,
    records_a: list[dict],
    records_b: list[dict],
    z: torch.Tensor,
    max_cot_tokens: int,
    *,
    layer_index: int = -1,
    feature_mode: Loss2FeatureMode = "pre_sem",
) -> dict:
    a = student_feature_forward(model_b_dec, tokenizer, records_a, z, max_cot_tokens, layer_index=layer_index, feature_mode=feature_mode)
    b = student_feature_forward(model_b_dec, tokenizer, records_b, z, max_cot_tokens, layer_index=layer_index, feature_mode=feature_mode)
    if not torch.equal(a.sem_positions, b.sem_positions):
        raise AssertionError("<SEM> positions changed in leakage check")
    sem_hidden_diff = (a.h_l.float() - b.h_l.float()).abs().max().item()
    batch = torch.arange(a.logits.shape[0], device=a.logits.device)
    sem_logits_diff = (a.logits[batch, a.sem_positions].float() - b.logits[batch, b.sem_positions].float()).abs().max().item()
    return {
        "sem_hidden_max_abs_diff": float(sem_hidden_diff),
        "sem_logits_max_abs_diff": float(sem_logits_diff),
    }


def find_same_question_pairs(records: list[dict]) -> list[tuple[int, int]]:
    by_question: dict[str, list[int]] = {}
    for i, record in enumerate(records):
        by_question.setdefault(record["question"], []).append(i)
    pairs = []
    for indices in by_question.values():
        if len(indices) < 2:
            continue
        for i in range(len(indices) - 1):
            pairs.append((indices[i], indices[i + 1]))
    return pairs
