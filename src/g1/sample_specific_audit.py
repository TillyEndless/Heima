from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .evaluator import generate_cot_sample
from .latent_reasoner import encode_question_batch, extract_latent
from .latent_retrieval import effective_rank, farthest_indices, pairwise_cosine, recall_at_k
from .synthetic_data import read_jsonl
from .token_category_metrics import (
    cyclic_derangement,
    decoded_token_categories,
    extract_expressions,
    extract_numbers,
    kl_from_logits,
    masked_mean,
    template_statistics,
    token_nll,
)
from .trainer import load_config
from .whole_cot_decoder import ensure_latent_token, loss1_forward


def load_trained(config_path: str):
    config = load_config(config_path)
    kwargs = {
        "local_files_only": bool(config.get("local_files_only", False)),
        "use_safetensors": bool(config.get("use_safetensors", True)),
    }
    path = config.get("model_name_or_path", config.get("model_name"))
    tokenizer = AutoTokenizer.from_pretrained(path, **kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    model_a = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model_b = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    ensure_latent_token(tokenizer, model_a, model_b)
    ckpt = Path(config["checkpoint_dir"])
    model_a.load_state_dict(torch.load(ckpt / "model_a.pt", map_location="cpu")["model_a"])
    model_b.load_state_dict(torch.load(ckpt / "model_b.pt", map_location="cpu")["model_b"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a.to(device).eval()
    model_b.to(device).eval()
    model_a.config.use_cache = False
    model_b.config.use_cache = False
    return config, tokenizer, model_a, model_b, device


def build_latent_bank(model_a, tokenizer, records, config, path: Path) -> dict:
    device = next(model_a.parameters()).device
    with torch.no_grad():
        q_ids, q_mask, _ = encode_question_batch(
            tokenizer, records, config["max_question_tokens"], device
        )
        z = extract_latent(model_a, q_ids, q_mask)
    bank = {
        "sample_id": [r["id"] for r in records],
        "question": [r["question"] for r in records],
        "gold_cot": [r["cot"] for r in records],
        "gold_answer": [r["answer"] for r in records],
        "latent": z.detach().cpu(),
        "latent_norm": z.float().norm(dim=1).detach().cpu(),
        "latent_mean": z.float().mean(dim=1).detach().cpu(),
        "latent_std": z.float().std(dim=1).detach().cpu(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, path)
    return bank


def make_conditions(z: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict]:
    n = z.size(0)
    perm = cyclic_derangement(n)
    far = farthest_indices(z)
    generator = torch.Generator(device=z.device)
    generator.manual_seed(915)
    rnd = torch.randn(z.shape, dtype=z.dtype, device=z.device, generator=generator)
    rnd = rnd / rnd.norm(dim=1, keepdim=True).clamp_min(1e-12) * z.norm(dim=1, keepdim=True)
    mean = z.mean(dim=0, keepdim=True).expand_as(z)
    conditions = {
        "normal": z,
        "cyclic_shuffle": z[perm],
        "farthest": z[far],
        "zero": torch.zeros_like(z),
        "norm_matched_random": rnd,
        "global_mean": mean,
    }
    metadata = {
        "cyclic_permutation": perm,
        "farthest_permutation": far,
        "cyclic_has_fixed_point": any(i == j for i, j in enumerate(perm)),
        "farthest_has_fixed_point": any(i == j for i, j in enumerate(far)),
    }
    return conditions, metadata


def cot_token_ids(tokenizer, records, max_cot_tokens: int) -> list[list[int]]:
    return [
        tokenizer(r["cot"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"][
            :max_cot_tokens
        ]
        for r in records
    ]


def condition_metrics(model_b, tokenizer, records, z_normal, z_cond, config, normal_logits=None):
    out = loss1_forward(model_b, tokenizer, records, z_normal, config["max_cot_tokens"], z_cond)
    nll, mask = token_nll(out.logits, out.labels)
    ids = cot_token_ids(tokenizer, records, config["max_cot_tokens"])
    cat_masks = {k: torch.zeros_like(mask) for k in ["numeric", "operator", "sample_specific"]}
    for i, row in enumerate(ids):
        cats = decoded_token_categories(tokenizer, row)
        target_positions = torch.where(mask[i])[0].tolist()
        for local_idx, shifted_pos in enumerate(target_positions[: len(row)]):
            for key in cat_masks:
                if cats[key][local_idx]:
                    cat_masks[key][i, shifted_pos] = True
    first_mask = torch.zeros_like(mask)
    first4_mask = torch.zeros_like(mask)
    first8_mask = torch.zeros_like(mask)
    for i in range(mask.size(0)):
        pos = torch.where(mask[i])[0]
        if pos.numel():
            first_mask[i, pos[:1]] = True
            first4_mask[i, pos[:4]] = True
            first8_mask[i, pos[:8]] = True
    metrics = {
        "full_cot_nll": masked_mean(nll, mask),
        "first_token_nll": masked_mean(nll, first_mask),
        "first_4_tokens_nll": masked_mean(nll, first4_mask),
        "first_8_tokens_nll": masked_mean(nll, first8_mask),
        "numeric_token_nll": masked_mean(nll, cat_masks["numeric"]),
        "operator_token_nll": masked_mean(nll, cat_masks["operator"]),
        "sample_specific_token_nll": masked_mean(nll, cat_masks["sample_specific"]),
    }
    if normal_logits is not None:
        kl = kl_from_logits(normal_logits[:, :-1, :], out.logits[:, :-1, :])
        metrics["first_token_logits_KL"] = masked_mean(kl, first_mask)
        metrics["average_token_logits_KL"] = masked_mean(kl, mask)
    else:
        metrics["first_token_logits_KL"] = 0.0
        metrics["average_token_logits_KL"] = 0.0
    return metrics, out.logits


def free_generation_metrics(model_b, tokenizer, records, z_conditions, max_samples=32):
    names = ["normal", "cyclic_shuffle", "farthest", "zero"]
    results = {}
    sample_lines = []
    gold_numbers = [set(extract_numbers(r["cot"])) for r in records[:max_samples]]
    gold_exprs = [set(extract_expressions(r["cot"])) for r in records[:max_samples]]
    for name in names:
        num_scores = []
        expr_scores = []
        first_mid_hits = []
        final_mid_hits = []
        op_hits = []
        for i, record in enumerate(records[:max_samples]):
            text = generate_cot_sample(
                model_b,
                tokenizer,
                z_conditions["normal"][i : i + 1],
                z_conditions[name][i : i + 1],
                max_new_tokens=64,
            ) or ""
            nums = set(extract_numbers(text))
            exprs = set(extract_expressions(text))
            gold_nums = gold_numbers[i]
            gold_ex = gold_exprs[i]
            num_scores.append(len(nums & gold_nums) / max(len(gold_nums), 1))
            expr_scores.append(len(exprs & gold_ex) / max(len(gold_ex), 1))
            gold_list = extract_numbers(record["cot"])
            gen_list = extract_numbers(text)
            first_mid_hits.append(float(bool(gold_list and gen_list and gold_list[0] == gen_list[0])))
            final_mid_hits.append(float(bool(gold_list and gen_list and gold_list[-1] in gen_list)))
            op_hits.append(float(any(op in text.lower() for op in ["add", "multiply", "+", "*"])))
            if i < 8:
                sample_lines.extend(
                    [
                        f"{name} sample {i+1}",
                        f"question: {record['question']}",
                        f"gold: {record['cot']}",
                        f"generated: {text}",
                        "",
                    ]
                )
        results[name] = {
            "number_match_rate": sum(num_scores) / len(num_scores),
            "expression_match_rate": sum(expr_scores) / len(expr_scores),
            "first_intermediate_number_match": sum(first_mid_hits) / len(first_mid_hits),
            "final_intermediate_result_match": sum(final_mid_hits) / len(final_mid_hits),
            "operation_type_match": sum(op_hits) / len(op_hits),
        }
    return results, "\n".join(sample_lines)


def cot_representations(model_b, tokenizer, records, config):
    device = next(model_b.parameters()).device
    texts = ["Reasoning:\n" + r["cot"] for r in records]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=config["max_cot_tokens"] + 16,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)
    with torch.no_grad():
        out = model_b(**enc, output_hidden_states=True, use_cache=False)
    pos = enc["attention_mask"].sum(dim=1) - 1
    batch = torch.arange(len(records), device=device)
    return out.hidden_states[-1][batch, pos, :]


def generate_audit_set(seed=4242) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for i in range(64):
        kind = i % 4
        if kind == 0:
            a, b, c = rng.randint(3, 40), rng.randint(3, 40), rng.randint(2, 12)
            y = (a + b) * c
            steps = [f"Add {a} and {b} to get {a+b}.", f"Multiply {a+b} by {c} to get {y}."]
            q = f"Compute ({a} + {b}) * {c}."
        elif kind == 1:
            a, b, c = rng.randint(3, 20), rng.randint(3, 20), rng.randint(2, 80)
            y = a * b - c
            steps = [f"Multiply {a} by {b} to get {a*b}.", f"Subtract {c} to get {y}."]
            q = f"Compute {a} * {b} - {c}."
        elif kind == 2:
            b, c, y = rng.randint(2, 20), rng.randint(2, 8), rng.randint(2, 20)
            a = b + c * y
            steps = [f"Subtract {b} from {a} to get {a-b}.", f"Divide {a-b} by {c} to get {y}."]
            q = f"Compute ({a} - {b}) / {c}."
        else:
            a, b, c, d = rng.randint(2, 30), rng.randint(2, 15), rng.randint(2, 12), rng.randint(2, 50)
            y = a + b * c - d
            steps = [
                f"Multiply {b} by {c} to get {b*c}.",
                f"Add {a} to get {a + b*c}.",
                f"Subtract {d} to get {y}.",
            ]
            q = f"Compute {a} + {b} * {c} - {d}."
        records.append(
            {
                "id": f"g15_audit_{i:04d}",
                "question": q,
                "cot": " ".join(steps),
                "answer": str(y),
                "steps": steps,
                "metadata": {"kind": kind},
            }
        )
    return records


def run_g15_audit() -> dict:
    config, tokenizer, model_a, model_b, device = load_trained(
        "experiments/g1_gpt2/configs/main_l1.yaml"
    )
    records = read_jsonl(config["validation_path"])
    bank = build_latent_bank(
        model_a,
        tokenizer,
        records,
        config,
        Path("experiments/g1_gpt2/reports/g15_latent_bank.pt"),
    )
    z = bank["latent"].to(device)
    conditions, permutation = make_conditions(z)
    Path("experiments/g1_gpt2/reports/g15_permutation.json").write_text(
        json.dumps(permutation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    normal_metrics, normal_logits = condition_metrics(
        model_b, tokenizer, records, z, conditions["normal"], config
    )
    condition_report = {"normal": normal_metrics}
    for name, value in conditions.items():
        if name == "normal":
            continue
        condition_report[name], _ = condition_metrics(
            model_b, tokenizer, records, z, value, config, normal_logits
        )
    free_gen, samples = free_generation_metrics(model_b, tokenizer, records, conditions)
    Path("experiments/g1_gpt2/reports/g15_free_generation_samples.txt").write_text(
        samples, encoding="utf-8"
    )
    tmpl = template_statistics(records, tokenizer)
    Path("experiments/g1_gpt2/reports/g15_template_statistics.json").write_text(
        json.dumps(tmpl, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    cos = pairwise_cosine(z).detach().cpu()
    cot_repr = cot_representations(model_b, tokenizer, records, config)
    retrieval_scores = torch.nn.functional.normalize(z.float(), dim=-1) @ torch.nn.functional.normalize(
        cot_repr.float(), dim=-1
    ).T
    audit_records = generate_audit_set()
    with torch.no_grad():
        q_ids, q_mask, _ = encode_question_batch(
            tokenizer, audit_records, config["max_question_tokens"], device
        )
        audit_z = extract_latent(model_a, q_ids, q_mask)
    audit_conditions, _ = make_conditions(audit_z)
    audit_normal, audit_logits = condition_metrics(
        model_b, tokenizer, audit_records, audit_z, audit_conditions["normal"], config
    )
    audit_shuffle, _ = condition_metrics(
        model_b,
        tokenizer,
        audit_records,
        audit_z,
        audit_conditions["cyclic_shuffle"],
        config,
        audit_logits,
    )
    pass_a = (
        condition_report["normal"]["first_8_tokens_nll"]
        < condition_report["cyclic_shuffle"]["first_8_tokens_nll"]
        and condition_report["normal"]["numeric_token_nll"]
        < condition_report["cyclic_shuffle"]["numeric_token_nll"]
    )
    pass_b = (
        free_gen["normal"]["number_match_rate"]
        > free_gen["cyclic_shuffle"]["number_match_rate"]
        and free_gen["normal"]["expression_match_rate"]
        >= free_gen["cyclic_shuffle"]["expression_match_rate"]
    )
    pass_c = recall_at_k(retrieval_scores, 1) > (1 / len(records))
    stop = []
    if not (pass_a or pass_b or pass_c):
        stop.append("STOP-G1.5")
    report = {
        "status": "pass" if not stop else "stop",
        "stop_gates_triggered": stop,
        "criteria": {"A": pass_a, "B": pass_b, "C": pass_c},
        "permutation": permutation,
        "condition_metrics": condition_report,
        "free_generation": free_gen,
        "latent_geometry": {
            "pairwise_cosine_mean": cos[~torch.eye(cos.size(0), dtype=torch.bool)].mean().item(),
            "pairwise_cosine_min": cos[~torch.eye(cos.size(0), dtype=torch.bool)].min().item(),
            "pairwise_cosine_max": cos[~torch.eye(cos.size(0), dtype=torch.bool)].max().item(),
            "within_dataset_variance": z.float().var(dim=0).mean().item(),
            "effective_rank": effective_rank(z),
        },
        "retrieval": {
            "recall_at_1": recall_at_k(retrieval_scores, 1),
            "recall_at_5": recall_at_k(retrieval_scores, 5),
            "random_recall_at_1": 1 / len(records),
            "random_recall_at_5": 5 / len(records),
        },
        "template_statistics": tmpl,
        "audit_set": {
            "num_records": len(audit_records),
            "normal": audit_normal,
            "cyclic_shuffle": audit_shuffle,
        },
    }
    out = Path("experiments/g1_gpt2/reports/g15_sample_specific_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
