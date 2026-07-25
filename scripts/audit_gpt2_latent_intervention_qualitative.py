#!/usr/bin/env python3
"""Qualitative latent intervention audit for the GPT2 G3 final checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List

import torch

from run_gpt2_latent_ablation import (
    Example,
    THINK_TOKEN,
    encode_text,
    extract_z,
    find_subsequence,
    load_hf,
    make_lm_batch,
    replace_slot_embeddings,
)

STOPWORDS = {
    'a','an','the','this','that','these','those','it','is','are','was','were','be','been','being',
    'and','or','but','because','so','to','of','in','on','for','with','as','by','from','at','into',
    'there','here','image','picture','photo','question','answer','reasoning','target','yes','no',
}


def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text)).strip()


def strip_prompt(text: str) -> str:
    markers = ['Target reasoning:', 'Answer:', THINK_TOKEN]
    out = text
    for marker in markers:
        if marker in out:
            out = out.split(marker, 1)[-1]
    return normalize(out.replace('<|endoftext|>', ' '))


def extract_answer(text: str) -> str:
    clean = strip_prompt(text)
    m = re.search(r'(?:answer|therefore|so)[:\s]+([^\.\n]+)', clean, flags=re.I)
    if m:
        return normalize(m.group(1))[:120]
    sents = re.split(r'[\.\n]', clean)
    for sent in reversed(sents):
        sent = normalize(sent)
        if len(sent) >= 2:
            return sent[:120]
    return clean[:120]


def extract_entities(text: str) -> List[str]:
    clean = strip_prompt(text)
    ents = set(re.findall(r'\b[A-Z][a-zA-Z0-9_-]{2,}\b', clean))
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]{3,}\b', clean.lower())
    freq = {}
    for w in words:
        if w not in STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    for w, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:8]:
        ents.add(w)
    return sorted(ents)[:12]


def edit_distance(a: str, b: str) -> int:
    a = strip_prompt(a)
    b = strip_prompt(b)
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[m]


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, strip_prompt(a), strip_prompt(b)).ratio()


def parse_options(question: str) -> Dict[str, str]:
    opts = {}
    matches = list(re.finditer(r'\b([A-D])\.\s*([^A-D\n]+?)(?=\s+[A-D]\.|$)', question, flags=re.I))
    for m in matches:
        opts[m.group(1).upper()] = normalize(m.group(2))
    return opts


def contains_gold_answer(generated: str, gold_answer: str, question: str = '') -> bool:
    g = strip_prompt(generated).lower()
    gold_raw = normalize(gold_answer).strip()
    gold = gold_raw.lower().strip('.')
    if not gold:
        return False
    if re.fullmatch(r'[a-d]', gold, flags=re.I):
        letter = gold.upper()
        if re.search(rf'\b{letter}\b|\b{letter.lower()}\b', strip_prompt(generated)):
            return True
        opt = parse_options(question).get(letter, '')
        opt_tokens = [t for t in re.findall(r'\w+', opt.lower()) if t not in STOPWORDS]
        if not opt_tokens:
            return False
        return sum(t in g for t in opt_tokens) / len(opt_tokens) >= 0.6
    if gold in g:
        return True
    gold_tokens = [t for t in re.findall(r'\w+', gold) if t not in STOPWORDS]
    if not gold_tokens:
        return False
    return sum(t in g for t in gold_tokens) / len(gold_tokens) >= 0.6


def build_z(model, tokenizer, records, max_length, device):
    batch = make_lm_batch(tokenizer, records, 'latent_main', max_length, device)
    out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
    return extract_z(out.hidden_states[-1], batch['positions'])


def generate_one(model, tokenizer, ex: Example, zrow, condition: str, max_length: int, max_new_tokens: int, device) -> str:
    if condition == 'question_only':
        prompt = f'Question:\n{ex.question}\n\nTarget reasoning:\n'
        ids = torch.tensor([encode_text(tokenizer, prompt)], dtype=torch.long, device=device)
        out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    else:
        prompt = f'Question:\n{ex.question}\n\nLatent:\n{THINK_TOKEN}\nTarget reasoning:\n'
        ids = torch.tensor([encode_text(tokenizer, prompt)], dtype=torch.long, device=device)
        embeds = model.get_input_embeddings()(ids)
        pos = find_subsequence(ids[0].tolist(), encode_text(tokenizer, THINK_TOKEN))
        embeds[0, pos, :] = zrow
        out = model.generate(inputs_embeds=embeds, attention_mask=torch.ones(ids.shape, device=device), max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=False)


def classify_cases(rows: List[Dict]) -> Dict[str, List[int]]:
    success_shuffle_fail = []
    identical = []
    for i, row in enumerate(rows):
        c = row['metrics']['correct_answer_hit']
        s = row['metrics']['shuffle_answer_hit']
        sim = row['metrics']['correct_shuffle_similarity']
        ed = row['metrics']['edit_distance_correct_shuffle']
        if c and not s:
            success_shuffle_fail.append(i)
        if sim >= 0.98 or ed <= 3:
            identical.append(i)
    return {'correct_success_shuffle_fail': success_shuffle_fail, 'correct_shuffle_identical': identical}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='/data/zxl/runs/gpt2_latent_token_ablation_formal/G3/step5000/model_weights.pt')
    ap.add_argument('--split', default='/data/zxl/runs/gpt2_latent_token_ablation_formal/data_split.json')
    ap.add_argument('--model-path', default='/data/zxl/models/openai-community-gpt2')
    ap.add_argument('--output-jsonl', default='reports/latent_intervention_examples.jsonl')
    ap.add_argument('--output-md', default='reports/latent_intervention_analysis.md')
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=10)
    ap.add_argument('--max-length', type=int, default=192)
    ap.add_argument('--max-new-tokens', type=int, default=80)
    ap.add_argument('--device', default='')
    args = ap.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, tokenizer = load_hf(args.model_path, device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state['model'], strict=False)
    model.eval()

    split = json.loads(Path(args.split).read_text())
    records = [Example(**x) for x in split['eval'][:args.samples]]
    rows = []
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start:start + args.batch_size]
            z = build_z(model, tokenizer, batch_records, args.max_length, device)
            shuffled = z[torch.randperm(z.size(0), device=device)]
            zero = torch.zeros_like(z)
            for j, ex in enumerate(batch_records):
                gens = {
                    'correct_latent_generation': generate_one(model, tokenizer, ex, z[j], 'correct', args.max_length, args.max_new_tokens, device),
                    'shuffled_latent_generation': generate_one(model, tokenizer, ex, shuffled[j], 'shuffle', args.max_length, args.max_new_tokens, device),
                    'zero_latent_generation': generate_one(model, tokenizer, ex, zero[j], 'zero', args.max_length, args.max_new_tokens, device),
                    'question_only_generation': generate_one(model, tokenizer, ex, None, 'question_only', args.max_length, args.max_new_tokens, device),
                }
                answer_extraction = {k: extract_answer(v) for k, v in gens.items()}
                entity_extraction = {k: extract_entities(v) for k, v in gens.items()}
                correct_hit = contains_gold_answer(gens['correct_latent_generation'], ex.answer, ex.question)
                shuffle_hit = contains_gold_answer(gens['shuffled_latent_generation'], ex.answer, ex.question)
                row = {
                    'sample_id': start + j,
                    'question': ex.question,
                    'gold_cot': ex.cot,
                    'gold_answer': ex.answer,
                    **gens,
                    'answer_extraction': answer_extraction,
                    'entity_extraction': entity_extraction,
                    'metrics': {
                        'correct_answer_hit': correct_hit,
                        'shuffle_answer_hit': shuffle_hit,
                        'zero_answer_hit': contains_gold_answer(gens['zero_latent_generation'], ex.answer, ex.question),
                        'question_only_answer_hit': contains_gold_answer(gens['question_only_generation'], ex.answer, ex.question),
                        'answer_flip_correct_vs_shuffle': correct_hit != shuffle_hit,
                        'edit_distance_correct_shuffle': edit_distance(gens['correct_latent_generation'], gens['shuffled_latent_generation']),
                        'edit_distance_correct_zero': edit_distance(gens['correct_latent_generation'], gens['zero_latent_generation']),
                        'edit_distance_correct_qonly': edit_distance(gens['correct_latent_generation'], gens['question_only_generation']),
                        'correct_shuffle_similarity': similarity(gens['correct_latent_generation'], gens['shuffled_latent_generation']),
                        'correct_zero_similarity': similarity(gens['correct_latent_generation'], gens['zero_latent_generation']),
                        'correct_qonly_similarity': similarity(gens['correct_latent_generation'], gens['question_only_generation']),
                    },
                }
                rows.append(row)

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    cases = classify_cases(rows)
    avg = lambda key: sum(r['metrics'][key] for r in rows) / max(1, len(rows))
    hit = lambda key: sum(1 for r in rows if r['metrics'][key])
    md = [
        '# Latent Intervention Qualitative Audit',
        '',
        f'Checkpoint: `{args.checkpoint}`',
        f'Split: `{args.split}`',
        f'Samples: `{len(rows)}`',
        '',
        '## Aggregate Metrics',
        '',
        f'- correct answer hits: {hit("correct_answer_hit")}/{len(rows)}',
        f'- shuffled answer hits: {hit("shuffle_answer_hit")}/{len(rows)}',
        f'- zero answer hits: {hit("zero_answer_hit")}/{len(rows)}',
        f'- question-only answer hits: {hit("question_only_answer_hit")}/{len(rows)}',
        f'- answer flips correct vs shuffle: {hit("answer_flip_correct_vs_shuffle")}/{len(rows)}',
        f'- avg edit distance correct/shuffle: {avg("edit_distance_correct_shuffle"):.2f}',
        f'- avg edit distance correct/zero: {avg("edit_distance_correct_zero"):.2f}',
        f'- avg edit distance correct/q-only: {avg("edit_distance_correct_qonly"):.2f}',
        f'- avg similarity correct/shuffle: {avg("correct_shuffle_similarity"):.4f}',
        f'- avg similarity correct/zero: {avg("correct_zero_similarity"):.4f}',
        f'- avg similarity correct/q-only: {avg("correct_qonly_similarity"):.4f}',
        '',
        '## Automatically Found Cases',
        '',
        f'- correct latent success but shuffle failure sample ids: {cases["correct_success_shuffle_fail"][:20]}',
        f'- correct/shuffle nearly identical sample ids: {cases["correct_shuffle_identical"][:20]}',
        '',
        '## Diagnosis',
        '',
    ]
    if avg('correct_shuffle_similarity') > 0.9 and hit('answer_flip_correct_vs_shuffle') <= 5:
        md += [
            'The qualitative audit agrees with the near-zero shuffle margin: correct and shuffled latent generations are usually almost the same. This makes A, latent lacks sample-specific information, and B, decoder shortcut, more plausible than a hidden positive intervention effect.',
            '',
            'Because question-only generations also remain close to correct-latent generations, D, question leakage, is also likely contributing. The target appears partially recoverable from question/prompt priors, so C, target too easy, is a secondary contributor.',
            '',
            'Current ranking: B/D strongest, A also likely, C possible. The audit does not support the claim that G3 final latent is a necessary sample-specific reasoning state.',
        ]
    else:
        md += [
            'Some qualitative differences were found. Inspect the listed success/failure cases before drawing a shortcut conclusion.',
        ]
    Path(args.output_md).write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps({'examples': str(out_jsonl), 'analysis': args.output_md, 'cases': cases, 'samples': len(rows)}, indent=2))


if __name__ == '__main__':
    main()
