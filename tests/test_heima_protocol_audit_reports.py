import json
from pathlib import Path


def test_prompt_alignment_schema():
    data = json.loads(Path('reports/prompt_alignment.json').read_text())
    assert len(data['rows']) == 3
    assert all(row['aligned'] for row in data['rows'])
    assert all('<THINKING_OF_' in row['official_style']['special_token'] for row in data['rows'])


def test_intervention_outputs_exist_and_have_rows():
    base = Path('reports/heima_official_intervention')
    for name in ['normal.jsonl', 'remove_latent.jsonl', 'shuffle.jsonl', 'zero.jsonl']:
        rows = [json.loads(line) for line in (base / name).read_text().splitlines() if line.strip()]
        assert len(rows) == 24
        assert {'question', 'gold_text', 'generated_text', 'nll', 'section'} <= set(rows[0])


def test_special_token_norms_are_recorded():
    data = json.loads(Path('reports/heima_official_intervention/special_token_embedding_norms.json').read_text())
    assert sorted(data) == ['caption', 'reasoning', 'summary']
    assert all(item['before_replacement_embedding_norm'] > 0 for item in data.values())
    assert all(item['remove_zero_after_replacement_norm'] == 0.0 for item in data.values())
