from __future__ import annotations

from src.htext.modeling import DECODER_TEMPLATE, THINKING_TOKEN


class TinyTokenizer:
    def __init__(self):
        self.vocab = {THINKING_TOKEN: 99}

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [self.vocab.get(part, i) for i, part in enumerate(text.split())]}

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]


def test_decoder_template_contains_question_and_one_thinking_token():
    prompt = DECODER_TEMPLATE.format(question="What is 2 + 2?")
    assert "Question:\nWhat is 2 + 2?" in prompt
    assert prompt.count(THINKING_TOKEN) == 1
