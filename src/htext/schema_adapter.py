from __future__ import annotations

from dataclasses import dataclass


OFFICIAL_REASONING_TOKEN = "<THINKING_OF_REASONING>"
LEGACY_THINKING_TOKEN = "<THINKING>"


@dataclass(frozen=True)
class TextOnlyHeimaRecord:
    question: str
    reasoning: str
    answer: str
    sample_id: str = "fixture_0"


def encoder_text(record: TextOnlyHeimaRecord, thinking_token: str = OFFICIAL_REASONING_TOKEN) -> str:
    return f"{record.question} {thinking_token}\nAnswer: {record.answer}"


def decoder_prompt(record: TextOnlyHeimaRecord, latent_token: str = OFFICIAL_REASONING_TOKEN) -> str:
    return (
        f"Question:\n{record.question}\n\n"
        "Instruction:\nExplain the reasoning information encoded in the latent state.\n\n"
        "Latent:\n"
        f"{latent_token}\n\n"
        "Reasoning:\n"
    )


def decoder_text(record: TextOnlyHeimaRecord, latent_token: str = OFFICIAL_REASONING_TOKEN) -> str:
    return decoder_prompt(record, latent_token) + record.reasoning


def minimal_fixture() -> TextOnlyHeimaRecord:
    return TextOnlyHeimaRecord(
        question="Compute (2 + 3) * 4.",
        reasoning="First add 2 and 3: 2 + 3 = 5. Then multiply 5 by 4: 5 * 4 = 20.",
        answer="20",
    )
