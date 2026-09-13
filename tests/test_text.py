"""Sentence splitting decides where the synthesiser draws breath."""

from __future__ import annotations

import pytest

from voxlab.text import split_sentences


def test_splits_on_sentence_final_punctuation() -> None:
    assert split_sentences("Guten Tag. Wie kann ich helfen?") == [
        "Guten Tag.",
        "Wie kann ich helfen?",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Wir liefern z.B. nach Hessen.",
        "Das Gerät wiegt ca. 12 Kilogramm.",
        "Bitte fragen Sie Dr. Meier.",
    ],
)
def test_abbreviations_do_not_break_a_sentence(text: str) -> None:
    # A break after "z." would be heard as a pause inside a word.
    assert split_sentences(text) == [text]


def test_empty_input_yields_nothing() -> None:
    assert split_sentences("   ") == []


def test_text_without_punctuation_stays_one_chunk() -> None:
    assert split_sentences("kurz und ohne punkt") == ["kurz und ohne punkt"]
