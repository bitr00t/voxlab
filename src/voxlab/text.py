"""Text handling between the language model and the synthesiser."""

from __future__ import annotations

import re

# Sentence-final punctuation, plus the abbreviations that would otherwise split
# a German sentence in the wrong place.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_ABBREVIATIONS = (
    "z.B.", "u.a.", "d.h.", "bzw.", "ca.", "ggf.", "inkl.", "Nr.", "Dr.", "Hr.", "Fr.",
    "St.", "evtl.", "usw.", "etc.",
)
_PLACEHOLDER = "\x00"


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for chunk-wise synthesis.

    The synthesiser is fed one sentence at a time so the first audio can leave
    before the whole reply has been rendered. That makes the boundaries audible,
    which is why abbreviations are protected: a break after "z." would be heard
    as a pause in the middle of a word.
    """
    if not text.strip():
        return []

    protected = text
    for abbreviation in _ABBREVIATIONS:
        protected = protected.replace(abbreviation, abbreviation.replace(".", _PLACEHOLDER))

    parts = [part.strip() for part in _SENTENCE_END.split(protected)]
    return [part.replace(_PLACEHOLDER, ".") for part in parts if part.strip()]
