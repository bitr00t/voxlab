"""A speech synthesiser that emits silence.

Produces correctly shaped audio chunks of a plausible duration, so the transport
and the metrics can be exercised without a TTS model installed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from voxlab.providers.base import TextToSpeech
from voxlab.providers.registry import register_tts
from voxlab.text import split_sentences
from voxlab.types import AudioChunk

# Rough German speaking rate, used to give the silence a realistic length.
_CHARS_PER_SECOND = 15.0


@register_tts("silent")
class SilentTts(TextToSpeech):
    """Yields one chunk of silence per sentence."""

    name = "silent"

    def __init__(self, sample_rate: int = 24000, **_: object) -> None:
        self._sample_rate = sample_rate

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        for sentence in split_sentences(text) or [""]:
            frames = max(1, int(len(sentence) / _CHARS_PER_SECOND * self._sample_rate))
            yield AudioChunk(pcm=b"\x00\x00" * frames, sample_rate=self._sample_rate)
