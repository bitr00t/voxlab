"""Data types exchanged between the provider layers.

Everything that crosses a provider boundary is defined here, so that swapping a
provider can never change the shape of the data flowing through the pipeline.

Audio convention for the whole project: 16-bit signed little-endian PCM, mono.
Sample rate travels with the chunk instead of being a global constant, because
speech-to-text and text-to-speech models rarely agree on one (Whisper wants
16 kHz, most neural vocoders emit 22.05 or 24 kHz). Resampling is the
responsibility of whoever consumes a chunk, not of whoever produces it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """A block of mono 16-bit PCM audio."""

    pcm: bytes
    sample_rate: int

    @property
    def duration_s(self) -> float:
        """Duration in seconds, derived from the byte count (2 bytes per frame)."""
        return len(self.pcm) / 2 / self.sample_rate


@dataclass(frozen=True, slots=True)
class Transcript:
    """The result of a speech-to-text pass.

    `is_final` distinguishes an interim hypothesis from a settled one. Phase 1
    only ever produces final transcripts; streaming partials arrive in Phase 2,
    where they are needed for barge-in handling.
    """

    text: str
    is_final: bool = True
    language: str | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in the conversation handed to the language model."""

    role: Role
    content: str


@dataclass(slots=True)
class TurnMetrics:
    """Latency measurements for a single user turn, in seconds.

    Collected from Phase 0 onwards even though the numbers are meaningless with
    the placeholder providers. The point is that every later phase inherits a
    pipeline that already measures itself, rather than one that gets
    instrumented as an afterthought once the latency is already disappointing.
    """

    stt_s: float | None = None
    llm_first_token_s: float | None = None
    llm_total_s: float | None = None
    tts_first_chunk_s: float | None = None
    total_s: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def time_to_first_audio_s(self) -> float | None:
        """The number that decides whether a caller stays on the line.

        Under roughly 800 ms a conversation feels responsive; past 1.5 s callers
        start talking over the agent or hang up.
        """
        parts = [self.stt_s, self.llm_first_token_s, self.tts_first_chunk_s]
        if any(p is None for p in parts):
            return None
        return sum(p for p in parts if p is not None)
