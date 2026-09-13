"""Qwen3-TTS on the local GPU.

DELIBERATELY UNIMPLEMENTED IN PHASE 0.

The interface below is the contract the rest of the pipeline relies on; the
model call is left open on purpose, because the Qwen3-TTS Python API and its
distribution name should be read off the current upstream release rather than
guessed at. Filling in `_synthesize_blocking` is the first task of Phase 1.

What is already decided, and what the implementation has to honour:

* Output is mono 16-bit PCM at `sample_rate`, matching the project convention in
  voxlab.types. Whatever the model emits natively gets converted here, not by
  the caller.
* Synthesis is chunked. Qwen3-TTS supports streaming output, which is the reason
  it was picked over Piper for this project: the first audio chunk has to leave
  before the full sentence is synthesised, or the latency budget is gone.
* The model call is blocking, so it runs in a worker thread like the Whisper one.
* Roughly 4 GB of VRAM is the documented minimum. Budget it alongside Whisper
  large-v3 at float16 and Qwen3 on Ollama before assuming all three fit.

A caveat worth keeping in the repository rather than in someone's head: local
German speech synthesis at conversational latency is still the weakest link in
a self-hosted stack. The evaluation harness in Phase 4 exists to measure the
real-time factor on this specific hardware instead of trusting a benchmark run
on someone else's machine. Piper with a Thorsten voice remains the fallback if
the numbers disappoint - which is precisely why this provider sits behind an
interface.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from voxlab.providers.base import TextToSpeech
from voxlab.providers.registry import register_tts
from voxlab.types import AudioChunk


class Qwen3TtsNotImplementedError(NotImplementedError):
    """Raised until the Phase 1 implementation lands."""


@register_tts("qwen3")
class Qwen3Tts(TextToSpeech):
    """Local neural speech synthesis with Qwen3-TTS."""

    name = "qwen3"

    def __init__(
        self,
        model: str = "qwen3-tts",
        voice: str = "de-default",
        device: str = "cuda",
        sample_rate: int = 24000,
        **_: object,
    ) -> None:
        self._model_name = model
        self._voice = voice
        self._device = device
        self._sample_rate = sample_rate

    async def start(self) -> None:
        raise Qwen3TtsNotImplementedError(
            "Qwen3-TTS is scheduled for Phase 1. Use VOXLAB__TTS__PROVIDER=silent "
            "until then, and check the upstream release for the current package "
            "name and API before implementing this."
        )

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        raise Qwen3TtsNotImplementedError("see start()")
        yield  # pragma: no cover  (makes this an async generator)
