"""Abstract provider interfaces.

Three seams, one rule each:

* SpeechToText  - audio in, text out
* LanguageModel - messages in, a stream of text fragments out
* TextToSpeech  - text in, a stream of audio chunks out

The language model and the speech synthesiser both stream, because the time to
first audio is what a caller perceives as responsiveness. Waiting for a complete
sentence from the model before starting synthesis wastes hundreds of
milliseconds that cannot be recovered later.

Providers are asynchronous even where the underlying library is synchronous.
Blocking calls belong in a worker thread inside the provider, so that a slow
model can never stall the audio transport.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Sequence

from voxlab.types import AudioChunk, Message, Transcript


class Provider(abc.ABC):
    """Shared lifecycle for everything that owns a model or a socket."""

    name: str = "unnamed"

    async def start(self) -> None:
        """Load models, open connections. Safe to call once."""

    async def aclose(self) -> None:
        """Release GPU memory and connections."""

    async def __aenter__(self) -> Provider:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


class SpeechToText(Provider):
    """Turns recorded audio into text."""

    @abc.abstractmethod
    async def transcribe(self, audio: AudioChunk) -> Transcript:
        """Transcribe one complete utterance.

        Phase 1 works on whole utterances delimited by voice activity detection.
        Streaming partial results get their own method in Phase 2 rather than
        overloading this one, so the simple path stays simple.
        """


class LanguageModel(Provider):
    """Produces the agent's reply, one fragment at a time."""

    @abc.abstractmethod
    def stream(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        """Yield text fragments as they are generated.

        Implementations are async generators. The method itself is deliberately
        not declared `async`, so that callers write `async for chunk in
        llm.stream(...)` without an extra await.
        """


class TextToSpeech(Provider):
    """Turns text into audio."""

    @abc.abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """Yield audio chunks for the given text as they become available."""
