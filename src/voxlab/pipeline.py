"""The turn loop: listen, transcribe, think, speak.

Phase 0 wires the seams together and measures itself. It is deliberately a
half-duplex loop - the agent finishes speaking before it listens again. Barge-in
turns this into a full-duplex problem and belongs to Phase 2, together with the
transport that can actually interrupt playback.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from voxlab.audio.transport import AudioTransport
from voxlab.providers.base import LanguageModel, SpeechToText, TextToSpeech
from voxlab.types import AudioChunk, Message, TurnMetrics

_LOG = logging.getLogger(__name__)


@dataclass
class Conversation:
    """The running message history handed to the language model."""

    system_prompt: str
    messages: list[Message] = field(default_factory=list)
    max_turns: int = 12

    def to_list(self) -> list[Message]:
        return [Message(role="system", content=self.system_prompt), *self.messages]

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))  # type: ignore[arg-type]
        # Keep the context bounded. Summarising instead of truncating is a
        # later refinement; dropping the oldest pairs is fine for short calls.
        excess = len(self.messages) - self.max_turns * 2
        if excess > 0:
            del self.messages[:excess]


class VoicePipeline:
    """Connects a transport to the three providers."""

    def __init__(
        self,
        transport: AudioTransport,
        stt: SpeechToText,
        llm: LanguageModel,
        tts: TextToSpeech,
        system_prompt: str,
    ) -> None:
        self._transport = transport
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._conversation = Conversation(system_prompt=system_prompt)
        self.metrics: list[TurnMetrics] = []

    async def start(self) -> None:
        await self._transport.start()
        await self._stt.start()
        await self._llm.start()
        await self._tts.start()

    async def aclose(self) -> None:
        # Closed in reverse order, and each one independently: a failure while
        # releasing the GPU must not leak the HTTP client.
        for component in (self._tts, self._llm, self._stt, self._transport):
            try:
                await component.aclose()
            except Exception:  # pragma: no cover - best effort cleanup
                _LOG.exception("error closing %s", type(component).__name__)

    async def run(self) -> None:
        """Handle utterances until the transport stops producing them."""
        async for utterance in self._transport.utterances():
            await self.handle_turn(utterance)

    async def handle_turn(self, utterance: AudioChunk) -> TurnMetrics:
        metrics = TurnMetrics()
        turn_started = time.perf_counter()

        transcript = await self._stt.transcribe(utterance)
        metrics.stt_s = time.perf_counter() - turn_started
        _LOG.info("user: %s", transcript.text)
        self._conversation.add("user", transcript.text)

        reply, llm_metrics = await self._collect_reply()
        metrics.llm_first_token_s = llm_metrics[0]
        metrics.llm_total_s = llm_metrics[1]
        _LOG.info("agent: %s", reply)
        self._conversation.add("assistant", reply)

        synthesis_started = time.perf_counter()
        first_chunk_seen = False

        async def timed_chunks() -> AsyncIterator[AudioChunk]:
            nonlocal first_chunk_seen
            async for chunk in self._tts.synthesize(reply):
                if not first_chunk_seen:
                    metrics.tts_first_chunk_s = time.perf_counter() - synthesis_started
                    first_chunk_seen = True
                yield chunk

        await self._transport.play(timed_chunks())
        metrics.total_s = time.perf_counter() - turn_started
        self.metrics.append(metrics)
        return metrics

    async def _collect_reply(self) -> tuple[str, tuple[float | None, float]]:
        """Drain the model stream into a string, timing the first fragment.

        Phase 2 replaces this with sentence-wise hand-off to the synthesiser so
        that audio starts before the model has finished. The measurement is here
        already so the improvement is visible in numbers when it lands.
        """
        started = time.perf_counter()
        first_token_s: float | None = None
        parts: list[str] = []
        async for fragment in self._llm.stream(self._conversation.to_list()):
            if first_token_s is None:
                first_token_s = time.perf_counter() - started
            parts.append(fragment)
        return "".join(parts).strip(), (first_token_s, time.perf_counter() - started)
