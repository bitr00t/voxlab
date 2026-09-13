"""The audio transport seam.

This is the architectural decision that shapes the project, so it gets its own
abstraction rather than being implicit in the pipeline:

    Phase 1 - audio in and out through the host OS (sounddevice, local mic).
    Phase 2 - audio in and out through the browser over WebRTC, with the models
              running behind HTTP endpoints.

Phase 2 exists because of Windows. WSL2 passes the GPU through but has no direct
microphone access, so any setup that puts the models in WSL2 or Docker and the
microphone on the host runs into a dead end. Routing audio through the browser
sidesteps the Windows audio stack entirely: the models become HTTP services that
do not care where they run, and the demo becomes something that can be shown
live over a screen share instead of as a recorded video.

The pipeline only ever sees this interface, so that switch costs one
configuration value.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator

from voxlab.types import AudioChunk


class AudioTransport(abc.ABC):
    """Where captured audio comes from and where synthesised audio goes."""

    name: str = "unnamed"

    async def start(self) -> None:
        """Open devices or sockets."""

    async def aclose(self) -> None:
        """Close them again."""

    @abc.abstractmethod
    def utterances(self) -> AsyncIterator[AudioChunk]:
        """Yield one complete user utterance at a time.

        Utterance segmentation - voice activity detection, end-of-turn
        detection, barge-in - belongs here and not in the speech-to-text
        provider, because it depends on how audio arrives rather than on which
        model transcribes it.
        """

    @abc.abstractmethod
    async def play(self, chunks: AsyncIterator[AudioChunk]) -> None:
        """Play synthesised audio, starting as soon as the first chunk arrives."""


class NullTransport(AudioTransport):
    """A transport with no audio hardware.

    Yields a fixed number of empty utterances and discards anything played.
    This is what makes the Phase 0 smoke test runnable in CI.
    """

    name = "null"

    def __init__(self, turns: int = 1, sample_rate: int = 16000, **_: object) -> None:
        self._turns = turns
        self._sample_rate = sample_rate
        self.played_chunks: list[AudioChunk] = []

    async def utterances(self) -> AsyncIterator[AudioChunk]:
        for _ in range(self._turns):
            yield AudioChunk(pcm=b"", sample_rate=self._sample_rate)

    async def play(self, chunks: AsyncIterator[AudioChunk]) -> None:
        async for chunk in chunks:
            self.played_chunks.append(chunk)


_TRANSPORTS: dict[str, type[AudioTransport]] = {"null": NullTransport}


def build_transport(name: str, **kwargs: object) -> AudioTransport:
    """Construct a transport by name.

    Phase 1 registers a sounddevice-backed transport here, Phase 2 a WebRTC one.
    """
    try:
        cls = _TRANSPORTS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_TRANSPORTS))
        raise KeyError(f"unknown transport '{name}'; available: {available}") from exc
    return cls(**kwargs)


def register_transport(name: str, cls: type[AudioTransport]) -> None:
    _TRANSPORTS[name] = cls
