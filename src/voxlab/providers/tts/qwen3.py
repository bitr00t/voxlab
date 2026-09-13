"""Qwen3-TTS on the local GPU.

Synthesis happens one sentence at a time rather than once for the whole reply.
That is what makes the first audio leave early: a four-sentence answer otherwise
stays silent until the last word has been rendered, which is most of the latency
budget spent on nothing.

The model call itself is blocking and runs in a worker thread. The uncertain
part - how to actually invoke Qwen3-TTS - lives in voxlab.providers.tts.engine,
so this module stays stable when the upstream API moves.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from voxlab.audio.resample import resample_pcm
from voxlab.providers.base import TextToSpeech
from voxlab.providers.registry import register_tts
from voxlab.providers.tts.engine import Engine, Samples, load_engine
from voxlab.text import split_sentences
from voxlab.types import AudioChunk

_LOG = logging.getLogger(__name__)


@register_tts("qwen3")
class Qwen3Tts(TextToSpeech):
    """Local neural speech synthesis."""

    name = "qwen3"

    def __init__(
        self,
        model: str = "qwen3-tts",
        voice: str = "de-default",
        device: str = "cuda",
        sample_rate: int = 24000,
        adapter: str | None = None,
        **_: object,
    ) -> None:
        self._model_name = model
        self._voice = voice
        self._device = device
        self._sample_rate = sample_rate
        self._adapter = adapter
        self._engine: Engine | None = None

    async def start(self) -> None:
        if self._engine is not None:
            return
        self._engine = await asyncio.to_thread(
            load_engine, self._model_name, self._device, self._adapter
        )
        _LOG.info("speech synthesis ready: %s", type(self._engine).__name__)

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        if self._engine is None:
            raise RuntimeError("call start() before synthesize()")

        for sentence in split_sentences(text):
            samples = await asyncio.to_thread(self._engine.synthesize, sentence, self._voice)
            yield self._to_chunk(samples)

    def _to_chunk(self, samples: Samples) -> AudioChunk:
        """Convert model output to the project's PCM convention.

        Models emit float32 in [-1, 1] or int16, at whichever rate they were
        trained on. Both are normalised here so that no other part of the system
        has to know which.
        """
        import numpy as np

        array = np.asarray(samples.data).squeeze()
        if array.dtype.kind == "f":
            # Clip before scaling: a model that overshoots would otherwise wrap
            # around and turn a loud passage into noise.
            array = np.clip(array, -1.0, 1.0)
            array = (array * 32767.0).astype(np.int16)
        else:
            array = array.astype(np.int16)

        pcm = resample_pcm(array.tobytes(), samples.sample_rate, self._sample_rate)
        return AudioChunk(pcm=pcm, sample_rate=self._sample_rate)

    async def aclose(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None
