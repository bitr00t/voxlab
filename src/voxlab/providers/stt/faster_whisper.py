"""faster-whisper large-v3 in float16 on the local GPU.

Roughly 4.5 GB of VRAM at float16, which leaves ample room on a 20 GB card for
the language model alongside it.

Two things about this provider are Windows-specific and account for most of the
setup pain:

1. faster-whisper runs on CTranslate2, which needs cuBLAS and cuDNN. Installing
   the nvidia-cublas-cu12 and nvidia-cudnn-cu12 wheels avoids a manual CUDA
   Toolkit install, but on Windows the DLLs then sit inside site-packages where
   the loader does not look. `ensure_cuda_dlls()` fixes that and must run before
   the first import of ctranslate2.
2. The cuDNN major version has to match what the installed CTranslate2 was built
   against. A mismatch shows up as a bare "Library cudnn_ops64_9.dll is not
   found" or an immediate process exit with no Python traceback at all.

The model call itself is synchronous and releases the GIL only partially, so it
runs in a worker thread to keep the event loop free for audio I/O.
"""

from __future__ import annotations

import asyncio
from typing import Any

from voxlab.providers.base import SpeechToText
from voxlab.providers.registry import register_stt
from voxlab.runtime.cuda_windows import ensure_cuda_dlls
from voxlab.types import AudioChunk, Transcript


@register_stt("faster-whisper")
class FasterWhisperStt(SpeechToText):
    """Local Whisper transcription via CTranslate2."""

    name = "faster-whisper"

    def __init__(
        self,
        model: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str | None = "de",
        **_: object,
    ) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._model: Any | None = None

    async def start(self) -> None:
        if self._model is not None:
            return
        ensure_cuda_dlls()
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise RuntimeError(
                "faster-whisper is not installed. Run: pip install -e .[stt]"
            ) from exc

        def _load() -> Any:
            return WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            )

        # The first call downloads several gigabytes of weights. Loading in a
        # thread keeps the event loop responsive while that happens.
        self._model = await asyncio.to_thread(_load)

    async def transcribe(self, audio: AudioChunk) -> Transcript:
        if self._model is None:
            raise RuntimeError("call start() before transcribe()")

        # CTranslate2 expects float32 samples in [-1, 1] at 16 kHz.
        import numpy as np

        samples = np.frombuffer(audio.pcm, dtype=np.int16).astype(np.float32) / 32768.0

        def _run() -> tuple[str, str | None, float | None]:
            assert self._model is not None
            segments, info = self._model.transcribe(
                samples,
                language=self._language,
                beam_size=5,
                vad_filter=False,  # voice activity detection lives in the transport
            )
            text = "".join(segment.text for segment in segments).strip()
            probability = getattr(info, "language_probability", None)
            return text, getattr(info, "language", None), probability

        text, language, confidence = await asyncio.to_thread(_run)
        return Transcript(text=text, is_final=True, language=language, confidence=confidence)

    async def aclose(self) -> None:
        # Dropping the reference is what frees the VRAM; CTranslate2 has no
        # explicit unload call.
        self._model = None
