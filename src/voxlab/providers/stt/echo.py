"""A speech-to-text provider that does not listen.

Its job is to let the pipeline, the CLI and the test suite run end to end on a
machine with no GPU and no model weights. Every phase keeps a placeholder
provider like this one for exactly that reason.
"""

from __future__ import annotations

from voxlab.providers.base import SpeechToText
from voxlab.providers.registry import register_stt
from voxlab.types import AudioChunk, Transcript


@register_stt("echo")
class EchoStt(SpeechToText):
    """Returns a fixed transcript, ignoring the audio."""

    name = "echo"

    def __init__(self, text: str = "Wie sind Ihre Öffnungszeiten?", **_: object) -> None:
        self._text = text

    async def transcribe(self, audio: AudioChunk) -> Transcript:
        return Transcript(text=self._text, is_final=True, language="de", confidence=1.0)
