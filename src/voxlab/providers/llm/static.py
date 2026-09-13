"""A language model that always says the same thing.

Placeholder for running the pipeline without Ollama.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from voxlab.providers.base import LanguageModel
from voxlab.providers.registry import register_llm
from voxlab.types import Message

_DEFAULT = "Wir haben montags bis freitags von acht bis siebzehn Uhr geöffnet."


@register_llm("static")
class StaticLlm(LanguageModel):
    """Streams a fixed answer word by word."""

    name = "static"

    def __init__(self, reply: str = _DEFAULT, delay_s: float = 0.0, **_: object) -> None:
        self._reply = reply
        self._delay_s = delay_s

    async def stream(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        for word in self._reply.split():
            if self._delay_s:
                await asyncio.sleep(self._delay_s)
            yield word + " "
