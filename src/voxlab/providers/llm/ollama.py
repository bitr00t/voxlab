"""Qwen3 served by Ollama over HTTP.

Ollama runs natively on Windows with CUDA support, which makes it the least
troublesome part of the local stack. The agent talks to it over HTTP rather than
loading the model in-process, so the model can later move into a container or
onto another machine without any change to this code.

A Qwen3 note that matters for a voice agent: the instruct models emit an
internal reasoning block before the answer. Reading that aloud would be
embarrassing, so it is stripped here rather than in the pipeline.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

import httpx

from voxlab.providers.base import LanguageModel
from voxlab.providers.registry import register_llm
from voxlab.types import Message

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


@register_llm("ollama")
class OllamaLlm(LanguageModel):
    """Streaming chat completions from a local Ollama server."""

    name = "ollama"

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://127.0.0.1:11434",
        temperature: float = 0.3,
        timeout_s: float = 120.0,
        **_: object,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout_s, connect=5.0),
            )

    async def stream(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        if self._client is None:
            raise RuntimeError("call start() before stream()")

        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {"temperature": self._temperature},
        }

        in_reasoning = False
        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                event = json.loads(line)
                fragment = event.get("message", {}).get("content", "")
                if not fragment:
                    continue

                # Suppress the reasoning block. Handled fragment by fragment
                # because the tags can arrive split across chunks.
                if _THINK_OPEN in fragment:
                    in_reasoning = True
                    fragment = fragment.split(_THINK_OPEN, 1)[0]
                if in_reasoning:
                    if _THINK_CLOSE in fragment:
                        in_reasoning = False
                        fragment = fragment.split(_THINK_CLOSE, 1)[1]
                    else:
                        continue
                if fragment:
                    yield fragment

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
