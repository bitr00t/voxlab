"""String-to-factory registry for providers.

Importing a provider module registers it; nothing else in the code base imports
a concrete provider class directly. This keeps optional heavy dependencies
optional: a missing faster-whisper installation makes one registry entry fail at
construction time, not the whole application at import time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

from voxlab.providers.base import LanguageModel, SpeechToText, TextToSpeech

T = TypeVar("T")

_STT: dict[str, Callable[..., SpeechToText]] = {}
_LLM: dict[str, Callable[..., LanguageModel]] = {}
_TTS: dict[str, Callable[..., TextToSpeech]] = {}


class ProviderNotFoundError(KeyError):
    """Raised when a configured provider name has no registration."""


def _register(table: dict[str, Callable[..., Any]], name: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        if name in table:
            raise ValueError(f"provider '{name}' is already registered")
        table[name] = cls
        return cls

    return decorator


def register_stt(name: str) -> Callable[[type[T]], type[T]]:
    return _register(_STT, name)


def register_llm(name: str) -> Callable[[type[T]], type[T]]:
    return _register(_LLM, name)


def register_tts(name: str) -> Callable[[type[T]], type[T]]:
    return _register(_TTS, name)


def _build(table: dict[str, Callable[..., Any]], kind: str, name: str, **kwargs: Any) -> Any:
    try:
        factory = table[name]
    except KeyError as exc:
        available = ", ".join(sorted(table)) or "none"
        raise ProviderNotFoundError(
            f"unknown {kind} provider '{name}'; available: {available}"
        ) from exc
    return factory(**kwargs)


def build_stt(name: str, **kwargs: Any) -> SpeechToText:
    return cast(SpeechToText, _build(_STT, "stt", name, **kwargs))


def build_llm(name: str, **kwargs: Any) -> LanguageModel:
    return cast(LanguageModel, _build(_LLM, "llm", name, **kwargs))


def build_tts(name: str, **kwargs: Any) -> TextToSpeech:
    return cast(TextToSpeech, _build(_TTS, "tts", name, **kwargs))


def available() -> dict[str, list[str]]:
    """Every registered provider name, grouped by kind."""
    return {
        "stt": sorted(_STT),
        "llm": sorted(_LLM),
        "tts": sorted(_TTS),
    }
