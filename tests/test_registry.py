"""The registry is the contract that keeps providers swappable."""

from __future__ import annotations

import pytest

from voxlab import providers  # noqa: F401  (registration side effect)
from voxlab.providers.registry import (
    ProviderNotFoundError,
    available,
    build_llm,
    build_stt,
    build_tts,
)


def test_every_kind_has_a_placeholder_provider() -> None:
    registered = available()
    assert "echo" in registered["stt"]
    assert "static" in registered["llm"]
    assert "silent" in registered["tts"]


def test_real_providers_are_registered_without_being_importable_at_load_time() -> None:
    registered = available()
    assert "faster-whisper" in registered["stt"]
    assert "ollama" in registered["llm"]
    assert "qwen3" in registered["tts"]


def test_unknown_name_lists_the_alternatives() -> None:
    with pytest.raises(ProviderNotFoundError) as excinfo:
        build_stt("does-not-exist")
    assert "available" in str(excinfo.value)


def test_providers_ignore_settings_they_do_not_use() -> None:
    # The factory passes the full settings block to every provider, so each one
    # has to tolerate keys meant for a different backend.
    build_stt("echo", model="large-v3", device="cuda", compute_type="float16", language="de")
    build_llm("static", model="qwen3:8b", base_url="http://127.0.0.1:11434", temperature=0.3)
    build_tts("silent", model="qwen3-tts", voice="de-default", device="cuda", sample_rate=24000)
