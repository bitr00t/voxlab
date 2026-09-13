"""The local transport, exercised against a fake PortAudio.

Audio hardware cannot be assumed in CI, so sounddevice is replaced with a double
that records what the transport asked it to do. That still covers the parts that
actually broke during development: rate negotiation, buffer draining and the
lazy opening of the output stream.
"""

from __future__ import annotations

import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from voxlab.audio import device as device_module
from voxlab.audio.device import LocalDeviceTransport
from voxlab.types import AudioChunk


class _FakeInputStream:
    def __init__(self, callback: Any, blocks: int, block_size: int, **kwargs: Any) -> None:
        self.callback = callback
        self.blocks = blocks
        self.block_size = block_size
        self.kwargs = kwargs

    def __enter__(self) -> _FakeInputStream:
        for _ in range(self.blocks):
            self.callback(b"\x01\x00" * self.block_size, self.block_size, None, None)
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _FakeOutputStream:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.written: list[bytes] = []
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def write(self, data: bytes) -> None:
        self.written.append(data)

    def stop(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _fake_sounddevice(
    *, accepts_target_rate: bool = True, default_rate: float = 48000.0, blocks: int = 3
) -> Any:
    created: dict[str, Any] = {}

    def check_input_settings(**_: Any) -> None:
        if not accepts_target_rate:
            raise ValueError("unsupported sample rate")

    def query_devices(*_: Any, **__: Any) -> dict[str, Any]:
        return {"name": "fake", "default_samplerate": default_rate}

    def raw_input_stream(**kwargs: Any) -> _FakeInputStream:
        stream = _FakeInputStream(
            callback=kwargs["callback"],
            blocks=blocks,
            block_size=kwargs["blocksize"],
            **{k: v for k, v in kwargs.items() if k != "callback"},
        )
        created["input"] = stream
        return stream

    def raw_output_stream(**kwargs: Any) -> _FakeOutputStream:
        stream = _FakeOutputStream(**kwargs)
        created["output"] = stream
        return stream

    module = types.SimpleNamespace(
        check_input_settings=check_input_settings,
        query_devices=query_devices,
        RawInputStream=raw_input_stream,
        RawOutputStream=raw_output_stream,
        created=created,
    )
    return module


@pytest.fixture
def answers(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Queue of replies to the push-to-talk prompts."""
    queued: list[str] = []

    async def fake_ask(_prompt: str) -> str:
        return queued.pop(0) if queued else "q"

    monkeypatch.setattr(device_module, "_ask", fake_ask)
    return queued


async def _install(
    monkeypatch: pytest.MonkeyPatch, transport: LocalDeviceTransport, sd: Any
) -> None:
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", sd)
    await transport.start()


async def test_capture_uses_the_target_rate_when_the_device_accepts_it(
    monkeypatch: pytest.MonkeyPatch, answers: list[str]
) -> None:
    sd = _fake_sounddevice(accepts_target_rate=True)
    transport = LocalDeviceTransport(target_sample_rate=16000, block_size=256)
    await _install(monkeypatch, transport, sd)

    answers.extend(["", "", "q"])
    captured = [chunk async for chunk in transport.utterances()]

    assert len(captured) == 1
    assert captured[0].sample_rate == 16000
    assert sd.created["input"].kwargs["samplerate"] == 16000
    assert len(captured[0].pcm) == 3 * 256 * 2


async def test_capture_falls_back_to_the_device_rate_and_resamples(
    monkeypatch: pytest.MonkeyPatch, answers: list[str]
) -> None:
    sd = _fake_sounddevice(accepts_target_rate=False, default_rate=48000.0)
    transport = LocalDeviceTransport(target_sample_rate=16000, block_size=480)
    await _install(monkeypatch, transport, sd)

    answers.extend(["", "", "q"])
    captured = [chunk async for chunk in transport.utterances()]

    assert sd.created["input"].kwargs["samplerate"] == 48000
    # Still handed to the recogniser at the rate it expects.
    assert captured[0].sample_rate == 16000
    frames = len(captured[0].pcm) // 2
    assert abs(frames - 480) < 50


async def test_quitting_at_the_prompt_ends_the_loop(
    monkeypatch: pytest.MonkeyPatch, answers: list[str]
) -> None:
    transport = LocalDeviceTransport()
    await _install(monkeypatch, transport, _fake_sounddevice())
    answers.append("q")
    assert [chunk async for chunk in transport.utterances()] == []


async def test_max_turns_stops_without_a_quit(
    monkeypatch: pytest.MonkeyPatch, answers: list[str]
) -> None:
    transport = LocalDeviceTransport(block_size=64, max_turns=2)
    await _install(monkeypatch, transport, _fake_sounddevice())
    answers.extend(["", "", "", ""])
    assert len([chunk async for chunk in transport.utterances()]) == 2


async def test_playback_opens_the_device_at_the_rate_of_the_audio(
    monkeypatch: pytest.MonkeyPatch, answers: list[str]
) -> None:
    sd = _fake_sounddevice()
    transport = LocalDeviceTransport()
    await _install(monkeypatch, transport, sd)

    async def chunks() -> AsyncIterator[AudioChunk]:
        for _ in range(2):
            yield AudioChunk(pcm=b"\x02\x00" * 50, sample_rate=24000)

    await transport.play(chunks())

    output = sd.created["output"]
    # The rate comes from the audio, not from configuration, so swapping the
    # synthesiser cannot silently detune playback.
    assert output.kwargs["samplerate"] == 24000
    assert output.started and output.closed
    assert len(output.written) == 2


async def test_recordings_are_written_when_a_directory_is_configured(
    monkeypatch: pytest.MonkeyPatch, answers: list[str], tmp_path: Path
) -> None:
    transport = LocalDeviceTransport(block_size=64, save_dir=str(tmp_path), max_turns=1)
    await _install(monkeypatch, transport, _fake_sounddevice())
    answers.extend(["", ""])

    async for _ in transport.utterances():
        pass

    assert list(tmp_path.glob("*-in.wav"))
