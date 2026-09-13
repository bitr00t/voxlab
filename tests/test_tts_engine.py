"""The speech synthesis seam.

Two things are worth locking down: that the conversion from model output to the
project's PCM convention is correct, and that a machine without a usable
Qwen3-TTS package gets an error that says what to do about it.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from voxlab.providers.tts.engine import CustomEngine, Engine, EngineError, Samples, load_engine
from voxlab.providers.tts.qwen3 import Qwen3Tts


class _FakeEngine(Engine):
    def __init__(self, sample_rate: int = 24000) -> None:
        self.calls: list[tuple[str, str]] = []
        self._sample_rate = sample_rate

    def synthesize(self, text: str, voice: str) -> Samples:
        self.calls.append((text, voice))
        samples = np.zeros(self._sample_rate // 10, dtype=np.float32)
        return Samples(data=samples, sample_rate=self._sample_rate)


@pytest.fixture
def fake_adapter_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("fake_adapters")

    def create(model: str, device: str) -> _FakeEngine:
        module.last = {"model": model, "device": device}  # type: ignore[attr-defined]
        return _FakeEngine()

    def create_callable(model: str, device: str):  # type: ignore[no-untyped-def]
        def call(text: str, voice: str):  # type: ignore[no-untyped-def]
            return np.zeros(240, dtype=np.int16), 24000

        return call

    module.create = create  # type: ignore[attr-defined]
    module.create_callable = create_callable  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_adapters", module)
    return module


def test_custom_adapter_is_preferred_over_discovery(fake_adapter_module: types.ModuleType) -> None:
    engine = load_engine("qwen3-tts", "cuda", adapter="fake_adapters:create")
    assert isinstance(engine, CustomEngine)
    assert fake_adapter_module.last == {"model": "qwen3-tts", "device": "cuda"}  # type: ignore[attr-defined]


def test_custom_adapter_accepts_a_plain_callable(fake_adapter_module: types.ModuleType) -> None:
    engine = load_engine("qwen3-tts", "cpu", adapter="fake_adapters:create_callable")
    samples = engine.synthesize("Hallo.", "de-default")
    assert samples.sample_rate == 24000


def test_malformed_adapter_spec_is_rejected() -> None:
    with pytest.raises(EngineError, match="module:function"):
        load_engine("qwen3-tts", "cpu", adapter="no_colon_here")


def test_missing_adapter_module_names_the_spec() -> None:
    with pytest.raises(EngineError, match="does_not_exist"):
        load_engine("qwen3-tts", "cpu", adapter="does_not_exist:create")


def test_failure_points_at_the_probe_command(monkeypatch: pytest.MonkeyPatch) -> None:
    # No candidate package installed: the message has to be actionable rather
    # than just true.
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(EngineError) as excinfo:
        load_engine("qwen3-tts", "cuda")
    message = str(excinfo.value)
    assert "tts-probe" in message
    assert "VOXLAB__TTS__ADAPTER" in message


async def test_synthesis_emits_one_chunk_per_sentence() -> None:
    tts = Qwen3Tts(sample_rate=24000)
    engine = _FakeEngine()
    tts._engine = engine

    chunks = [chunk async for chunk in tts.synthesize("Guten Tag. Wie kann ich helfen?")]

    assert len(chunks) == 2
    assert [text for text, _ in engine.calls] == ["Guten Tag.", "Wie kann ich helfen?"]
    assert all(chunk.sample_rate == 24000 for chunk in chunks)


async def test_model_output_is_resampled_to_the_configured_rate() -> None:
    tts = Qwen3Tts(sample_rate=16000)
    tts._engine = _FakeEngine(sample_rate=24000)

    chunks = [chunk async for chunk in tts.synthesize("Ein Satz.")]

    assert chunks[0].sample_rate == 16000
    # 0.1 s of audio, regardless of what the model produced it at.
    assert abs(chunks[0].duration_s - 0.1) < 0.02


def test_float_output_is_clipped_before_scaling() -> None:
    tts = Qwen3Tts(sample_rate=24000)
    # A model that overshoots would wrap around on conversion and turn a loud
    # passage into noise.
    loud = np.array([2.0, -2.0, 0.5], dtype=np.float32)
    chunk = tts._to_chunk(Samples(data=loud, sample_rate=24000))
    values = np.frombuffer(chunk.pcm, dtype=np.int16)
    assert values[0] == 32767
    assert values[1] == -32767


def test_integer_output_passes_through_unscaled() -> None:
    tts = Qwen3Tts(sample_rate=24000)
    chunk = tts._to_chunk(Samples(data=np.array([100, -100], dtype=np.int16), sample_rate=24000))
    assert list(np.frombuffer(chunk.pcm, dtype=np.int16)) == [100, -100]


async def test_synthesize_before_start_is_an_error() -> None:
    tts = Qwen3Tts()
    with pytest.raises(RuntimeError, match="start"):
        [chunk async for chunk in tts.synthesize("Hallo.")]
