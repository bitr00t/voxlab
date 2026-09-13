"""Recordings are the raw material for the phase 4 evaluation harness."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from voxlab.audio.wav import turn_paths, write_wav
from voxlab.types import AudioChunk


def test_roundtrip_preserves_frames_and_rate(tmp_path: Path) -> None:
    chunk = AudioChunk(pcm=b"\x01\x00" * 800, sample_rate=16000)
    path = write_wav(tmp_path / "nested" / "turn.wav", chunk)

    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000
        assert handle.getnframes() == 800


def test_multiple_chunks_are_concatenated(tmp_path: Path) -> None:
    chunks = [AudioChunk(pcm=b"\x00\x00" * 100, sample_rate=24000) for _ in range(3)]
    path = write_wav(tmp_path / "out.wav", chunks)
    with wave.open(str(path), "rb") as handle:
        assert handle.getnframes() == 300


def test_mixed_sample_rates_are_rejected(tmp_path: Path) -> None:
    chunks = [
        AudioChunk(pcm=b"\x00\x00", sample_rate=16000),
        AudioChunk(pcm=b"\x00\x00", sample_rate=24000),
    ]
    with pytest.raises(ValueError):
        write_wav(tmp_path / "bad.wav", chunks)


def test_turn_paths_are_distinct_and_ordered(tmp_path: Path) -> None:
    inbound, outbound = turn_paths(tmp_path, 7)
    assert inbound != outbound
    assert "turn007" in inbound.name
    assert inbound.name.endswith("-in.wav")
    assert outbound.name.endswith("-out.wav")
