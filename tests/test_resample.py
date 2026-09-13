"""Resampling sits in front of the recogniser, so its arithmetic has to hold."""

from __future__ import annotations

import numpy as np

from voxlab.audio.resample import resample_pcm


def _tone(frequency: int, rate: int, seconds: float) -> bytes:
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return (np.sin(2 * np.pi * frequency * t) * 16000).astype(np.int16).tobytes()


def test_identical_rates_are_a_no_op() -> None:
    pcm = _tone(440, 16000, 0.1)
    assert resample_pcm(pcm, 16000, 16000) is pcm


def test_empty_input_is_returned_unchanged() -> None:
    assert resample_pcm(b"", 44100, 16000) == b""


def test_downsampling_scales_the_sample_count() -> None:
    pcm = _tone(440, 48000, 0.25)
    converted = resample_pcm(pcm, 48000, 16000)
    frames = len(converted) // 2
    assert abs(frames - 4000) < 100  # a few frames of slack for filter delay


def test_upsampling_scales_the_sample_count() -> None:
    pcm = _tone(440, 16000, 0.1)
    converted = resample_pcm(pcm, 16000, 24000)
    frames = len(converted) // 2
    assert abs(frames - 2400) < 100


def test_output_stays_within_int16_range() -> None:
    pcm = _tone(3000, 44100, 0.2)
    converted = np.frombuffer(resample_pcm(pcm, 44100, 16000), dtype=np.int16)
    assert converted.size
    assert np.abs(converted).max() <= 32767
