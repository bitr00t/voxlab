"""Writing captured and synthesised audio to disk.

Recording every turn is not a debugging convenience. The evaluation harness in
phase 4 needs real utterances from a real microphone in a real room, and those
have to be collected while the agent is being used rather than produced on
demand afterwards.
"""

from __future__ import annotations

import datetime as dt
import wave
from pathlib import Path

from voxlab.types import AudioChunk


def write_wav(path: Path, chunks: list[AudioChunk] | AudioChunk) -> Path:
    """Write mono 16-bit PCM to a WAV file, creating parent directories."""
    blocks = [chunks] if isinstance(chunks, AudioChunk) else list(chunks)
    if not blocks:
        raise ValueError("nothing to write")

    sample_rate = blocks[0].sample_rate
    if any(block.sample_rate != sample_rate for block in blocks):
        raise ValueError("all chunks must share one sample rate")

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(block.pcm for block in blocks))
    return path


def turn_paths(directory: Path, turn: int) -> tuple[Path, Path]:
    """Timestamped file names for one turn's input and output."""
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = f"{stamp}-turn{turn:03d}"
    return directory / f"{prefix}-in.wav", directory / f"{prefix}-out.wav"
