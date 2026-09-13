"""Push-to-talk capture and playback through the host audio devices.

The phase 1 transport. Turn taking is manual: press Enter to start speaking,
Enter again when finished. That is a deliberate simplification - automatic
end-of-turn detection is a quality problem of its own, and solving it at the
same time as getting three models to speak German would have hidden which of
the two was broken. Voice activity detection arrives with the browser transport
in phase 2, behind this same interface.

Everything here runs off the event loop. PortAudio delivers capture buffers on
its own thread into a queue, and the blocking playback write happens in a worker
thread, so a slow synthesiser cannot stall the input side.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from voxlab.audio.resample import resample_pcm
from voxlab.audio.transport import AudioTransport, register_transport
from voxlab.audio.wav import turn_paths, write_wav
from voxlab.types import AudioChunk

_LOG = logging.getLogger(__name__)

_PROMPT_SPEAK = "\n[Enter] sprechen  ·  [q] beenden > "
_PROMPT_STOP = "aufnahme läuft … [Enter] fertig > "


class LocalDeviceTransport(AudioTransport):
    """Microphone in, speakers out."""

    name = "local"

    def __init__(
        self,
        target_sample_rate: int = 16000,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        block_size: int = 1024,
        save_dir: str | None = None,
        max_turns: int | None = None,
        **_: object,
    ) -> None:
        self._target_sample_rate = target_sample_rate
        self._input_device = input_device
        self._output_device = output_device
        self._block_size = block_size
        self._save_dir = Path(save_dir) if save_dir else None
        self._max_turns = max_turns
        self._sd: Any | None = None
        self._capture_rate: int = target_sample_rate
        self._turn = 0

    async def start(self) -> None:
        if self._sd is not None:
            return
        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:
            # OSError rather than ImportError when the PortAudio shared library
            # is missing, which is a different problem with the same fix.
            raise RuntimeError(
                "sounddevice is unavailable. Install the audio extra: "
                "pip install -e .[audio]"
            ) from exc
        self._sd = sd
        self._capture_rate = self._resolve_capture_rate()
        if self._capture_rate != self._target_sample_rate:
            _LOG.info(
                "capturing at %d Hz and resampling to %d Hz",
                self._capture_rate,
                self._target_sample_rate,
            )

    def _resolve_capture_rate(self) -> int:
        """Prefer the rate the recogniser wants, fall back to the device's own.

        Not every input device accepts 16 kHz, and PortAudio reports that by
        refusing to open the stream rather than by resampling silently.
        """
        assert self._sd is not None
        try:
            self._sd.check_input_settings(
                device=self._input_device,
                channels=1,
                dtype="int16",
                samplerate=self._target_sample_rate,
            )
        except Exception:  # noqa: BLE001 - PortAudioError and ValueError both occur
            info = self._sd.query_devices(self._input_device, "input")
            return int(info["default_samplerate"])
        return self._target_sample_rate

    async def utterances(self) -> AsyncIterator[AudioChunk]:
        while self._max_turns is None or self._turn < self._max_turns:
            answer = (await _ask(_PROMPT_SPEAK)).strip().lower()
            if answer in {"q", "quit", "exit"}:
                return
            self._turn += 1
            captured = await self._record()
            if not captured.pcm:
                _LOG.warning("no audio captured - is the right input device selected?")
                continue
            if self._save_dir:
                write_wav(turn_paths(self._save_dir, self._turn)[0], captured)
            yield captured

    async def _record(self) -> AudioChunk:
        """Capture until the user presses Enter again."""
        assert self._sd is not None
        buffers: queue.Queue[bytes] = queue.Queue()

        def callback(indata: Any, _frames: int, _time: Any, status: Any) -> None:
            if status:  # overflows are recoverable and worth knowing about
                _LOG.debug("input stream status: %s", status)
            buffers.put(bytes(indata))

        stream = self._sd.RawInputStream(
            samplerate=self._capture_rate,
            blocksize=self._block_size,
            device=self._input_device,
            channels=1,
            dtype="int16",
            callback=callback,
        )
        with stream:
            await _ask(_PROMPT_STOP)

        blocks: list[bytes] = []
        while not buffers.empty():
            blocks.append(buffers.get_nowait())
        pcm = resample_pcm(b"".join(blocks), self._capture_rate, self._target_sample_rate)
        return AudioChunk(pcm=pcm, sample_rate=self._target_sample_rate)

    async def play(self, chunks: AsyncIterator[AudioChunk]) -> None:
        """Play chunks as they arrive, opening the device on the first one.

        The output rate comes from the audio itself rather than from
        configuration, so a change of synthesiser cannot silently detune the
        playback.
        """
        assert self._sd is not None
        stream: Any | None = None
        played: list[AudioChunk] = []
        try:
            async for chunk in chunks:
                if not chunk.pcm:
                    continue
                if stream is None:
                    stream = self._sd.RawOutputStream(
                        samplerate=chunk.sample_rate,
                        device=self._output_device,
                        channels=1,
                        dtype="int16",
                    )
                    stream.start()
                elif chunk.sample_rate != played[0].sample_rate:
                    raise ValueError("sample rate changed mid-utterance")
                played.append(chunk)
                # Blocking write, moved off the event loop.
                await asyncio.to_thread(stream.write, chunk.pcm)
        finally:
            if stream is not None:
                stream.stop()
                stream.close()

        if self._save_dir and played:
            write_wav(turn_paths(self._save_dir, self._turn)[1], played)

    async def aclose(self) -> None:
        self._sd = None


async def _ask(prompt: str) -> str:
    """Read a line from stdin without blocking the event loop."""
    def _read() -> str:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        return sys.stdin.readline()

    return await asyncio.to_thread(_read)


register_transport("local", LocalDeviceTransport)
