"""Sample rate conversion.

Whisper works at 16 kHz, capture devices often do not offer that rate, and
speech synthesis usually emits 22.05 or 24 kHz. Something has to convert, and
doing it in one place keeps the providers free of audio plumbing.

Quality matters more on the way in than on the way out: a poor resampler in
front of speech recognition costs accuracy, while a poor one in front of the
speakers merely sounds slightly worse. soxr is used when it is installed, which
the `audio` extra ensures; the linear fallback exists so that a missing optional
dependency degrades the sound instead of breaking the call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

_LOG = logging.getLogger(__name__)
_warned = False


def resample_pcm(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """Convert mono 16-bit PCM between sample rates."""
    if source_rate == target_rate or not pcm:
        return pcm

    import numpy as np

    samples = np.frombuffer(pcm, dtype=np.int16)
    converted = _resample_array(samples, source_rate, target_rate)
    # Annotated rather than returned directly: without numpy installed, mypy
    # sees the whole module as Any and rejects the inferred return type under
    # warn_return_any. The annotation makes the contract explicit either way.
    raw: bytes = converted.astype(np.int16).tobytes()
    return raw


def _resample_array(
    samples: npt.NDArray[np.int16], source_rate: int, target_rate: int
) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:
    import numpy as np

    array = np.asarray(samples)
    try:
        import soxr
    except ImportError:
        return _resample_linear(array, source_rate, target_rate)
    converted: npt.NDArray[np.float32] = soxr.resample(
        array.astype(np.float32), source_rate, target_rate
    )
    return converted


def _resample_linear(
    samples: npt.NDArray[np.int16], source_rate: int, target_rate: int
) -> npt.NDArray[np.float64]:
    """Fallback: linear interpolation.

    Audible as slight aliasing on the way out, and measurable as a small
    accuracy loss on the way in. Acceptable as a fallback, not as a default.
    """
    global _warned
    import numpy as np

    if not _warned:
        _LOG.warning(
            "soxr is not installed; falling back to linear resampling. "
            "Install the audio extra for better quality: pip install -e .[audio]"
        )
        _warned = True

    array = np.asarray(samples, dtype=np.float32)
    target_length = max(1, int(round(len(array) * target_rate / source_rate)))
    source_positions = np.arange(len(array), dtype=np.float64)
    target_positions = np.linspace(0, len(array) - 1, target_length, dtype=np.float64)
    return np.interp(target_positions, source_positions, array)
