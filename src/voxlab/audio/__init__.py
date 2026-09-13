"""Audio capture, playback and the conversions between them.

Importing this package registers every transport. Transport modules must stay
import-safe: no device access and no PortAudio initialisation at import time.
"""

from voxlab.audio import device  # noqa: F401  (imported for registration)
from voxlab.audio.transport import (  # noqa: F401
    AudioTransport,
    NullTransport,
    build_transport,
    register_transport,
)

__all__ = ["AudioTransport", "NullTransport", "build_transport", "register_transport"]
