"""Provider implementations.

Importing this package imports every provider module for its registration side
effect. Provider modules must therefore stay import-safe: no model loading, no
CUDA initialisation and no network access at import time.
"""

from voxlab.providers import llm, stt, tts  # noqa: F401  (imported for registration)
