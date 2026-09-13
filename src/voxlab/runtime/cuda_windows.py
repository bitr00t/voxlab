"""Make the pip-installed CUDA libraries loadable on Windows.

Installing nvidia-cublas-cu12 and nvidia-cudnn-cu12 from PyPI avoids a manual
CUDA Toolkit installation, but it only solves half the problem on Windows: the
DLLs land in site-packages, and the Windows loader does not search there. The
symptom is a hard failure at the first CTranslate2 import - sometimes an error
about a missing cudnn_ops64_9.dll, sometimes a silent process exit with no
traceback at all.

Since Python 3.8, `os.add_dll_directory` is the supported fix. It must run
before the first import of ctranslate2, which is why providers call this and not
the other way round.

On Linux and macOS this function does nothing: there the wheels ship an RPATH
that already resolves.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_LOG = logging.getLogger(__name__)

# Relative to the site-packages root.
_DLL_SUBDIRECTORIES = (
    Path("nvidia") / "cudnn" / "bin",
    Path("nvidia") / "cublas" / "bin",
    Path("nvidia") / "cuda_runtime" / "bin",
)

_applied = False


def ensure_cuda_dlls() -> list[Path]:
    """Add the bundled NVIDIA DLL directories to the search path.

    Returns the directories that were added. Idempotent, and safe to call on
    platforms where it is a no-op.
    """
    global _applied
    if _applied or not sys.platform.startswith("win"):
        return []

    # Looked up dynamically: the function only exists on Windows, so a direct
    # attribute access does not type check on other platforms.
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:  # pragma: no cover - Windows only
        return []

    added: list[Path] = []
    for site_dir in _site_package_roots():
        for subdirectory in _DLL_SUBDIRECTORIES:
            candidate = site_dir / subdirectory
            if not candidate.is_dir():
                continue
            try:
                add_dll_directory(str(candidate))
            except OSError:  # pragma: no cover - platform specific
                _LOG.warning("could not add DLL directory %s", candidate)
                continue
            added.append(candidate)

    if not added:
        _LOG.warning(
            "no bundled NVIDIA DLL directories found. If CTranslate2 fails to "
            "load, install the CUDA extras: pip install -e .[stt]"
        )
    _applied = True
    return added


def _site_package_roots() -> list[Path]:
    import site

    roots: list[Path] = []
    getter = getattr(site, "getsitepackages", None)
    if getter is not None:
        roots.extend(Path(p) for p in getter())
    user_site = getattr(site, "getusersitepackages", None)
    if callable(user_site):
        roots.append(Path(user_site()))
    # A virtual environment's site-packages is not always covered by the above.
    roots.extend(Path(p) for p in sys.path if p.endswith("site-packages"))
    return [r for r in dict.fromkeys(roots) if r.is_dir()]
