"""Environment checks.

Every failure mode this project hit during setup gets a check here, so the next
person - including future me on a fresh machine - gets a diagnosis instead of a
stack trace.
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal

import httpx

Status = Literal["ok", "warn", "fail", "skip"]

# Read from a constant rather than compared inline, so the guard survives even
# when a static analyser assumes the declared minimum interpreter.
_MIN_PYTHON = (3, 11)


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str


def run_all(
    llm_base_url: str = "http://127.0.0.1:11434",
    llm_model: str = "qwen3:8b",
) -> list[Check]:
    return [
        _check_python(),
        _check_platform(),
        _check_nvidia_smi(),
        _check_cuda_dlls(),
        _check_ctranslate2(),
        _check_faster_whisper(),
        _check_ollama(llm_base_url, llm_model),
        _check_audio_devices(),
    ]


def _check_python() -> Check:
    version = ".".join(str(p) for p in sys.version_info[:3])
    if sys.version_info[:2] < _MIN_PYTHON:
        minimum = ".".join(str(p) for p in _MIN_PYTHON)
        return Check("python", "fail", f"{version} - this project needs {minimum} or newer")
    return Check("python", "ok", version)


def _check_platform() -> Check:
    detail = f"{platform.system()} {platform.release()}"
    if sys.platform.startswith("win"):
        detail += " - the Windows CUDA DLL fix is active"
    return Check("platform", "ok", detail)


def _check_nvidia_smi() -> Check:
    if shutil.which("nvidia-smi") is None:
        return Check("gpu", "warn", "nvidia-smi not found - CPU-only providers will still work")
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        return Check("gpu", "warn", f"nvidia-smi failed: {exc}")
    return Check("gpu", "ok", output.replace("\n", " | "))


def _check_cuda_dlls() -> Check:
    from voxlab.runtime.cuda_windows import ensure_cuda_dlls

    if not sys.platform.startswith("win"):
        return Check("cuda-dlls", "skip", "not Windows")
    added = ensure_cuda_dlls()
    if not added:
        return Check(
            "cuda-dlls", "warn",
            "no bundled cuBLAS/cuDNN found - install the extras: pip install -e .[stt]",
        )
    return Check("cuda-dlls", "ok", f"{len(added)} directory/ies added to the DLL search path")


def _check_ctranslate2() -> Check:
    if importlib.util.find_spec("ctranslate2") is None:
        return Check("ctranslate2", "warn", "not installed - pip install -e .[stt]")
    from voxlab.runtime.cuda_windows import ensure_cuda_dlls

    ensure_cuda_dlls()
    try:
        import ctranslate2
    except OSError as exc:
        # The classic Windows symptom: the wheel is there, the DLLs are not.
        return Check(
            "ctranslate2", "fail",
            f"import failed ({exc}). Usually a cuDNN major version mismatch - "
            "check that nvidia-cudnn-cu12 matches what this CTranslate2 build expects.",
        )
    except Exception as exc:  # pragma: no cover
        return Check("ctranslate2", "fail", f"import failed: {exc}")
    count = getattr(ctranslate2, "get_cuda_device_count", lambda: 0)()
    status: Status = "ok" if count else "warn"
    return Check("ctranslate2", status, f"{ctranslate2.__version__}, CUDA devices: {count}")


def _check_faster_whisper() -> Check:
    if importlib.util.find_spec("faster_whisper") is None:
        return Check("faster-whisper", "warn", "not installed - pip install -e .[stt]")
    return Check("faster-whisper", "ok", "installed")


def _check_ollama(base_url: str, model: str) -> Check:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=3.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return Check("ollama", "warn", f"not reachable at {base_url} ({type(exc).__name__})")
    names = [entry.get("name", "") for entry in response.json().get("models", [])]
    if not any(name.startswith(model.split(":")[0]) for name in names):
        return Check(
            "ollama", "warn", f"reachable, but no '{model}' model - run: ollama pull {model}"
        )
    return Check("ollama", "ok", f"reachable, {len(names)} model(s) available")


def _check_audio_devices() -> Check:
    if importlib.util.find_spec("sounddevice") is None:
        return Check("audio", "skip", "sounddevice not installed - needed from Phase 1")
    try:
        import sounddevice as sd

        inputs = [d for d in sd.query_devices() if d.get("max_input_channels", 0) > 0]
    except Exception as exc:  # pragma: no cover - depends on the host
        return Check("audio", "warn", f"device query failed: {exc}")
    status: Status = "ok" if inputs else "warn"
    return Check("audio", status, f"{len(inputs)} input device(s)")
