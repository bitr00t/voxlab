"""The Windows DLL fix has to be harmless everywhere else."""

from __future__ import annotations

import sys

from voxlab.runtime import cuda_windows


def test_no_op_off_windows(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cuda_windows, "_applied", False)
    if not sys.platform.startswith("win"):
        assert cuda_windows.ensure_cuda_dlls() == []


def test_is_idempotent(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cuda_windows, "_applied", False)
    cuda_windows.ensure_cuda_dlls()
    assert cuda_windows.ensure_cuda_dlls() == []
