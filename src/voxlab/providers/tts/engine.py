"""The boundary between this project and the Qwen3-TTS package.

Everything uncertain about the upstream library is confined to this module. The
provider above it handles sentence chunking, threading and PCM conversion and
does not change when the model API does.

That separation is the point. Local speech synthesis is the youngest part of the
stack and its Python interfaces move: a package gets renamed, a class gains a
streaming method, a call signature changes. Rather than guessing at one shape
and hard-coding it, an engine is anything that can turn text into samples, and
three ways of obtaining one are supported:

1. An explicit factory named in the configuration, as `module:function`. This is
   the escape hatch, and the one to reach for after reading the upstream README:
   fifteen lines in a file of your own beats fighting a wrong assumption here.
2. A transformers-based engine, for weights published as a Hugging Face model.
3. A package-level entry point, tried against a few plausible call shapes.

`voxlab tts-probe` reports which of these are available on this machine and what
each one exposes, so a mismatch is a two-minute fix rather than an afternoon.
"""

from __future__ import annotations

import abc
import importlib
import importlib.util
import logging
from dataclasses import dataclass
from typing import Any

_LOG = logging.getLogger(__name__)

# Candidate distribution names, newest naming first.
_CANDIDATE_MODULES = ("qwen_tts", "qwen3_tts", "qwentts")
# Candidate methods on a loaded model object, in the order they are tried.
_CANDIDATE_METHODS = ("generate", "synthesize", "tts", "infer", "__call__")


class EngineError(RuntimeError):
    """Raised when no usable engine can be constructed."""


@dataclass(frozen=True)
class Samples:
    """Raw model output, before conversion to the project's PCM convention."""

    data: Any  # numpy array, float32 in [-1, 1] or int16
    sample_rate: int


class Engine(abc.ABC):
    """Turns a piece of text into samples. Blocking by design."""

    @abc.abstractmethod
    def synthesize(self, text: str, voice: str) -> Samples: ...

    def close(self) -> None:
        """Release the model. Dropping the reference is usually enough."""


class CustomEngine(Engine):
    """Wraps a callable supplied through configuration.

    The factory is called as `factory(model=..., voice=..., device=...)` and must
    return an object with a `synthesize(text, voice) -> Samples` method, or a
    plain callable `(text, voice) -> (samples, sample_rate)`.
    """

    def __init__(self, spec: str, model: str, device: str) -> None:
        module_name, _, attribute = spec.partition(":")
        if not attribute:
            raise EngineError(f"adapter spec must be 'module:function', got '{spec}'")
        try:
            module = importlib.import_module(module_name)
            factory = getattr(module, attribute)
        except (ImportError, AttributeError) as exc:
            raise EngineError(f"could not load adapter '{spec}': {exc}") from exc
        self._inner = factory(model=model, device=device)

    def synthesize(self, text: str, voice: str) -> Samples:
        inner_synthesize = getattr(self._inner, "synthesize", None)
        if inner_synthesize is not None:
            result = inner_synthesize(text, voice)
            return result if isinstance(result, Samples) else Samples(*result)
        data, sample_rate = self._inner(text, voice)
        return Samples(data=data, sample_rate=sample_rate)


class TransformersEngine(Engine):
    """For weights published as a Hugging Face model.

    Best effort, and the first thing to check against the model card if the
    output is silent or garbled: processors differ in whether they want the
    voice as a `speaker` argument, as part of the prompt, or not at all.
    """

    def __init__(self, model: str, device: str) -> None:
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise EngineError("transformers is not installed") from exc
        try:
            self._processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(model, trust_remote_code=True).to(device)
        except Exception as exc:  # noqa: BLE001 - the failure modes here are many
            raise EngineError(f"could not load '{model}' via transformers: {exc}") from exc
        self._device = device

    def synthesize(self, text: str, voice: str) -> Samples:
        inputs = self._processor(text=text, return_tensors="pt").to(self._device)
        output = self._model.generate(**inputs)
        sample_rate = int(
            getattr(getattr(self._model, "config", None), "sampling_rate", 0)
            or getattr(self._processor, "sampling_rate", 0)
            or 24000
        )
        waveform = output.detach().to("cpu").float().numpy().squeeze()
        return Samples(data=waveform, sample_rate=sample_rate)


class PackageEngine(Engine):
    """For a package that exposes a model class directly."""

    def __init__(self, module_name: str, model: str, device: str) -> None:
        module = importlib.import_module(module_name)
        model_class = _first_model_class(module)
        if model_class is None:
            raise EngineError(f"no usable class found in '{module_name}'")
        try:
            self._model = model_class(model, device=device)
        except TypeError:
            self._model = model_class(model)
        method = _first_method(self._model)
        if method is None:
            raise EngineError(
                f"'{module_name}' model exposes none of: {', '.join(_CANDIDATE_METHODS)}"
            )
        self._method: str = method

    def synthesize(self, text: str, voice: str) -> Samples:
        call = getattr(self._model, self._method) if self._method != "__call__" else self._model
        try:
            result = call(text, voice=voice)
        except TypeError:
            result = call(text)
        if isinstance(result, Samples):
            return result
        if isinstance(result, tuple) and len(result) == 2:
            return Samples(data=result[0], sample_rate=int(result[1]))
        raise EngineError(
            "unexpected return value from the model; wrap it with a custom adapter "
            "(VOXLAB__TTS__ADAPTER=module:function)"
        )


def load_engine(model: str, device: str, adapter: str | None = None) -> Engine:
    """Construct an engine, most explicit option first."""
    if adapter:
        return CustomEngine(adapter, model=model, device=device)

    attempts: list[str] = []
    for module_name in _CANDIDATE_MODULES:
        if importlib.util.find_spec(module_name) is None:
            continue
        try:
            return PackageEngine(module_name, model=model, device=device)
        except (EngineError, Exception) as exc:  # noqa: BLE001
            attempts.append(f"{module_name}: {exc}")

    if importlib.util.find_spec("transformers") is not None:
        try:
            return TransformersEngine(model=model, device=device)
        except EngineError as exc:
            attempts.append(f"transformers: {exc}")

    detail = "; ".join(attempts) if attempts else "no candidate package is installed"
    raise EngineError(
        f"no Qwen3-TTS engine could be constructed ({detail}). "
        "Run 'voxlab tts-probe' to see what is installed, then point "
        "VOXLAB__TTS__ADAPTER at your own module:function factory."
    )


def probe() -> dict[str, Any]:
    """What is installed and what it exposes. Used by `voxlab tts-probe`."""
    report: dict[str, Any] = {}
    for module_name in (*_CANDIDATE_MODULES, "transformers", "torch", "soundfile"):
        if importlib.util.find_spec(module_name) is None:
            report[module_name] = None
            continue
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            report[module_name] = f"import failed: {exc}"
            continue
        report[module_name] = {
            "version": getattr(module, "__version__", "unknown"),
            "public_names": [name for name in dir(module) if not name.startswith("_")][:30],
        }
    return report


def _first_model_class(module: Any) -> Any | None:
    for name in dir(module):
        if name.startswith("_"):
            continue
        candidate = getattr(module, name)
        if isinstance(candidate, type) and "tts" in name.lower():
            return candidate
    return None


def _first_method(model: Any) -> str | None:
    for method in _CANDIDATE_METHODS:
        if callable(getattr(model, method, None)):
            return method
    return None
