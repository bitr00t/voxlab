"""Configuration, loaded from environment variables and an optional .env file.

Provider selection is a string in the configuration, never an import in the
calling code. That is the whole point of Phase 0: `VOXLAB__TTS__PROVIDER=qwen3`
and `VOXLAB__TTS__PROVIDER=silent` must be interchangeable without touching the
pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SttSettings(BaseModel):
    provider: str = "echo"
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = "de"
    # Only used by HTTP-backed providers (Phase 2 deployment style).
    base_url: str | None = None


class LlmSettings(BaseModel):
    provider: str = "static"
    model: str = "qwen3:8b"
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.3
    system_prompt: str = (
        "Du bist die telefonische Erstauskunft eines mittelständischen "
        "Unternehmens. Antworte kurz, höflich und in gesprochenem Deutsch. "
        "Wenn du etwas nicht sicher weißt, sage das und biete an, an einen "
        "Mitarbeiter weiterzuleiten."
    )


class TtsSettings(BaseModel):
    provider: str = "silent"
    model: str = "qwen3-tts"
    voice: str = "de-default"
    device: str = "cuda"
    sample_rate: int = 24000
    base_url: str | None = None
    # Escape hatch for a moving upstream API: "module:function", returning an
    # engine. See voxlab.providers.tts.engine.
    adapter: str | None = None


class AudioSettings(BaseModel):
    """Capture and playback, used by the local device transport."""

    # None means the operating system default. An index or a substring of the
    # device name both work; `voxlab devices` lists them.
    input_device: int | str | None = None
    output_device: int | str | None = None
    # What the recogniser wants. The transport falls back to the device's own
    # rate and resamples when this one is refused.
    capture_sample_rate: int = 16000
    block_size: int = 1024
    # Where to write one WAV per turn. Off by default; the evaluation harness in
    # phase 4 is what this collects material for.
    save_dir: str | None = None
    # None means "until the user quits". Mostly useful for scripted runs.
    max_turns: int | None = None


class Settings(BaseSettings):
    """Root configuration object."""

    model_config = SettingsConfigDict(
        env_prefix="VOXLAB__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    stt: SttSettings = Field(default_factory=SttSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    tts: TtsSettings = Field(default_factory=TtsSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    transport: str = "null"


def load_settings() -> Settings:
    """Read the configuration. Kept as a function so tests can bypass it."""
    return Settings()
