"""Build a pipeline from settings.

The single place where configuration strings turn into objects.
"""

from __future__ import annotations

from voxlab.audio.transport import build_transport
from voxlab.config import Settings
from voxlab.pipeline import VoicePipeline
from voxlab.providers import registry


def build_pipeline(settings: Settings) -> VoicePipeline:
    stt = registry.build_stt(
        settings.stt.provider,
        model=settings.stt.model,
        device=settings.stt.device,
        compute_type=settings.stt.compute_type,
        language=settings.stt.language,
        base_url=settings.stt.base_url,
    )
    llm = registry.build_llm(
        settings.llm.provider,
        model=settings.llm.model,
        base_url=settings.llm.base_url,
        temperature=settings.llm.temperature,
    )
    tts = registry.build_tts(
        settings.tts.provider,
        model=settings.tts.model,
        voice=settings.tts.voice,
        device=settings.tts.device,
        sample_rate=settings.tts.sample_rate,
        base_url=settings.tts.base_url,
    )
    transport = build_transport(settings.transport)
    return VoicePipeline(
        transport=transport,
        stt=stt,
        llm=llm,
        tts=tts,
        system_prompt=settings.llm.system_prompt,
    )
