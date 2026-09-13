"""The Phase 0 acceptance test: one turn survives the whole pipeline."""

from __future__ import annotations

from voxlab import providers  # noqa: F401
from voxlab.audio.transport import NullTransport
from voxlab.config import Settings
from voxlab.factory import build_pipeline
from voxlab.pipeline import VoicePipeline
from voxlab.providers.registry import build_llm, build_stt, build_tts
from voxlab.types import AudioChunk


async def test_turn_produces_audio_and_metrics() -> None:
    transport = NullTransport(turns=2)
    pipeline = VoicePipeline(
        transport=transport,
        stt=build_stt("echo", text="Wie sind Ihre Öffnungszeiten?"),
        llm=build_llm("static", reply="Montags bis freitags von acht bis siebzehn Uhr."),
        tts=build_tts("silent", sample_rate=24000),
        system_prompt="test",
    )
    await pipeline.start()
    try:
        await pipeline.run()
    finally:
        await pipeline.aclose()

    assert len(pipeline.metrics) == 2
    assert transport.played_chunks
    assert all(chunk.sample_rate == 24000 for chunk in transport.played_chunks)
    for metrics in pipeline.metrics:
        assert metrics.total_s is not None
        assert metrics.time_to_first_audio_s is not None


async def test_conversation_history_grows_with_both_roles() -> None:
    pipeline = VoicePipeline(
        transport=NullTransport(turns=1),
        stt=build_stt("echo"),
        llm=build_llm("static"),
        tts=build_tts("silent"),
        system_prompt="test",
    )
    await pipeline.start()
    await pipeline.handle_turn(AudioChunk(pcm=b"", sample_rate=16000))
    await pipeline.aclose()

    history = pipeline._conversation.to_list()
    assert [m.role for m in history] == ["system", "user", "assistant"]


async def test_factory_builds_from_default_settings() -> None:
    # Defaults must be the placeholder providers, so a fresh clone runs.
    pipeline = build_pipeline(Settings())
    await pipeline.start()
    await pipeline.aclose()
