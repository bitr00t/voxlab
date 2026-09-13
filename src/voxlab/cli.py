"""Command line entry point."""

from __future__ import annotations

import asyncio
import logging

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from voxlab import audio as audio_package  # noqa: F401  (transport registration)
from voxlab import providers  # noqa: F401  (imported for provider registration)
from voxlab.config import load_settings
from voxlab.factory import build_pipeline
from voxlab.providers.registry import available
from voxlab.runtime import doctor as doctor_module
from voxlab.types import TurnMetrics

app = typer.Typer(add_completion=False, help="voxlab - local-first voice agent")
console = Console()

_STATUS_STYLE = {"ok": "green", "warn": "yellow", "fail": "red", "skip": "dim"}


@app.command()
def doctor() -> None:
    """Check that this machine can run the stack."""
    settings = load_settings()
    table = Table(title="voxlab environment", show_lines=False)
    table.add_column("check", style="bold")
    table.add_column("status")
    table.add_column("detail", overflow="fold")

    checks = doctor_module.run_all(settings.llm.base_url, settings.llm.model)
    for check in checks:
        style = _STATUS_STYLE[check.status]
        # Escaped: details contain extras like .[stt], which rich would
        # otherwise swallow as markup.
        table.add_row(check.name, f"[{style}]{check.status}[/{style}]", escape(check.detail))
    console.print(table)

    if any(check.status == "fail" for check in checks):
        raise typer.Exit(code=1)


@app.command("providers")
def list_providers() -> None:
    """Show every registered provider."""
    table = Table(title="registered providers")
    table.add_column("kind", style="bold")
    table.add_column("names")
    for kind, names in available().items():
        table.add_row(kind, ", ".join(names))
    console.print(table)


@app.command()
def config() -> None:
    """Show the effective configuration."""
    console.print_json(load_settings().model_dump_json(indent=2))


@app.command()
def smoke(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Run one turn through the configured pipeline and print the timings.

    With the default placeholder providers this needs no GPU, no model weights
    and no microphone - it proves the wiring, not the speech.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings()
    pipeline = build_pipeline(settings)

    async def _run() -> None:
        await pipeline.start()
        try:
            await pipeline.run()
        finally:
            await pipeline.aclose()

    asyncio.run(_run())

    _print_metrics(list(pipeline.metrics))
    console.print(
        f"providers: stt={settings.stt.provider} llm={settings.llm.provider} "
        f"tts={settings.tts.provider} transport={settings.transport}"
    )


@app.command()
def devices() -> None:
    """List the audio devices, with the index to put in the configuration."""
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        console.print(f"[red]sounddevice unavailable:[/red] {exc}")
        console.print("Install the audio extra: pip install -e \".[audio]\"")
        raise typer.Exit(code=1) from exc

    table = Table(title="audio devices")
    for column in ("index", "name", "in", "out", "default rate"):
        table.add_column(column)
    for index, device in enumerate(sd.query_devices()):
        table.add_row(
            str(index),
            escape(str(device["name"])),
            str(device["max_input_channels"]),
            str(device["max_output_channels"]),
            f"{device['default_samplerate']:.0f}",
        )
    console.print(table)
    default_in, default_out = sd.default.device
    console.print(f"defaults: input={default_in} output={default_out}")


@app.command("tts-probe")
def tts_probe() -> None:
    """Report which speech synthesis packages are installed and what they expose.

    Run this when 'voxlab talk' fails to construct an engine: the output shows
    which adapter to write, or which package is missing.
    """
    from voxlab.providers.tts.engine import probe

    table = Table(title="speech synthesis packages")
    table.add_column("module", style="bold")
    table.add_column("status")
    table.add_column("details", overflow="fold")
    for module, info in probe().items():
        if info is None:
            table.add_row(module, "[dim]not installed[/dim]", "")
        elif isinstance(info, str):
            table.add_row(module, "[red]error[/red]", escape(info))
        else:
            names = ", ".join(info["public_names"])
            detail = escape(f"{info['version']} | {names}")
            table.add_row(module, "[green]installed[/green]", detail)
    console.print(table)


@app.command()
def talk(
    turns: int | None = typer.Option(None, "--turns", "-n", help="stop after this many turns"),
    save_dir: str | None = typer.Option(None, "--save-dir", help="write one WAV per turn"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Hold a conversation through the local microphone and speakers.

    Press Enter to start speaking, Enter again when finished, 'q' to stop.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings()
    # Command line arguments win over the configuration file.
    settings.transport = "local"
    if save_dir is not None:
        settings.audio.save_dir = save_dir
    if turns is not None:
        settings.audio.max_turns = turns

    pipeline = build_pipeline(settings)

    console.print(
        f"stt=[bold]{settings.stt.provider}[/bold] "
        f"llm=[bold]{settings.llm.provider}[/bold] ({settings.llm.model}) "
        f"tts=[bold]{settings.tts.provider}[/bold]"
    )
    console.print("[dim]loading models …[/dim]")

    async def _run() -> None:
        await pipeline.start()
        try:
            await pipeline.run()
        finally:
            await pipeline.aclose()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/dim]")

    if pipeline.metrics:
        _print_metrics(pipeline.metrics)


def _print_metrics(metrics: list[TurnMetrics]) -> None:
    table = Table(title="turn metrics (seconds)")
    table.add_column("turn", justify="right")
    for column in ("stt", "llm first token", "llm total", "tts first chunk", "total"):
        table.add_column(column, justify="right")
    for index, entry in enumerate(metrics, start=1):
        table.add_row(
            str(index),
            _fmt(entry.stt_s),
            _fmt(entry.llm_first_token_s),
            _fmt(entry.llm_total_s),
            _fmt(entry.tts_first_chunk_s),
            _fmt(entry.total_s),
        )
    console.print(table)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    app()
