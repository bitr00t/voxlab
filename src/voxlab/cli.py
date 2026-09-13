"""Command line entry point."""

from __future__ import annotations

import asyncio
import logging

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from voxlab import providers  # noqa: F401  (imported for provider registration)
from voxlab.config import load_settings
from voxlab.factory import build_pipeline
from voxlab.providers.registry import available
from voxlab.runtime import doctor as doctor_module

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

    table = Table(title="turn metrics (seconds)")
    table.add_column("turn", justify="right")
    for column in ("stt", "llm first token", "llm total", "tts first chunk", "total"):
        table.add_column(column, justify="right")
    for index, metrics in enumerate(pipeline.metrics, start=1):
        table.add_row(
            str(index),
            _fmt(metrics.stt_s),
            _fmt(metrics.llm_first_token_s),
            _fmt(metrics.llm_total_s),
            _fmt(metrics.tts_first_chunk_s),
            _fmt(metrics.total_s),
        )
    console.print(table)
    console.print(
        f"providers: stt={settings.stt.provider} llm={settings.llm.provider} "
        f"tts={settings.tts.provider} transport={settings.transport}"
    )


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    app()
