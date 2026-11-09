"""
SENTINEL logging utilities.

Provides Rich-based colored console logging with optional progress bars
for long-running data acquisition tasks.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import Generator, Sequence

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

_console = Console(stderr=True)


def get_logger(
    name: str,
    level: int | str = logging.INFO,
    *,
    show_path: bool = False,
    rich_tracebacks: bool = True,
) -> logging.Logger:
    """Return a logger configured with a Rich console handler.

    Parameters
    ----------
    name:
        Logger name (typically ``__name__``).
    level:
        Logging level.
    show_path:
        If *True*, display the source file path in each log line.
    rich_tracebacks:
        If *True*, use Rich for traceback rendering.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    handler = RichHandler(
        console=_console,
        show_time=True,
        show_level=True,
        show_path=show_path,
        rich_tracebacks=rich_tracebacks,
        tracebacks_show_locals=False,
        markup=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def make_progress(
    *,
    transient: bool = False,
    disable: bool = False,
) -> Progress:
    """Create a Rich Progress bar suited for data download / processing.

    Usage::

        progress = make_progress()
        with progress:
            task = progress.add_task("Downloading tiles...", total=100)
            for tile in tiles:
                download(tile)
                progress.advance(task)
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=_console,
        transient=transient,
        disable=disable,
    )


@contextmanager
def progress_context(
    description: str,
    total: int | float,
    *,
    transient: bool = False,
) -> Generator[tuple[Progress, int], None, None]:
    """Convenience context manager that yields ``(progress, task_id)``.

    Usage::

        with progress_context("Processing stations", len(stations)) as (prog, tid):
            for station in stations:
                process(station)
                prog.advance(tid)
    """
    progress = make_progress(transient=transient)
    with progress:
        task_id = progress.add_task(description, total=total)
        yield progress, task_id


def log_banner(title: str, *, items: Sequence[str] | None = None) -> None:
    """Print a styled banner to the console."""
    _console.rule(f"[bold green]{title}")
    if items:
        for item in items:
            _console.print(f"  [cyan]•[/cyan] {item}")
