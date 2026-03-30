"""
utils/logger.py

Unified Rich console output and file logging for all DeepAxon entry points.
Instantiate DeepAxonLogger once at startup, pass the log path derived from
the run's output directory. All output goes to both console and log file.

Log format:
    ======================================================================
    DEEPAXON SEGMENT LOG
    ======================================================================
    Start time : 2026-04-01 14:32:11
    Git commit : 7560233
    ======================================================================

    ── SCANNING STUDY ────────────────────────────────────────────────────
    [14:32:11] [INFO]  Animals found: 3
    [14:32:11] [WARN]  Rb41/Rb41_Left_B — already segmented
    [14:32:12] [OK]    Rb41_RA_40X_001.tif -> 12.4s
    [14:32:24] [ERROR] Rb41_RA_40X_003.tif - FAILED: ...

    ── RUN COMPLETE ──────────────────────────────────────────────────────
    End time : 2026-04-01 14:47:33
    Elapsed  : 15m 22s
    ======================================================================
"""

from __future__ import annotations

import re
import os
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from utils.helpers import get_git_commit


class DeepAxonLogger:
    def __init__(
        self,
        log_path: str = None,
        program:  str = "DeepAxon",
        context:  dict = None
    ):
        """
        Args:
            log_path: Path to log file. If None logging is disabled.
            program:  Program name shown in log header.
            context:  Optional dict of key-value pairs written into header
                      e.g. {'Model': 'rabbit_40x_v1.keras', 'Magnification': '40X'}
        """
        self.console    = Console()
        self.log_path   = log_path
        self.program    = program
        self.context    = context or {}
        self._t_start   = datetime.now()

        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self._write_log_header()

    def _ts(self) -> str:
        """Current time as HH:MM:SS for per-line timestamps."""
        return datetime.now().strftime('%H:%M:%S')

    def _write_log_header(self):
        """Write structured header with run context."""
        git_commit = get_git_commit()
        lines = [
            f"{'=' * 72}",
            f"{self.program} LOG",
            f"{'=' * 72}",
            f"Start time : {self._t_start.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Git commit : {git_commit}",
            f"Log file   : {self.log_path}",
        ]
        # Write any additional context passed at init
        if self.context:
            for k, v in self.context.items():
                lines.append(f"{k:<11}: {v}")
        lines.append(f"{'=' * 72}")
        lines.append("")

        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

    def _append(self, text: str, tag: str = ""):
        """
        Append timestamped plain text to log file.
        Format: [HH:MM:SS] [TAG]  message
        """
        if self.log_path:
            ts      = self._ts()
            prefix  = f"[{ts}] {tag}" if tag else f"[{ts}]"
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(f"{prefix} {text}\n")

    def _append_raw(self, text: str):
        """Append text with no timestamp — for headers, dividers, sections."""
        if self.log_path:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(text + '\n')

    def info(self, msg: str):
        self.console.print(f"[cyan]ℹ[/cyan]  {msg}")
        self._append(f"[INFO]  {msg}")

    def success(self, msg: str):
        self.console.print(f"[green]✔[/green]  {msg}")
        self._append(f"[OK]    {msg}")

    def warn(self, msg: str):
        self.console.print(f"[yellow]⚠[/yellow]  {msg}")
        self._append(f"[WARN]  {msg}")

    def error(self, msg: str):
        self.console.print(f"[red]✖[/red]  {msg}")
        self._append(f"[ERROR] {msg}")

    def rule(self, title: str = ""):
        """Console rule + log section divider."""
        self.console.rule(f"[bold]{title}[/bold]")
        divider = f"\n── {title} {'─' * max(0, 68 - len(title))}"
        self._append_raw(divider)

    def print(self, msg: str):
        """Raw print — Rich markup on console, plain text in log."""
        self.console.print(msg)
        plain = re.sub(r'\[.*?\]', '', msg)
        self._append(plain)

    def log_dict(self, data: dict, title: str = ""):
        """Log a dictionary as aligned key: value pairs."""
        if title:
            self._append_raw(f"\n{title}")
        max_key = max(len(k) for k in data.keys()) if data else 0
        for k, v in data.items():
            line = f"  {k:<{max_key}} : {v}"
            self.console.print(line)
            self._append_raw(f"  {k:<{max_key}} : {v}")

    def write_section(self, title: str, content: str):
        """
        Write a named section directly to log only — not console.
        Used for large structured data like epoch tables and JSON metadata.
        Parseable by grep: search for '=== TITLE ===' to extract sections.
        """
        if self.log_path:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(
                    f"\n{'=' * 72}\n"
                    f"=== {title} ===\n"
                    f"{'=' * 72}\n"
                    f"{content}\n"
                    f"{'=' * 72}\n"
                )

    def finalize(self, summary: dict = None):
        """
        Write structured footer to log and console.
        Includes end time and elapsed duration.
        """
        t_end     = datetime.now()
        elapsed   = t_end - self._t_start
        mins      = int(elapsed.total_seconds() // 60)
        secs      = int(elapsed.total_seconds() % 60)
        elapsed_str = f"{mins}m {secs}s"

        self.rule("RUN COMPLETE")

        footer_lines = [
            f"End time : {t_end.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Elapsed  : {elapsed_str}",
        ]
        if summary:
            footer_lines.append("")
            max_key = max(len(k) for k in summary.keys()) if summary else 0
            for k, v in summary.items():
                footer_lines.append(f"  {k:<{max_key}} : {v}")
                self.console.print(f"  {k:<{max_key}} : {v}")

        footer_lines.append(f"\n{'=' * 72}")

        self._append_raw('\n'.join(footer_lines))

    def progress(self):
        """Return a Rich Progress context manager pre-configured for DeepAxon."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        )