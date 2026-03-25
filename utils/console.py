"""
utils/console.py

Unified Rich console output and file logging for all DeepAxon entry points.
Instantiate DeepAxonLogger once at startup, pass the log path derived from
the run's output directory. All output goes to both console and log file.
"""

from __future__ import annotations

import re
import os
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn


class DeepAxonLogger:
    def __init__(self, log_path: str = None, program: str = "DeepAxon"):
        self.console  = Console()
        self.log_path = log_path
        self.program  = program

        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self._write_log_header()

    def _write_log_header(self):
        header = (
            f"{'=' * 70}\n"
            f"{self.program} LOG\n"
            f"{'=' * 70}\n"
            f"Start time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Log file   : {self.log_path}\n"
            f"{'=' * 70}\n\n"
        )
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write(header)

    def _append(self, text: str):
        """Append plain text to log file."""
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
        self.console.rule(f"[bold]{title}[/bold]")
        self._append(f"\n{'─' * 70}  {title}\n")

    def print(self, msg: str):
        """Raw print — use for Rich markup that should also go to log."""
        self.console.print(msg)
        plain = re.sub(r'\[.*?\]', '', msg)
        self._append(plain)

    def log_dict(self, data: dict, title: str = ""):
        """Log a dictionary as aligned key: value pairs."""
        if title:
            self._append(f"\n{title}")
        max_key = max(len(k) for k in data.keys()) if data else 0
        lines   = []
        for k, v in data.items():
            line = f"  {k:<{max_key}} : {v}"
            lines.append(line)
            self.console.print(line)
        self._append('\n'.join(lines))

    def write_section(self, title: str, content: str):
        """Write a named section directly to log (not console)."""
        if self.log_path:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 70}\n{title}\n{'=' * 70}\n{content}\n")

    def finalize(self, summary: dict = None):
        """Write final summary to log and console."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.rule("RUN COMPLETE")
        self._append(f"\nEnd time : {timestamp}\n")
        if summary:
            self.log_dict(summary, title="FINAL SUMMARY")

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
