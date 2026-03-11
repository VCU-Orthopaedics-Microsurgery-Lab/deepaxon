# train/utils/console_utils.py
"""
-------------------------------- DEEPAXON --------------------------------
Console utilities for styled output and progress using Rich.
"""

# ------------------------------ Third-Party Imports --------------------------- #
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn

# ------------------------------ Initialize Console ----------------------------- #
console = Console()

# ------------------------------ Printing Helpers ------------------------------ #
def info(msg: str):
    """Print an informational message."""
    console.print(f"[cyan][INFO][/cyan] {msg}")

def success(msg: str):
    """Print a success message."""
    console.print(f"[green][SUCCESS][/green] {msg}")

def warn(msg: str):
    """Print a warning message."""
    console.print(f"[yellow][WARN][/yellow] {msg}")

def error(msg: str):
    """Print an error message."""
    console.print(f"[bold red][ERROR][/bold red] {msg}")

def header(title: str):
    """Print a styled section header."""
    console.print(Panel.fit(Text(title, justify="center", style="bold magenta"), title="DeepAxon++", border_style="magenta"))

def rule(text: str):
    """Draw a horizontal rule with text."""
    console.rule(f"[bold magenta]{text}[/bold magenta]")

# ------------------------------ Progress Helpers ------------------------------ #
def progress_bar():
    """Return a progress bar context manager."""
    return Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
    )