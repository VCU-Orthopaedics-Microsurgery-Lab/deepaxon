"""
utils/gpu.py

GPU detection and setup for DeepAxon training.
Called once at startup by train/__main__.py.
"""

import os
from rich.console import Console
from rich.panel import Panel
from rich.box import DOUBLE
from rich.text import Text

console = Console()


def setup_gpu_console(use_gpu: bool = None) -> bool:
    """
    Configure TensorFlow GPU memory growth and return whether GPU is active.

    Args:
        use_gpu: if None, prompts the user. If True/False, sets directly.

    Returns:
        True if GPU is enabled, False if CPU only.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        t = Text(justify="center")
        t.append("No GPU detected — running on CPU only.\n", style="yellow")
        t.append("Training will be slower. Consider using a machine with a CUDA-capable GPU.")
        console.print(Panel(
            t,
            title="[bold yellow]Device[/bold yellow]",
            border_style="yellow",
            box=DOUBLE,
            expand=True
        ))
        return False

    if use_gpu is None:
        raw = input(f"GPU detected ({len(gpus)} device(s)). Use GPU acceleration? [Y/n]: ").strip().lower()
        use_gpu = raw not in ('n', 'no')

    if use_gpu:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            gpu_names = [gpu.name for gpu in gpus]
            t = Text(justify="center")
            t.append("GPU acceleration ENABLED\n", style="green")
            t.append(f"Device(s): {', '.join(gpu_names)}\n")
            t.append("Memory growth: enabled")
            console.print(Panel(
                t,
                title="[bold green]Device[/bold green]",
                border_style="green",
                box=DOUBLE,
                expand=True
            ))
            return True
        except RuntimeError as e:
            console.print(f"[red]GPU setup failed: {e}[/red]")
            console.print("[yellow]Falling back to CPU.[/yellow]")
            return False
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        t = Text("Running on CPU only (user selected).", justify="center")
        console.print(Panel(
            t,
            title="[bold cyan]Device[/bold cyan]",
            border_style="cyan",
            box=DOUBLE,
            expand=True
        ))
        return False