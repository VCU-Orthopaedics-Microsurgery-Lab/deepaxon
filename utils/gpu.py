"""
utils/gpu.py

Detect GPU availability and print device panel.
Returns True if GPU is available, False if CPU only.
"""

from rich.panel import Panel
from rich.console import Console
from rich.box import DOUBLE
from rich.text import Text
from utils.helpers import print_panel

import torch

console = Console()

def setup_gpu_console() -> bool:
    """
    Configure PyTorch GPU and return whether GPU is active.

    Args:
        use_gpu: if None, prompts the user. If True/False, sets directly.

    Returns:
        True if GPU is enabled, False if CPU only.
    """

    if not torch.cuda.is_available():
        t = Text(justify="center")
        t.append("No GPU detected — running on CPU only.\n", style="yellow")
        t.append("Training and Segmentation will be slower without GPU acceleration.")
        print_panel(console, Panel(
            t,
            title="[bold yellow]Device[/bold yellow]",
            border_style="yellow",
            box=DOUBLE,
            expand=True
        ))
        return False

    num_gpus = torch.cuda.device_count()

    
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
    t = Text(justify="center")
    t.append("GPU acceleration ENABLED\n", style="green")
    t.append(f"Device(s): {', '.join(gpu_names)}\n")
    t.append(f"CUDA version: {torch.version.cuda}")
    print_panel(console, Panel(
        t,
        title="[bold green]Device[/bold green]",
        border_style="green",
        box=DOUBLE,
        expand=True
    ))
    return True