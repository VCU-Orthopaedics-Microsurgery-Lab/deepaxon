# train/utils/gpu.py
"""
GPU utilities for TensorFlow/Keras
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error

import tensorflow as tf
from rich.console import Console
from rich.panel import Panel

console = Console()

def setup_gpu_console():
    """
    Enable memory growth for all available GPUs and report status to console.
    Returns the number of GPUs detected.
    """
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        console.print(Panel.fit(
            "[bold yellow]No GPU detected. Running on CPU.[/bold yellow]",
            border_style="red"
        ))
        return 0

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    gpu_names = ", ".join([gpu.name for gpu in gpus])
    console.print(Panel.fit(
        f"[bold green]Detected GPUs: {gpu_names}. Running with GPU acceleration (TensorFlow {tf.__version__}).[/bold green]",
        border_style="green"
    ))

    return len(gpus)