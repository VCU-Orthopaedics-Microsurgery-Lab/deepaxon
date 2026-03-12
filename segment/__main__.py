"""
segment/__main__.py

Entry point for: python segment

Walks a study directory, detects magnification, presents model selection menu,
segments all nerves. Skips nerves that already have a Segmented folder.
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF warnings 0=all, 1=info, 2=warning, 3=error
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.console import DeepAxonLogger
from utils.helpers import scan_study, detect_study_mag, list_models, load_config
from utils.gpu import setup_gpu_console
from segment.segment import segment_dir

console = Console()


def select_model(models: list) -> Path:
    """Present an interactive model selection menu."""
    console.print("\n[bold cyan]Available models:[/bold cyan]")
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Model", style="white")
    
    for i, m in enumerate(models, 1):
        table.add_row(str(i), m.stem)
        
    console.print(table)

    while True:
        raw = input(f"Select model [1-{len(models)}]: ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(models):
                return models[idx]
        except ValueError:
            pass
        console.print(f"[red]Please enter a number between 1 and {len(models)}[/red]")

def main():
    console.print(Panel(
        "[bold white]Nerve Cross-Section Segmentation[/bold white]",
        title="[bold cyan]DeepAxon — Segment[/bold cyan]",
        border_style="cyan",
        expand=False
    ))

    # ── Study folder input ────────────────────────────────────────────────────
    study_dir = input("\nInput the path to the study folder: ").strip().strip('"')
    if not os.path.isdir(study_dir):
        console.print(f"[red]Study folder not found: {study_dir}[/red]")
        sys.exit(1)

    study_name = Path(study_dir).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = str(Path(study_dir) / f"segment_log_{timestamp}.txt")
    log = DeepAxonLogger(log_path=log_path, program="DeepAxon Segment")

    log.info(f"Study: {study_dir}")

    # ── Scan study ────────────────────────────────────────────────────────────
    log.rule("SCANNING STUDY")
    study = scan_study(study_dir)

    total_nerves = sum(len(nerves) for nerves in study.values())
    if total_nerves == 0:
        log.error("No nerve folders with TIFF directories found.")
        sys.exit(1)

    mag = detect_study_mag(study)
    log.info(f"Detected magnification: [bold]{mag}[/bold]")
    log.info(f"Animals found: {len(study)}")
    log.info(f"Nerves found: {total_nerves}")

    # Identify already-segmented nerves
    to_process = []
    to_skip = []
    for animal, nerves in study.items():
        for nerve, info in nerves.items():
            if info['segmented_dir'] is not None:
                to_skip.append(f"{animal}/{nerve}")
            else:
                to_process.append((animal, nerve, info))

    if to_skip:
        log.warn(f"Skipping {len(to_skip)} already-segmented nerve(s):")
        for s in to_skip:
            log.warn(f"  {s}")

    if not to_process:
        log.success("All nerves already segmented. Nothing to do.")
        sys.exit(0)

    log.info(f"To process: {len(to_process)} nerve(s)")

    # ── Model selection ───────────────────────────────────────────────────────
    models = list_models()
    if not models:
        log.error("No .keras models found in models/ directory.")
        sys.exit(1)

    selected_model_path = select_model(models)
    log.info(f"Selected model: {selected_model_path.stem}")

    # ── GPU setup ─────────────────────────────────────────────────────────────
    setup_gpu_console()

    # ── Load model ────────────────────────────────────────────────────────────
    log.rule("LOADING MODEL")
    import tensorflow as tf
    from utils.metrics import dice_coef, dice_loss, iou_coef, combined_loss

    custom_objects = {
        'dice_coef': dice_coef,
        'dice_loss': dice_loss,
        'iou_coef': iou_coef,
        'combined_loss': combined_loss,
    }

    try:
        model = tf.keras.models.load_model(
            str(selected_model_path),
            custom_objects=custom_objects,
            compile=False
        )
        log.success(f"Model loaded: {selected_model_path.stem}")
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        sys.exit(1)

    # ── Process nerves ────────────────────────────────────────────────────────
    config = load_config()
    seg_folder = config.get("segmented_folder", "Segmented")

    for animal, nerve, info in to_process:
        tiff_dir = info['tiff_dir']
        nerve_path = tiff_dir.parent
        output_dir = nerve_path / seg_folder
        timing_csv = output_dir / "timing.csv"

        log.rule(f"{animal} / {nerve}")
        segment_dir(
            tiff_dir=str(tiff_dir),
            output_dir=str(output_dir),
            model=model,
            mag=mag,
            log=log,
            timing_csv=str(timing_csv)
        )

    log.finalize(summary={
        "Study": study_dir,
        "Model": selected_model_path.stem,
        "Magnification": mag,
        "Nerves processed": len(to_process),
        "Nerves skipped": len(to_skip),
    })
    console.print(f"\n[dim]Log saved to: {log_path}[/dim]")


if __name__ == "__main__":
    main()
