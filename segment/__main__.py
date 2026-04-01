"""
segment/__main__.py

Entry point for: python -m segment

Walks a study directory, detects magnification, presents model selection menu,
segments all nerves. Skips nerves that already have a Segmented folder.
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.box import DOUBLE
from rich.panel import Panel
from rich.align import Align

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import DeepAxonLogger
from utils.helpers import detect_study_mag, list_models, load_config, resolve_scan
from utils.gpu import setup_gpu_console
from segment.segment import segment_dir

import torch
import segmentation_models_pytorch as smp

console = Console()


def select_model(models: list) -> Path:
    """Present an interactive model selection menu."""
    table = Table(
        box=DOUBLE,
        show_header=True,
        header_style="bold bright_cyan",
        border_style="bright_cyan",
        expand=True,
        padding=(0, 2)
    )
    table.add_column("#", style="bold cyan", width=4, justify="center")
    table.add_column("Model", style="bold white", justify="left")

    for i, m in enumerate(models, 1):
        table.add_row(str(i), m.stem)

    console.print("\n")
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
        Align.center("[bold white]Automated Axon-Myelin Brightfield Image Segmentation[/bold white]"),
        title="[bold cyan]DEEPAXON — SEGMENT[/bold cyan]",
        border_style="bright_cyan",
        box=DOUBLE,
        expand=True,
        padding=(1, 4)
    ))

    # ── Study folder input ────────────────────────────────────────────────────
    while True:
        input_dir = input("\nInput the path to the study, animal, or nerve folder: ").strip().strip('"')
        if not os.path.isdir(input_dir):
            console.print(f"[red]✗  Folder not found: {input_dir}[/red]")
            continue
        try:
            study, study_dir = resolve_scan(input_dir)
            break
        except ValueError as e:
            console.print(f"[red]✗  {e}[/red]")
            continue

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config    = load_config()
    logging_cfg = config.get("logging", {})
    logging_on  = logging_cfg.get("segment", False) if isinstance(logging_cfg, dict) else bool(logging_cfg)
    log_path    = str(Path(study_dir) / f"segment_log_{timestamp}.txt") if logging_on else None
    log       = DeepAxonLogger(log_path=log_path, program="DeepAxon Segment")

    log.info(f"Study: {study_dir}")

    # ── Scan study ────────────────────────────────────────────────────────────
    log.rule("SCANNING STUDY")

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
    to_skip    = []
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=1,
            classes=3,
            activation=None,
        )
        model.load_state_dict(torch.load(str(selected_model_path), map_location=device))
        model.to(device)
        model.eval()

        log.success(f"Model loaded: {selected_model_path.stem}")
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        sys.exit(1)

    # ── Process nerves ────────────────────────────────────────────────────────
    seg_folder = config.get("segmented_folder", "Segmented")

    for animal, nerve, info in to_process:
        tiff_dir   = info['tiff_dir']
        nerve_path = tiff_dir.parent
        output_dir = nerve_path / seg_folder
        timing_csv = str(output_dir / "timing.csv") if config.get("timing", False) else None

        log.rule(f"{animal} / {nerve}")
        segment_dir(
        tiff_dir=str(tiff_dir),
        output_dir=str(output_dir),
        model=model,
        model_path=selected_model_path,
        mag=mag,
        log=log,
        timing_csv=timing_csv
    )

    log.finalize(summary={
        "Study":             study_dir,
        "Model":             selected_model_path.stem,
        "Magnification":     mag,
        "Nerves processed":  len(to_process),
        "Nerves skipped":    len(to_skip),
    })
    console.print(f"\n[dim]Log saved to: {log_path}[/dim]")


if __name__ == "__main__":
    main()