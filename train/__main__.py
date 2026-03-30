"""
train/__main__.py

Entry point for: python -m train

Handles all user prompts, GPU setup, then calls train.train_model().
User provides only the images folder path — everything else is derived.
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.box import DOUBLE
from rich.align import Align

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import DeepAxonLogger
from utils.gpu import setup_gpu_console
from utils.helpers import (
    get_int_input, get_float_input, get_yes_no,
    compute_batch_size, compute_aug_prob,
    get_model_dir, get_log_dir, count_patches,
    list_files, load_config
)
from train.train import (
    train_model,
    REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_MIN_LR,
    EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA
)

console = Console()

MAG_OPTIONS = ['40X', '100X']


def select_magnification() -> str:
    """Prompt user to select imaging magnification."""
    console.print("\n[bold]Select imaging magnification:[/bold]")
    for i, m in enumerate(MAG_OPTIONS, 1):
        console.print(f"  [{i}] {m}")
    while True:
        raw = input(f"Select [1-{len(MAG_OPTIONS)}]: ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(MAG_OPTIONS):
                return MAG_OPTIONS[idx]
        except ValueError:
            pass
        console.print(f"[red]Please enter a number between 1 and {len(MAG_OPTIONS)}[/red]")


def main():
    console.print(Panel(
        Align.center("[bold white]UNet++ Nerve Segmentation\nModel Training[/bold white]"),
        title="[bold cyan]DEEPAXON — TRAIN[/bold cyan]",
        border_style="bright_cyan",
        box=DOUBLE,
        expand=True,
        padding=(1, 4)
    ))

    # ── GPU setup ─────────────────────────────────────────────────────────────
    setup_gpu_console()

    # ── Images folder ─────────────────────────────────────────────────────────
    images_dir = input("\nInput the path to the training images folder: ").strip().strip('"')
    if not os.path.isdir(images_dir):
        console.print(f"[red]Folder not found: {images_dir}[/red]")
        sys.exit(1)

    images_path = Path(images_dir).resolve()
    model_dir   = get_model_dir(str(images_path))
    log_dir     = get_log_dir(str(images_path))
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Load config ───────────────────────────────────────────────────────────
    config    = load_config()
    train_cfg = config.get("training", {})
    patch_cfg = config.get("patch_size", {})
    aug_cfg   = config.get("augmentation", {})

    # ── Magnification ─────────────────────────────────────────────────────────
    mag = select_magnification()

    # ── Augmentation ──────────────────────────────────────────────────────────
    use_aug  = get_yes_no("Use data augmentation?", default=False)
    n_images = len(list_files(str(images_path), extensions=('.tif', '.tiff')))
    aug_prob = compute_aug_prob(n_images * 20)  # Estimate — recalculated in train.py after preprocessing

    if use_aug:
        console.print(Panel(
            Align.center(
                f"[green]Data augmentation ENABLED[/green]\n"
                f"Estimated probability: {aug_prob:.2f}  |  "
                f"Flips, rotation ±{aug_cfg.get('rotation_deg', 15)}°, "
                f"brightness {aug_cfg.get('brightness_range', [0.8, 1.2])}, "
                f"gamma {aug_cfg.get('gamma_range', [0.7, 1.4])}, "
                f"noise σ={aug_cfg.get('noise_sigma', 0.02)}"
            ),
            border_style="green",
            box=DOUBLE,
            expand=True
        ))

    # ── Model name ────────────────────────────────────────────────────────────
    auto_name  = f"deepaxon_{mag.lower()}_{timestamp}"
    console.print(f"\n[dim]Auto-generated name: {auto_name}[/dim]")
    custom     = input("Enter model name (press Enter to use auto-generated): ").strip()
    model_name = custom if custom else auto_name

    # ── Training hyperparameters ──────────────────────────────────────────────
    default_epochs = train_cfg.get("epochs", 200)
    epochs = get_int_input(
        f"Epochs (press Enter for default={default_epochs}): ",
        default=default_epochs, min_val=1
    )
    val_fraction = get_float_input(
        "Validation fraction 0–1 (press Enter for default=0.1): ",
        default=0.1, min_val=0.05, max_val=0.5
    )

    # ── Batch size ────────────────────────────────────────────────────────────
    patches_dir = images_path / "cropped" / "patches"
    if patches_dir.exists():
        n_patches = count_patches(str(patches_dir))
    else:
        n_patches = n_images * 20  # Estimate: ~20 patches per image at 50% overlap

    recommended_bs = compute_batch_size(n_patches)
    use_recommended = get_yes_no(
        f"Use recommended batch size ({recommended_bs})?", default=True
    )
    batch_size = recommended_bs if use_recommended else get_int_input(
        "Enter custom batch size: ", min_val=1
    )

    # ── Log setup ─────────────────────────────────────────────────────────────
    # Training always logs — config logging.train flag reserved for future verbosity control
    log_path = str(log_dir / f"{model_name}_training_log.txt")
    log      = DeepAxonLogger(log_path=log_path, program="DeepAxon Train")

    patch_size = patch_cfg.get(mag, 256)

    log.log_dict({
        'Model name':        model_name,
        'Architecture':      'UNet++ (DeepAxon++)',
        'Magnification':     mag,
        'Input size':        f"{patch_size}×{patch_size}×1",
        'Classes':           3,
        'Patch size':        patch_size,
        'Batch size':        batch_size,
        'Epoch limit':       epochs,
        'Val fraction':      val_fraction,
        'Augmentations':     (
            f"H/V flip, rotation ±{aug_cfg.get('rotation_deg', 15)}°, "
            f"brightness {aug_cfg.get('brightness_range', [0.8, 1.2])}, "
            f"gamma {aug_cfg.get('gamma_range', [0.7, 1.4])}, "
            f"noise σ={aug_cfg.get('noise_sigma', 0.02)}"
            if use_aug else 'None'
        ),
        'ReduceLROnPlateau': f"factor={REDUCE_LR_FACTOR}, patience={REDUCE_LR_PATIENCE}, min_lr={REDUCE_LR_MIN_LR}",
        'EarlyStopping':     f"patience={EARLY_STOP_PATIENCE}, min_delta={EARLY_STOP_MIN_DELTA}",
        'Images folder':     str(images_path),
        'Models folder':     str(model_dir),
        'Log file':          log_path,
    })

    # ── Run training ──────────────────────────────────────────────────────────
    train_model(
        images_dir=str(images_path),
        model_name=model_name,
        epochs=epochs,
        val_fraction=val_fraction,
        batch_size=batch_size,
        use_aug=use_aug,
        aug_prob=aug_prob,
        log=log,
        mag=mag,
    )

    console.print(f"\n[dim]Log saved to: {log_path}[/dim]")


if __name__ == "__main__":
    main()