"""
train/__main__.py

Entry point for: python -m train

Handles all user prompts, GPU setup, then calls train.train_model().
User provides only the images folder path — everything else is derived.

Prompt order:
    1. GPU setup
    2. Images folder
    3. Magnification
    4. Model name
    5. Epoch limit
    6. Val fraction       ← before batch size so patch estimate is accurate
    7. Augmentation
    8. Batch size menu    ← uses val_fraction for correct training patch estimate
    9. Log setup + run
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
from datetime import datetime

import torch
from rich.console import Console
from rich.panel import Panel
from rich.box import DOUBLE
from rich.align import Align

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import DeepAxonLogger
from utils.gpu import setup_gpu_console
from utils.helpers import (
    get_int_input, get_float_input, get_yes_no,
    compute_batch_options, get_model_dir, get_log_dir,
    count_patches, list_files, load_config
)
from train.train import (
    train_model,
    REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_MIN_LR,
    EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA
)

console  = Console()
has_gpu  = torch.cuda.is_available()

MAG_OPTIONS = ['40X', '100X']
PATCHES_PER_IMAGE = 63  # at 50% overlap on 1440px images


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


def _format_bs_line(bs: int, remainder: int, n_patches: int, suffix: str = "") -> str:
    """Format a single batch size line for the menu."""
    if remainder == 0:
        n_batches = n_patches // bs
        return f"bs={bs:<4} — {n_patches} patches | {n_batches} full batches (perfect fit){suffix}"
    else:
        n_full    = n_patches // bs
        fullness  = int(remainder / bs * 100)
        return (
            f"bs={bs:<4} — {n_patches} patches | "
            f"{n_full} full batches + {remainder}/{bs} remainder ({fullness}% full){suffix}"
        )


def _format_trim_line(bs: int, n_dropped: int, pct_dropped: float, n_patches: int) -> str:
    """Format a trim option line for the menu."""
    n_kept    = n_patches - n_dropped
    n_batches = n_kept // bs
    return (
        f"bs={bs:<4} — {n_kept} patches | "
        f"{n_batches} full batches ({n_dropped} patches dropped, {pct_dropped}%)"
    )


def _evaluate_custom_bs(bs: int, n_patches: int) -> tuple[str, str]:
    """
    Evaluate a custom batch size and return (status, message).
    status: 'perfect' | 'acceptable' | 'excluded' | 'danger' | 'invalid'
    """
    if bs < 1:
        return 'invalid', "Batch size must be ≥ 1."
    if n_patches < bs * 2:
        return 'invalid', f"Need at least {bs * 2} patches for 2 full batches — you have {n_patches}."

    remainder = n_patches % bs
    if remainder == 0:
        return 'perfect', _format_bs_line(bs, 0, n_patches)

    fullness = remainder / bs
    if fullness >= 0.75:
        return 'acceptable', _format_bs_line(bs, remainder, n_patches)
    elif fullness >= 0.25:
        return 'excluded', _format_bs_line(bs, remainder, n_patches)
    else:
        n_kept = n_patches - remainder
        return 'danger', _format_trim_line(bs, remainder, round(remainder/n_patches*100, 1), n_patches)


def select_batch_size(n_train_patches: int, use_gpu: bool) -> tuple[int, int, str]:
    """
    Present batch size menu and return (batch_size, actual_remainder, status).

    actual_remainder: patches in last batch after selection
                      0 if perfect fit or trim option selected
    status: 'perfect' | 'acceptable' | 'trim' | 'custom_ok' | 'custom_warn'
    """
    opts = compute_batch_options(n_train_patches, use_gpu=use_gpu)

    # ── Build menu ────────────────────────────────────────────────────────────
    ideal_str = " or ".join(str(b) for b in opts['ideal'])
    console.print("\n" + "─" * 72)
    console.print("  BATCH SIZE SELECTION")
    console.print("─" * 72)
    console.print(f"  Device                           : {opts['device_label']}")
    console.print(f"  Ideal batch size for this device : {ideal_str}")
    console.print(f"  Training patches (estimated)     : ~{n_train_patches}")
    console.print()

    menu_items  = []  # list of (label, bs, remainder, status)

    # Acceptable options
    if opts['acceptable']:
        console.print("  [green]✅ Acceptable (≥75% last batch full):[/green]")
        for bs, remainder in opts['acceptable']:
            idx   = len(menu_items) + 1
            line  = _format_bs_line(bs, remainder, n_train_patches)
            console.print(f"  [{idx}]  {line}")
            menu_items.append((idx, bs, remainder, 'perfect' if remainder == 0 else 'acceptable'))
        console.print()

    # Trim options
    if opts['trim']:
        console.print("  [cyan]✅ Trim to perfect fit (<25% last batch — drops remainder):[/cyan]")
        for bs, n_dropped, pct_dropped in opts['trim']:
            idx  = len(menu_items) + 1
            line = _format_trim_line(bs, n_dropped, pct_dropped, n_train_patches)
            console.print(f"  [{idx}]  {line}")
            menu_items.append((idx, bs, 0, 'trim'))
        console.print()

    # Excluded — shown as info only, not selectable
    if opts['excluded']:
        excl_str = ", ".join(f"bs={bs} ({pct}%)" for bs, _, pct in opts['excluded'])
        console.print(f"  [yellow]⚠  Excluded (25–75% last batch): {excl_str}[/yellow]")
        console.print()

    # Custom entry always last
    custom_idx = len(menu_items) + 1
    console.print(f"  [{custom_idx}]  Enter custom batch size")
    console.print("─" * 72)

    # ── Selection loop ────────────────────────────────────────────────────────
    while True:
        raw = input(f"Select [1-{custom_idx}]: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            console.print("[red]Please enter a number.[/red]")
            continue

        if choice == custom_idx:
            # Custom entry
            while True:
                try:
                    custom_bs = int(input("  Enter custom batch size: ").strip())
                except ValueError:
                    console.print("[red]Please enter a valid integer.[/red]")
                    continue

                status, msg = _evaluate_custom_bs(custom_bs, n_train_patches)

                if status == 'invalid':
                    console.print(f"[red]  ✗ {msg}[/red]")
                    continue
                elif status == 'perfect':
                    console.print(f"[green]  ✓ Perfect fit: {msg}[/green]")
                    return custom_bs, 0, 'custom_ok'
                elif status == 'acceptable':
                    console.print(f"[green]  ✓ Acceptable: {msg}[/green]")
                    remainder = n_train_patches % custom_bs
                    return custom_bs, remainder, 'custom_ok'
                elif status == 'excluded':
                    remainder = n_train_patches % custom_bs
                    fullness  = int(remainder / custom_bs * 100)
                    console.print(
                        f"[yellow]  ⚠ Last batch {fullness}% full (25–75% excluded zone). "
                        f"Strongly recommend choosing a different size.[/yellow]"
                    )
                    if get_yes_no("  Proceed anyway?", default=False):
                        return custom_bs, remainder, 'custom_warn'
                elif status == 'danger':
                    n_dropped = n_train_patches % custom_bs
                    pct       = round(n_dropped / n_train_patches * 100, 1)
                    console.print(
                        f"[red]  ⚠ Last batch <25% full. "
                        f"{n_dropped} patches ({pct}%) will be dropped.[/red]"
                    )
                    if get_yes_no("  Proceed with trim?", default=False):
                        return custom_bs, 0, 'custom_warn'

        elif 1 <= choice <= len(menu_items):
            _, bs, remainder, status = menu_items[choice - 1]
            return bs, remainder, status

        else:
            console.print(f"[red]Please enter a number between 1 and {custom_idx}.[/red]")


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
    images_sub  = images_path / "images"
    model_dir   = get_model_dir(str(images_path))
    log_dir     = get_log_dir(str(images_path))
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Load config ───────────────────────────────────────────────────────────
    config    = load_config()
    train_cfg = config.get("training", {})
    patch_cfg = config.get("patch_size", {})
    aug_cfg   = config.get("augmentation", {})
    prob_cfg  = aug_cfg.get("probabilities", {})
    param_cfg = aug_cfg.get("parameters", {})

    geo_prob   = prob_cfg.get("geometric_prob",  0.5)
    photo_prob = prob_cfg.get("photometric_prob", 0.25)

    # ── Magnification ─────────────────────────────────────────────────────────
    mag = select_magnification()

    # ── Model name ────────────────────────────────────────────────────────────
    auto_name  = f"deepaxon_{mag.lower()}_{timestamp}"
    console.print(f"\n[dim]Auto-generated name: {auto_name}[/dim]")
    custom     = input("Enter model name (press Enter to use auto-generated): ").strip()
    model_name = custom if custom else auto_name

    # ── Epoch limit ───────────────────────────────────────────────────────────
    default_epochs = train_cfg.get("epochs", 200)
    epochs = get_int_input(
        f"Epochs (press Enter for default={default_epochs}): ",
        default=default_epochs, min_val=1
    )

    # ── Val fraction — must come before batch size ────────────────────────────
    val_fraction = get_float_input(
        "Validation fraction 0–1 (press Enter for default=0.2): ",
        default=0.2, min_val=0.05, max_val=0.5
    )

    # ── Augmentation ──────────────────────────────────────────────────────────
    use_aug = get_yes_no("Use data augmentation?", default=False)
    if use_aug:
        console.print(Panel(
            Align.center(
                f"[green]Data augmentation ENABLED[/green]\n"
                f"Geometric prob: {geo_prob:.2f} | Photometric prob: {photo_prob:.2f}\n"
                f"Flips, rotation ±{param_cfg.get('rotation_deg', 15)}°, "
                f"brightness {param_cfg.get('brightness_range', [0.8, 1.2])}, "
                f"gamma {param_cfg.get('gamma_range', [0.7, 1.4])}, "
                f"noise σ={param_cfg.get('noise_sigma', 0.02)}"
            ),
            border_style="green",
            box=DOUBLE,
            expand=True
        ))

    # ── Batch size menu ───────────────────────────────────────────────────────
    # Estimate training patches using val_fraction so menu reflects
    # what will actually be trained on after the split.
    patches_dir = images_sub / "cropped" / "patches"
    if patches_dir.exists():
        total_patches   = count_patches(str(patches_dir))
        n_train_patches = int(total_patches * (1 - val_fraction))
    else:
        n_images        = len(list_files(str(images_sub), extensions=('.tif', '.tiff', '.png', '.bmp')))
        n_train_patches = int(n_images * (1 - val_fraction)) * PATCHES_PER_IMAGE

    batch_size, remainder, bs_status = select_batch_size(n_train_patches, use_gpu=has_gpu)

    # ── Log setup ─────────────────────────────────────────────────────────────
    log_path = str(log_dir / f"{model_name}_training_log.txt")
    log      = DeepAxonLogger(log_path=log_path, program="DeepAxon Train")

    # Batch size description for log — reflects actual math
    if remainder == 0:
        bs_log = f"{batch_size} (perfect fit)"
    else:
        bs_log = (
            f"{batch_size} "
            f"(last batch {remainder}/{batch_size} = {int(remainder/batch_size*100)}% full)"
        )
    if bs_status == 'custom_warn':
        bs_log += " [user override — outside recommended zones]"

    log.log_dict({
        'Model name':       model_name,
        'Magnification':    mag,
        'Epoch limit':      epochs,
        'Val fraction':     val_fraction,
        'Augmentation':     'ON' if use_aug else 'OFF',
        'Batch size':       bs_log,
        'Est. train patches': f"~{n_train_patches} (actual logged after load)",
        'Images folder':    str(images_path),
        'Models folder':    str(model_dir),
        'Log file':         log_path,
    })

    # ── Run training ──────────────────────────────────────────────────────────
    train_model(
        images_dir=str(images_path),
        model_name=model_name,
        epochs=epochs,
        val_fraction=val_fraction,
        batch_size=batch_size,
        use_aug=use_aug,
        log=log,
        mag=mag,
    )

    console.print(f"\n[dim]Log saved to: {log_path}[/dim]")


if __name__ == "__main__":
    main()