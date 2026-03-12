"""
train/__main__.py

Entry point for: python train

Handles all user prompts, GPU setup, then calls train.train_model().
User provides only the images folder path — everything else is derived.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.console import DeepAxonLogger
from utils.gpu import setup_gpu_console
from utils.helpers import (
    get_int_input, get_float_input, get_yes_no,
    compute_batch_size, compute_aug_prob,
    get_model_dir, get_log_dir, count_patches,
    list_files
)
from train.train import train_model

console = Console()


def main():
    console.print(Panel(
        "[bold white]UNet++ Nerve Segmentation Training[/bold white]",
        title="[bold cyan]DeepAxon — Train[/bold cyan]",
        border_style="cyan",
        expand=False
    ))

    # ── GPU setup ─────────────────────────────────────────────────────────────
    setup_gpu_console()

    # ── Images folder ─────────────────────────────────────────────────────────
    images_dir = input("\nInput the path to the training images folder: ").strip().strip('"')
    if not os.path.isdir(images_dir):
        console.print(f"[red]Folder not found: {images_dir}[/red]")
        sys.exit(1)

    images_path = Path(images_dir).resolve()
    training_root = images_path.parent
    model_dir = get_model_dir(str(images_path))
    log_dir = get_log_dir(str(images_path))
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Augmentation ──────────────────────────────────────────────────────────
    use_aug = get_yes_no("Use data augmentation?", default=False)
    n_images = len(list_files(str(images_path), extensions=('.tif', '.tiff')))
    aug_prob = compute_aug_prob(n_images * 20)  # Estimate based on expected patches

    if use_aug:
        console.print(Panel(
            f"[green]Data augmentation ENABLED[/green]\n"
            f"  • Global augmentation probability: {aug_prob:.2f}\n"
            f"  • Random flips, small rotations, brightness/gamma jitter, light noise",
            expand=False
        ))

    # ── Model name ────────────────────────────────────────────────────────────
    auto_name = f"deepaxon_{timestamp}"
    console.print(f"\n[dim]Auto-generated name: {auto_name}[/dim]")
    custom = input("Enter model name (press Enter to use auto-generated): ").strip()
    model_name = custom if custom else auto_name

    # ── Training hyperparameters ──────────────────────────────────────────────
    epochs = get_int_input("Epochs (press Enter for default=200): ", default=200, min_val=1)
    val_fraction = get_float_input(
        "Validation fraction 0–1 (press Enter for default=0.1): ",
        default=0.1, min_val=0.05, max_val=0.5
    )

    # ── Batch size ────────────────────────────────────────────────────────────
    # Estimate patch count for batch size recommendation
    patches_dir = images_path / "cropped" / "patches"
    if patches_dir.exists():
        n_patches = count_patches(str(patches_dir))
    else:
        n_patches = n_images * 20  # Estimate: 20 patches per image

    recommended_bs = compute_batch_size(n_patches)
    use_recommended = get_yes_no(
        f"Use recommended batch size ({recommended_bs})?", default=True
    )
    if use_recommended:
        batch_size = recommended_bs
    else:
        batch_size = get_int_input("Enter custom batch size: ", min_val=1)

    # ── Log setup ─────────────────────────────────────────────────────────────
    log_path = str(log_dir / f"{model_name}_training_log.txt")
    log = DeepAxonLogger(log_path=log_path, program="DeepAxon Train")

    # Write config header to log
    log.log_dict({
        'Model name': model_name,
        'Architecture': 'UNet++ (DeepAxon++)',
        'Input size (H×W×C)': '256×256×1',
        'Classes': 3,
        'Patch size': 256,
        'Batch size': batch_size,
        'Epoch limit': epochs,
        'Val fraction': val_fraction,
        'Augmentations': 'H/V flip, small rotation, brightness/contrast, gamma, Gaussian noise' if use_aug else 'None',
        'ReduceLROnPlateau': 'factor=0.5, patience=15, min_delta=0.001, min_lr=1e-6',
        'EarlyStopping': 'patience=40, min_delta=0.001, restore_best_weights=True',
        'Images folder': str(images_path),
        'Models folder': str(model_dir),
        'Log file': log_path,
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
        model_type='unet++'
    )

    console.print(f"\n[dim]Log saved to: {log_path}[/dim]")


if __name__ == "__main__":
    main()
