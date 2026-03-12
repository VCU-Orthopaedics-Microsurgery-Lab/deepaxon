"""
train/train.py

Training orchestrator for DeepAxon.
Imports exclusively from train/data/ and train/models/ — no inline duplicates.
"""

from __future__ import annotations

import os
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, Callback
)
from rich.table import Table
from rich import box

from utils.console import DeepAxonLogger
from utils.metrics import dice_coef, dice_loss, iou_coef, combined_loss
from utils.helpers import (
    get_model_dir, get_log_dir, compute_batch_size, compute_aug_prob,
    count_patches
)
from train.data.preprocess import batch_process
from train.data.data_loader import load_all_patches
from train.data.augment import augment_dataset_np
from train.models.unet_plus_plus import build_model


# ─── Training logger callback ─────────────────────────────────────────────────

class TrainingLogger(Callback):
    """
    Keras callback that logs per-epoch metrics to DeepAxonLogger
    and builds a summary table at the end of training.
    """

    def __init__(self, log: DeepAxonLogger, use_aug: bool, aug_prob: float):
        super().__init__()
        self.log = log
        self.use_aug = use_aug
        self.aug_prob = aug_prob
        self.epoch_rows = []
        self.aug_counts = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row = {
            'epoch': epoch + 1,
            'loss': logs.get('loss', float('nan')),
            'dice': logs.get('dice_coef', float('nan')),
            'iou': logs.get('iou_coef', float('nan')),
            'val_loss': logs.get('val_loss', float('nan')),
            'val_dice': logs.get('val_dice_coef', float('nan')),
            'val_iou': logs.get('val_iou_coef', float('nan')),
            'lr': logs.get('lr', float('nan')),
        }
        self.epoch_rows.append(row)

        # Console progress line
        self.log.print(
            f"  Ep {row['epoch']:>4} | "
            f"loss {row['loss']:.4f} dice {row['dice']:.4f} iou {row['iou']:.4f} | "
            f"val_loss {row['val_loss']:.4f} val_dice {row['val_dice']:.4f} | "
            f"lr {row['lr']:.2e}"
        )

    def on_train_end(self, logs=None):
        if not self.epoch_rows:
            return

        table = Table(
            title="DeepAxon++ Training Progress",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold cyan"
        )
        for col in ['Epoch', 'Loss', 'Dice', 'IoU', 'Val Loss', 'Val Dice', 'Val IoU', 'LR']:
            table.add_column(col, justify='right')

        for r in self.epoch_rows:
            table.add_row(
                str(r['epoch']),
                f"{r['loss']:.4f}",
                f"{r['dice']:.4f}",
                f"{r['iou']:.4f}",
                f"{r['val_loss']:.4f}",
                f"{r['val_dice']:.4f}",
                f"{r['val_iou']:.4f}",
                f"{r['lr']:.2e}",
            )

        self.log.console.print(table)
        self.log.write_section("EPOCH METRICS", self._plain_table())

    def _plain_table(self) -> str:
        lines = ["Epoch | Loss   | Dice   | IoU    | Val Loss | Val Dice | Val IoU  | LR"]
        lines.append("-" * 75)
        for r in self.epoch_rows:
            lines.append(
                f"{r['epoch']:>5} | {r['loss']:>6.4f} | {r['dice']:>6.4f} | "
                f"{r['iou']:>6.4f} | {r['val_loss']:>8.4f} | {r['val_dice']:>8.4f} | "
                f"{r['val_iou']:>8.4f} | {r['lr']:.2e}"
            )
        return '\n'.join(lines)


# ─── Dataset preparation ──────────────────────────────────────────────────────

def prepare_dataset(images_dir: str, log: DeepAxonLogger) -> dict:
    """
    Verify and prepare the dataset structure.
    Creates patches/ and val/ directories if needed.
    Returns paths dict.
    """
    images_dir = Path(images_dir).resolve()
    masks_dir = images_dir.parent / "masks"
    patches_img = images_dir / "cropped" / "patches"
    patches_mask = masks_dir / "cropped" / "patches"
    val_img = images_dir / "val"
    val_mask = masks_dir / "val"

    log.rule("VERIFYING DATASET")

    # Check masks folder
    if not masks_dir.exists():
        log.warn(f"Masks folder not found: {masks_dir}")
        log.warn("Please add your mask images to that folder and re-run.")
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    # Verify image/mask pairs
    from utils.helpers import list_files
    imgs = list_files(str(images_dir), extensions=('.tif', '.tiff'))
    masks = list_files(str(masks_dir), extensions=('.tif', '.tiff'))

    img_stems = {p.stem for p in imgs}
    mask_stems = {p.stem for p in masks}
    matched = img_stems & mask_stems
    missing_masks = img_stems - mask_stems
    missing_imgs = mask_stems - img_stems

    from rich.panel import Panel
    from rich.console import Console
    Console().print(Panel(
        f"Found pairs:    {len(matched)}\n"
        f"Missing masks:  {len(missing_masks)}\n"
        f"Missing images: {len(missing_imgs)}",
        title="Dataset Verification",
        expand=False
    ))

    if missing_masks:
        log.warn(f"Images without masks: {sorted(missing_masks)}")
    if missing_imgs:
        log.warn(f"Masks without images: {sorted(missing_imgs)}")

    return {
        'images_dir': images_dir,
        'masks_dir': masks_dir,
        'cropped_img': cropped_img,
        'cropped_mask': cropped_mask,
        'patches_img': patches_img,
        'patches_mask': patches_mask,
        'val_img': val_img,
        'val_mask': val_mask,
        'n_pairs': len(matched),
    }


# ─── Main training function ───────────────────────────────────────────────────

def train_model(
    images_dir: str,
    model_name: str,
    epochs: int,
    val_fraction: float,
    batch_size: int,
    use_aug: bool,
    aug_prob: float,
    log: DeepAxonLogger,
    model_type: str = 'unet++'
):
    """
    Full training pipeline.
    """
    t_start = datetime.now()

    paths = prepare_dataset(images_dir, log)

    # ── Preprocess if patches don't exist ────────────────────────────────────
    if not paths['patches_img'].exists() or count_patches(str(paths['patches_img'])) == 0:
        log.rule("PREPROCESSING")
        n_img, n_mask = batch_process(
            str(paths['images_dir']),
            str(paths['masks_dir']),
            str(paths['patches_img']),
            str(paths['patches_mask']),
            str(paths['cropped_img']),
            str(paths['cropped_mask']),
            mag=model_type,
            log=log
        )
        log.success(f"Patches created: {n_img} image, {n_mask} mask")
    else:
        n_img = count_patches(str(paths['patches_img']))
        log.info(f"Using existing patches: {n_img}")

    # ── Verify patch alignment ────────────────────────────────────────────────
    n_img_p = count_patches(str(paths['patches_img']))
    n_mask_p = count_patches(str(paths['patches_mask']))
    if n_img_p != n_mask_p:
        raise ValueError(f"Patch count mismatch: {n_img_p} images vs {n_mask_p} masks")

    from rich.panel import Panel
    from rich.console import Console
    Console().print(Panel(
        f"✅ Patch alignment verified\nImages: {n_img_p} | Masks: {n_mask_p}",
        expand=False
    ))

    # ── Load patches ─────────────────────────────────────────────────────────
    log.rule("LOADING PATCHES")
    X_train, Y_train, X_val, Y_val = load_all_patches(
        str(paths['patches_img']),
        str(paths['patches_mask']),
        str(paths['val_img'] / 'cropped' / 'patches') if paths['val_img'].exists() else None,
        str(paths['val_mask'] / 'cropped' / 'patches') if paths['val_mask'].exists() else None,
        val_fraction=val_fraction
    )

    log.info(f"Train: {len(X_train)} patches | Val: {len(X_val)} patches | Classes: 3")

    # ── Augmentation ─────────────────────────────────────────────────────────
    aug_count = 0
    if use_aug:
        log.rule("AUGMENTATION")
        X_train, Y_train, aug_count = augment_dataset_np(X_train, Y_train, prob=aug_prob)
        log.success(f"Augmented {aug_count}/{len(X_train)} patches (p={aug_prob:.2f})")

    # ── Training setup summary ────────────────────────────────────────────────
    log.rule("TRAINING SETUP")
    log.log_dict({
        'Model name': model_name,
        'Architecture': 'UNet++ (DeepAxon++)',
        'Input size': '256×256×1',
        'Classes': 3,
        'Train patches': len(X_train),
        'Val patches': len(X_val),
        'Batch size': batch_size,
        'Epoch limit': epochs,
        'Augmentation': f"ON (p={aug_prob:.2f})" if use_aug else "OFF",
        'Val fraction': val_fraction,
    })

    # ── Build and compile model ───────────────────────────────────────────────
    log.rule("BUILDING MODEL")
    model = build_model(model_type=model_type, input_shape=(256, 256, 1), n_classes=3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=combined_loss,
        metrics=[dice_coef, iou_coef]
    )
    log.success(f"Model built: {model.count_params():,} parameters")

    # ── Callbacks ────────────────────────────────────────────────────────────
    model_dir = get_model_dir(images_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}.keras"

    training_logger = TrainingLogger(log, use_aug, aug_prob)
    callbacks = [
        training_logger,
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=15,
            min_delta=0.001, min_lr=1e-6, verbose=1
        ),
        EarlyStopping(
            monitor='val_loss', patience=40,
            min_delta=0.001, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path), monitor='val_loss',
            save_best_only=True, verbose=0
        ),
    ]

    # ── Training ──────────────────────────────────────────────────────────────
    log.rule("TRAINING")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0  # Handled by TrainingLogger
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    t_elapsed = datetime.now() - t_start
    elapsed_str = f"{int(t_elapsed.total_seconds() // 60)}m {int(t_elapsed.total_seconds() % 60)}s"

    final = history.history
    n_epochs = len(final['loss'])
    total_patches_processed = n_epochs * len(X_train)

    from rich.panel import Panel
    from rich.console import Console
    Console().print(Panel(
        f"Model saved at: {model_path}\n\n"
        f"Final Training Loss:      {final['loss'][-1]:.4f}\n"
        f"Final Validation Loss:    {final['val_loss'][-1]:.4f}\n"
        f"Final Training Dice:      {final['dice_coef'][-1]:.4f}\n"
        f"Final Validation Dice:    {final['val_dice_coef'][-1]:.4f}\n"
        f"Final Training IoU:       {final['iou_coef'][-1]:.4f}\n"
        f"Final Validation IoU:     {final['val_iou_coef'][-1]:.4f}\n\n"
        f"Total Training Time:      {elapsed_str}\n"
        f"Epochs completed:         {n_epochs}\n"
        f"Total patches processed:  {total_patches_processed:,}\n"
        f"Patches augmented:        {aug_count:,}",
        title="[bold green]DeepAxon++ Training Complete[/bold green]",
        border_style="green",
        expand=False
    ))

    log.finalize(summary={
        'Model': str(model_path),
        'Epochs': n_epochs,
        'Final train loss': f"{final['loss'][-1]:.4f}",
        'Final val loss': f"{final['val_loss'][-1]:.4f}",
        'Final train dice': f"{final['dice_coef'][-1]:.4f}",
        'Final val dice': f"{final['val_dice_coef'][-1]:.4f}",
        'Final train IoU': f"{final['iou_coef'][-1]:.4f}",
        'Final val IoU': f"{final['val_iou_coef'][-1]:.4f}",
        'Training time': elapsed_str,
        'Patches processed': total_patches_processed,
        'Patches augmented': aug_count,
    })
