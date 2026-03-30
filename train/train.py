"""
train/train.py

Training orchestrator for DeepAxon.
Imports exclusively from train/dataset/ and train/architectures/ — no inline duplicates.
"""

from __future__ import annotations

import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import to_categorical
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich import box

from utils.logger import DeepAxonLogger
from utils.helpers import (
    get_model_dir, compute_aug_prob,
    count_patches, list_files, load_config
)
from utils.metrics import dice_coef, dice_coef_axon, dice_coef_myelin, iou_coef, combined_loss
from train.dataset.preprocess import batch_process
from train.dataset.data_loader import load_all_patches
from train.dataset.augment import augment_dataset_np
from train.architectures import build_model


# ─── Constants ────────────────────────────────────────────────────────────────

_train_cfg           = load_config().get("training", {})
LEARNING_RATE        = _train_cfg.get("learning_rate",       1e-3)
REDUCE_LR_PATIENCE   = _train_cfg.get("reduce_lr_patience",  15)
REDUCE_LR_FACTOR     = _train_cfg.get("reduce_lr_factor",    0.5)
REDUCE_LR_MIN_LR     = _train_cfg.get("reduce_lr_min_lr",    1e-6)
EARLY_STOP_PATIENCE  = _train_cfg.get("early_stop_patience",  40)
EARLY_STOP_MIN_DELTA = _train_cfg.get("early_stop_min_delta", 0.001)


# ─── Training logger callback ─────────────────────────────────────────────────

class TrainingLogger(Callback):
    """
    Keras callback that logs per-epoch metrics to DeepAxonLogger
    and builds a summary table at the end of training.
    """

    def __init__(self, log: DeepAxonLogger, use_aug: bool, aug_prob: float):
        super().__init__()
        self.log      = log
        self.use_aug  = use_aug
        self.aug_prob = aug_prob
        self.epoch_rows = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row = {
            'epoch':    epoch + 1,
            'loss':     logs.get('loss',          float('nan')),
            'dice':     logs.get('dice_coef',     float('nan')),
            'dice_axon':     logs.get('dice_coef_axon',     float('nan')),
            'dice_myelin':   logs.get('dice_coef_myelin',   float('nan')),
            'iou':      logs.get('iou_coef',      float('nan')),
            'val_loss': logs.get('val_loss',       float('nan')),
            'val_dice': logs.get('val_dice_coef', float('nan')),
            'val_dice_axon': logs.get('val_dice_coef_axon',  float('nan')),
            'val_dice_myel': logs.get('val_dice_coef_myelin',float('nan')),
            'val_iou':  logs.get('val_iou_coef',  float('nan')),
            'lr':       logs.get('lr',             float('nan')),
        }
        self.epoch_rows.append(row)
        self.log.print(
            f"  Ep {row['epoch']:>4} | "
            f"loss {row['loss']:.4f} | "
            f"dice {row['dice']:.4f} axon {row['dice_axon']:.4f} myel {row['dice_myelin']:.4f} | "
            f"val_dice {row['val_dice']:.4f} val_axon {row['val_dice_axon']:.4f} | "
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
                f"{r['loss']:.4f}",     f"{r['dice']:.4f}",
                f"{r['iou']:.4f}",      f"{r['val_loss']:.4f}",
                f"{r['val_dice']:.4f}", f"{r['val_iou']:.4f}",
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

def prepare_dataset(images_dir: str, mag: str, log: DeepAxonLogger) -> dict:
    """
    Verify and prepare the dataset structure.
    Returns paths dict used by train_model().
    """
    images_dir = Path(images_dir).resolve()
    masks_dir  = images_dir.parent / "masks"

    cropped_img  = images_dir / "cropped"
    cropped_mask = masks_dir  / "cropped"
    patches_img  = cropped_img  / "patches"
    patches_mask = cropped_mask / "patches"
    val_img      = images_dir / "val"
    val_mask     = masks_dir  / "val"

    log.rule("VERIFYING DATASET")

    if not masks_dir.exists():
        log.warn(f"Masks folder not found: {masks_dir}")
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    imgs  = list_files(str(images_dir), extensions=('.tif', '.tiff'))
    masks = list_files(str(masks_dir),  extensions=('.tif', '.tiff'))

    img_stems  = {p.stem for p in imgs}
    mask_stems = {p.stem for p in masks}
    matched       = img_stems & mask_stems
    missing_masks = img_stems - mask_stems
    missing_imgs  = mask_stems - img_stems

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
        'images_dir':   images_dir,
        'masks_dir':    masks_dir,
        'cropped_img':  cropped_img,
        'cropped_mask': cropped_mask,
        'patches_img':  patches_img,
        'patches_mask': patches_mask,
        'val_img':      val_img,
        'val_mask':     val_mask,
        'n_pairs':      len(matched),
        'mag':          mag,
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
    mag: str,
    model_type: str = 'unet++'
):
    """
    Full training pipeline.

    Pipeline:
        prepare_dataset → preprocess (if needed) → verify patch alignment
        → load patches → one-hot encode masks → augment → build model
        → compile → callbacks → fit → summarize → prompt model promotion
    """
    t_start = datetime.now()

    paths = prepare_dataset(images_dir, mag, log)

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
            mag=mag,
            log=log
        )
        log.success(f"Patches created: {n_img} image, {n_mask} mask")
    else:
        n_img = count_patches(str(paths['patches_img']))
        log.info(f"Using existing patches: {n_img}")

    # ── Verify patch alignment ────────────────────────────────────────────────
    n_img_p  = count_patches(str(paths['patches_img']))
    n_mask_p = count_patches(str(paths['patches_mask']))
    if n_img_p != n_mask_p:
        raise ValueError(f"Patch count mismatch: {n_img_p} images vs {n_mask_p} masks")

    Console().print(Panel(
        f"✅ Patch alignment verified\nImages: {n_img_p} | Masks: {n_mask_p}",
        expand=False
    ))

    # ── Load patches ──────────────────────────────────────────────────────────
    log.rule("LOADING PATCHES")
    X_train, Y_train, X_val, Y_val = load_all_patches(
        str(paths['patches_img']),
        str(paths['patches_mask']),
        str(paths['val_img']  / 'cropped' / 'patches') if paths['val_img'].exists()  else None,
        str(paths['val_mask'] / 'cropped' / 'patches') if paths['val_mask'].exists() else None,
        val_fraction=val_fraction
    )
    log.info(f"Train: {len(X_train)} patches | Val: {len(X_val)} patches | Classes: 3")

    # ── One-hot encode masks ──────────────────────────────────────────────────
    # Masks from load_all_patches() are class indices (0/1/2).
    # to_categorical() converts to one-hot (N, H, W, 3) for softmax training.
    # Matches the original DeepAxon v1 training pipeline.
    Y_train = to_categorical(Y_train, num_classes=3).reshape(
        Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], 3
    ).astype(np.float32)
    Y_val = to_categorical(Y_val, num_classes=3).reshape(
        Y_val.shape[0], Y_val.shape[1], Y_val.shape[2], 3
    ).astype(np.float32)

    # ── Augmentation ──────────────────────────────────────────────────────────
    aug_count = 0
    if use_aug:
        log.rule("AUGMENTATION")
        # Recalculate aug_prob from actual patch count now that preprocessing is done
        aug_prob = compute_aug_prob(len(X_train))
        X_train, Y_train, aug_count = augment_dataset_np(X_train, Y_train, prob=aug_prob)
        log.success(f"Augmented {aug_count}/{len(X_train)} patches (p={aug_prob:.2f})")

    # ── Training setup summary ────────────────────────────────────────────────
    log.rule("TRAINING SETUP")
    log.log_dict({
        'Model name':    model_name,
        'Architecture':  'UNet++ (DeepAxon++)',
        'Input size':    '256×256×1',
        'Classes':       3,
        'Magnification': mag,
        'Train patches': len(X_train),
        'Val patches':   len(X_val),
        'Batch size':    batch_size,
        'Epoch limit':   epochs,
        'Augmentation':  f"ON (p={aug_prob:.2f})" if use_aug else "OFF",
        'Val fraction':  val_fraction,
        'ReduceLR':      f"patience={REDUCE_LR_PATIENCE}, factor={REDUCE_LR_FACTOR}",
        'EarlyStopping': f"patience={EARLY_STOP_PATIENCE}, min_delta={EARLY_STOP_MIN_DELTA}",
    })

    # ── Build and compile model ───────────────────────────────────────────────
    log.rule("BUILDING MODEL")
    model = build_model(model_type=model_type, input_shape=(256, 256, 1), n_classes=3)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=combined_loss,
        metrics=[dice_coef, dice_coef_axon, dice_coef_myelin, iou_coef]
    )
    log.success(f"Model built: {model.count_params():,} parameters")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    model_dir  = get_model_dir(images_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}.keras"

    callbacks = [
        TrainingLogger(log, use_aug, aug_prob),
        ReduceLROnPlateau(
            monitor='val_loss', factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA,
            min_lr=REDUCE_LR_MIN_LR, verbose=1
        ),
        EarlyStopping(
            monitor='val_loss', patience=EARLY_STOP_PATIENCE,
            min_delta=EARLY_STOP_MIN_DELTA, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
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
    t_elapsed   = datetime.now() - t_start
    elapsed_str = f"{int(t_elapsed.total_seconds() // 60)}m {int(t_elapsed.total_seconds() % 60)}s"

    final    = history.history
    n_epochs = len(final['loss'])
    total_patches_processed = n_epochs * len(X_train)

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
        'Model':             str(model_path),
        'Epochs':            n_epochs,
        'Final train loss':  f"{final['loss'][-1]:.4f}",
        'Final val loss':    f"{final['val_loss'][-1]:.4f}",
        'Final train dice':  f"{final['dice_coef'][-1]:.4f}",
        'Final val dice':    f"{final['val_dice_coef'][-1]:.4f}",
        'Final train IoU':   f"{final['iou_coef'][-1]:.4f}",
        'Final val IoU':     f"{final['val_iou_coef'][-1]:.4f}",
        'Training time':     elapsed_str,
        'Patches processed': total_patches_processed,
        'Patches augmented': aug_count,
    })

    # ── Prompt model promotion ────────────────────────────────────────────────
    repo_root   = Path(__file__).resolve().parent.parent
    prod_models = repo_root / "models"
    promote = input(f"\nCopy model to {prod_models}? [Y/n]: ").strip().lower()
    if promote in ('', 'y', 'yes'):
        dest = prod_models / model_path.name
        shutil.copy(str(model_path), str(dest))
        log.success(f"Model promoted to: {dest}")
    else:
        log.info(f"Model not promoted. Find it at: {model_path}")