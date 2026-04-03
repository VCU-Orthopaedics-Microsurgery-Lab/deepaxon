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

from rich.panel import Panel
from rich.console import Console

from utils.logger import DeepAxonLogger
from utils.helpers import (get_model_dir, count_patches, list_files, load_config)
from train.dataset.preprocess import batch_process
from train.dataset.data_loader import load_all_patches
from train.dataset.augment import augment_dataset_np

import torch
from torch.utils.data import DataLoader, TensorDataset
import segmentation_models_pytorch as smp

# ─── Global Config ────────────────────────────────────────────────────────────
_config    = load_config()
_train_cfg = _config.get("training", {})
_aug_cfg   = _config.get("augmentation", {})
_prob_cfg  = _aug_cfg.get("probabilities", {})

# ─── Training constants ───────────────────────────────────────────────────────
LEARNING_RATE        = _train_cfg.get("learning_rate",        1e-3)
REDUCE_LR_PATIENCE   = _train_cfg.get("reduce_lr_patience",   15)
REDUCE_LR_FACTOR     = _train_cfg.get("reduce_lr_factor",     0.5)
REDUCE_LR_MIN_LR     = _train_cfg.get("reduce_lr_min_lr",     1e-6)
EARLY_STOP_PATIENCE  = _train_cfg.get("early_stop_patience",  40)
EARLY_STOP_MIN_DELTA = _train_cfg.get("early_stop_min_delta", 0.001)
DICE_WEIGHT          = _train_cfg.get("dice_weight",          0.5)
CE_WEIGHT            = _train_cfg.get("ce_weight",            0.5)
CE_SMOOTH            = _train_cfg.get("ce_smooth",            0.1)

# ─── Augmentation constants ───────────────────────────────────────────────────
GEO_PROB   = _prob_cfg.get("geometric_prob",   0.5)
PHOTO_PROB = _prob_cfg.get("photometric_prob", 0.25)

# ─── Loss functions ───────────────────────────────────────────────────────────
_dice_loss = smp.losses.DiceLoss(mode="multiclass")
_ce_loss   = smp.losses.SoftCrossEntropyLoss(smooth_factor=CE_SMOOTH)

def loss_fn(pred, target):
    return DICE_WEIGHT * _dice_loss(pred, target) + CE_WEIGHT * _ce_loss(pred, target)


# ─── Training logger ──────────────────────────────────────────────────────────

class TrainingLogger():
    """
    Logs per-epoch metrics to DeepAxonLogger.
    Stores epoch rows for checkpoint summary at end of training.
    Checkpoint flags are passed in from train_model() where
    checkpoint logic lives — logger only handles display.
    """

    def __init__(self, log: DeepAxonLogger, use_aug: bool):
        self.log      = log
        self.use_aug  = use_aug
        self.epoch_rows = []

    def log_epoch(self, epoch: int, logs: dict, checkpoint_flag: str = ""):
        row = {
            'epoch':         epoch + 1,
            'loss':          logs.get('loss',                float('nan')),
            'dice':          logs.get('dice_coef',           float('nan')),
            'dice_axon':     logs.get('dice_coef_axon',      float('nan')),
            'dice_myelin':   logs.get('dice_coef_myelin',    float('nan')),
            'iou':           logs.get('iou_coef',            float('nan')),
            'val_loss':      logs.get('val_loss',            float('nan')),
            'val_dice':      logs.get('val_dice_coef',       float('nan')),
            'val_dice_axon': logs.get('val_dice_coef_axon',  float('nan')),
            'val_dice_myel': logs.get('val_dice_coef_myelin',float('nan')),
            'val_iou':       logs.get('val_iou_coef',        float('nan')),
            'lr':            logs.get('lr',                  float('nan')),
            'checkpoint':    checkpoint_flag,
        }
        self.epoch_rows.append(row)
        self.log.print(
            f"  Ep {row['epoch']:>4} | "
            f"loss {row['loss']:.4f} | "
            f"dice {row['dice']:.4f} axon {row['dice_axon']:.4f} myel {row['dice_myelin']:.4f} | "
            f"val_dice {row['val_dice']:.4f} val_axon {row['val_dice_axon']:.4f}"
            f"{checkpoint_flag} | lr {row['lr']:.2e}"
        )

    def on_train_end(self, checkpoint_info: dict):
        """
        Write checkpoint summary to log file only.
        Replaces the redundant epoch metrics table — per-epoch detail
        is already in the live training log lines above.

        checkpoint_info keys: epoch, combined, axon, myelin, loss, path
        """
        if not self.epoch_rows:
            return

        summary = (
            f"Best combined (axon + myelin dice) : {checkpoint_info['combined']:.4f} "
            f"@ epoch {checkpoint_info['epoch']}\n"
            f"  Val axon dice   : {checkpoint_info['axon']:.4f}\n"
            f"  Val myelin dice : {checkpoint_info['myelin']:.4f}\n"
            f"  Val loss        : {checkpoint_info['loss']:.4f}\n"
            f"  Saved to        : {checkpoint_info['path']}"
        )
        self.log.write_section("CHECKPOINT SUMMARY", summary)


# ─── Dataset preparation ──────────────────────────────────────────────────────

def prepare_dataset(images_dir: str, mag: str, log: DeepAxonLogger) -> dict:
    """
    Verify and prepare the dataset structure.
    Returns paths dict used by train_model().
    """
    images_dir = Path(images_dir).resolve() / "images"
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

    imgs  = list_files(str(images_dir), extensions=('.tif', '.tiff', '.png'))
    masks = list_files(str(masks_dir),  extensions=('.tif', '.tiff', '.png'))

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
    log: DeepAxonLogger,
    mag: str,
    model_type: str = 'unet++'
):
    """
    Full training pipeline.

    Pipeline:
        prepare_dataset → preprocess (if needed) → verify patch alignment
        → load patches → augment → build model → train loop
        → checkpoint summary → finalize → promote
    
    Checkpointing strategy:
        Primary trigger  : val_dice_axon + val_dice_myelin improves by > EARLY_STOP_MIN_DELTA
        Tiebreaker       : same combined dice but val_loss improves by > EARLY_STOP_MIN_DELTA
                           (updates checkpoint but does NOT reset patience counter)
        Early stopping   : patience counter reaches EARLY_STOP_PATIENCE epochs
                           without primary improvement
    """
    t_start = datetime.now()
    paths = prepare_dataset(images_dir, mag, log)

    # ── Preprocess if patches don't exist ────────────────────────────────────
    if not paths['patches_img'].exists() or count_patches(str(paths['patches_img'])) == 0:
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
    log.info(f"Train: {len(X_train)} patches | Val: {len(X_val)} patches | Classes: 3 (background, myelin, axon)")

    Y_train = Y_train.astype(np.int64)
    Y_val   = Y_val.astype(np.int64)

    # ── Augmentation ──────────────────────────────────────────────────────────────
    aug_count  = 0
    aug_counts = {}
    if use_aug:
        log.rule("AUGMENTATION")
        X_train, Y_train, aug_count, aug_counts = augment_dataset_np(X_train, Y_train)
        log.success(
            f"Augmented {aug_count}/{len(X_train)} patches "
            f"(geo_prob={GEO_PROB:.2f} photo_prob={PHOTO_PROB:.2f})"
        )
        log.info(f"  Geometric  — H-flip: {aug_counts['hflip']}  "
                f"V-flip: {aug_counts['vflip']}  "
                f"Rotation: {aug_counts['rotation']}"
        )
        log.info(f"  Photometric — Brightness: {aug_counts['brightness']}  "
                f"Gamma: {aug_counts['gamma']}  "
                f"Noise: {aug_counts['noise']}"
        )

    # ── Build and compile model ───────────────────────────────────────────────
    log.rule("BUILDING MODEL")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=3,
        activation=None,  # raw logits; loss handles softmax internally
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.success(f"Model built: {n_params:,} parameters")

    optimizer  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE, min_lr=REDUCE_LR_MIN_LR
    )

    # ── Training setup summary ────────────────────────────────────────────────
    log.rule("TRAINING SETUP")
    log.log_dict({
        'Architecture':   'UNet++ (DeepAxon++) — resnet34 encoder',
        'Input size':     f"256×256×1",
        'Classes':        '3 (background, myelin, axon)',
        'Device':         str(device),
        'Train patches':  len(X_train),
        'Val patches':    len(X_val),
        'Augmentation':   (
            f"ON — geo_prob={GEO_PROB:.2f} photo_prob={PHOTO_PROB:.2f}"
            if use_aug else "OFF"
        ),
        'Loss function':  f"Dice ({DICE_WEIGHT}) + SoftCE smooth={CE_SMOOTH} ({CE_WEIGHT})",
        'Optimizer':      f"Adam lr={LEARNING_RATE}",
        'ReduceLR':       f"patience={REDUCE_LR_PATIENCE}, factor={REDUCE_LR_FACTOR}, min_lr={REDUCE_LR_MIN_LR} — monitors val_loss",
        'EarlyStopping':  f"patience={EARLY_STOP_PATIENCE}, min_delta={EARLY_STOP_MIN_DELTA} — monitors axon+myelin dice",
    })
    
    # ── DataLoaders ───────────────────────────────────────────────────────────────
    # PyTorch expects (N, C, H, W) — transpose from (N, H, W, C)
    X_train_t = torch.from_numpy(X_train.transpose(0, 3, 1, 2)).float()  # (N,1,H,W)
    X_val_t   = torch.from_numpy(X_val.transpose(0, 3, 1, 2)).float()
    # Remove Channel dimension
    Y_train_t = torch.from_numpy(Y_train.squeeze(-1)).long()             # (N,H,W)
    Y_val_t   = torch.from_numpy(Y_val.squeeze(-1)).long()

    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   Y_val_t),   batch_size=batch_size, shuffle=False)

    # ── Callbacks setup ───────────────────────────────────────────────────────────
    model_dir  = get_model_dir(images_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}.pt"

    best_combined       = 0.0
    best_combined_epoch = 0
    best_combined_axon  = 0.0
    best_combined_myel  = 0.0
    best_combined_loss  = float('inf')
    epochs_no_improve   = 0
    training_logger     = TrainingLogger(log, use_aug)
    history             = {
        'loss': [], 'val_loss': [],
        'dice_coef': [], 'val_dice_coef': [],
        'iou_coef': [], 'val_iou_coef': [],
        'lr': []
    }

    # ── Training loop ─────────────────────────────────────────────────────────────
    log.rule("TRAINING")
    for epoch in range(epochs):
        
        # — Train —
        model.train()
        train_loss = 0.0
        train_tp, train_fp, train_fn, train_tn = [], [], [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss  = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            tp, fp, fn, tn = smp.metrics.get_stats(preds.argmax(dim=1), yb, mode="multiclass", num_classes=3)
            train_tp.append(tp); train_fp.append(fp); train_fn.append(fn); train_tn.append(tn)

        train_tp = torch.cat(train_tp)
        train_fp = torch.cat(train_fp)
        train_fn = torch.cat(train_fn)
        train_tn = torch.cat(train_tn)
        
        # — Validate —
        model.eval()
        val_loss = 0.0
        val_tp, val_fp, val_fn, val_tn = [], [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds  = model(xb)
                val_loss += loss_fn(preds, yb).item() * len(xb)
                tp, fp, fn, tn = smp.metrics.get_stats(preds.argmax(dim=1), yb, mode="multiclass", num_classes=3)
                val_tp.append(tp); val_fp.append(fp); val_fn.append(fn); val_tn.append(tn)
            
        val_tp = torch.cat(val_tp)
        val_fp = torch.cat(val_fp)
        val_fn = torch.cat(val_fn)
        val_tn = torch.cat(val_tn)
        
        # — Aggregate metrics —
        train_loss /= len(X_train_t)
        val_loss   /= len(X_val_t)
        train_dice  = smp.metrics.f1_score( train_tp, train_fp, train_fn, train_tn, reduction="macro").item()
        val_dice    = smp.metrics.f1_score( val_tp,   val_fp,   val_fn,   val_tn,   reduction="macro").item()
        train_iou   = smp.metrics.iou_score(train_tp, train_fp, train_fn, train_tn, reduction="macro").item()
        val_iou     = smp.metrics.iou_score(val_tp,   val_fp,   val_fn,   val_tn,   reduction="macro").item()
        current_lr  = optimizer.param_groups[0]['lr']

        # — Per-class dice (axon=1, myelin=2) —
        train_dice_axon  = smp.metrics.f1_score(train_tp, train_fp, train_fn, train_tn, reduction="none")[:, 1].mean().item()
        train_dice_myel  = smp.metrics.f1_score(train_tp, train_fp, train_fn, train_tn, reduction="none")[:, 2].mean().item()
        val_dice_axon    = smp.metrics.f1_score(val_tp,   val_fp,   val_fn,   val_tn,   reduction="none")[:, 1].mean().item()
        val_dice_myel    = smp.metrics.f1_score(val_tp,   val_fp,   val_fn,   val_tn,   reduction="none")[:, 2].mean().item()

        # — Log history —
        history['loss'].append(train_loss);         history['val_loss'].append(val_loss)
        history['dice_coef'].append(train_dice);    history['val_dice_coef'].append(val_dice)
        history['iou_coef'].append(train_iou);      history['val_iou_coef'].append(val_iou)
        history['lr'].append(current_lr)

        #— Checkpoint and early stopping ──────────────────────────────────────
        # Primary trigger: axon+myelin dice improves meaningfully
        # Tiebreaker: same combined dice but loss improves meaningfully
        #             → updates checkpoint but does NOT reset patience counter
        combined = val_dice_axon + val_dice_myel

        if combined > best_combined + EARLY_STOP_MIN_DELTA:
            best_combined       = combined
            best_combined_epoch = epoch + 1
            best_combined_axon  = val_dice_axon
            best_combined_myel  = val_dice_myel
            best_combined_loss  = val_loss
            torch.save(model.state_dict(), str(model_path))
            epochs_no_improve   = 0
            checkpoint_flag     = " ← CHECKPOINT"

        elif combined >= best_combined and val_loss < best_combined_loss - EARLY_STOP_MIN_DELTA:
            best_combined_loss  = val_loss
            best_combined_epoch = epoch + 1
            torch.save(model.state_dict(), str(model_path))
            checkpoint_flag     = " ← CHECKPOINT (loss tiebreak)"

        else:
            epochs_no_improve  += 1
            checkpoint_flag     = ""

        # — Epoch log — includes checkpoint flag ───────────────────────────────
        training_logger.log_epoch(epoch, {
            'loss':                train_loss,
            'val_loss':            val_loss,
            'dice_coef':           train_dice,
            'val_dice_coef':       val_dice,
            'dice_coef_axon':      train_dice_axon,
            'dice_coef_myelin':    train_dice_myel,
            'val_dice_coef_axon':  val_dice_axon,
            'val_dice_coef_myelin':val_dice_myel,
            'iou_coef':            train_iou,
            'val_iou_coef':        val_iou,
            'lr':                  current_lr,
        }, checkpoint_flag=checkpoint_flag)

        # — ReduceLROnPlateau —
        scheduler.step(val_loss)

        # — Early stopping —
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            log.info(
                f"Early stopping at epoch {epoch + 1} — "
                f"no improvement in axon+myelin dice for {EARLY_STOP_PATIENCE} epochs"
            )
            break
        
    # ── Load best weights and write checkpoint summary ─────────────────────────
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    training_logger.on_train_end({
        'epoch':    best_combined_epoch,
        'combined': best_combined,
        'axon':     best_combined_axon,
        'myelin':   best_combined_myel,
        'loss':     best_combined_loss,
        'path':     model_path.name,
    })

    # ── Final summary ──────────────────────────────────────────────────────────
    t_elapsed   = datetime.now() - t_start
    elapsed_str = f"{int(t_elapsed.total_seconds() // 60)}m {int(t_elapsed.total_seconds() % 60)}s"
    final       = history
    n_epochs    = len(final['loss'])
    total_patches_processed = n_epochs * len(X_train)

    Console().print(Panel(
        f"Model saved at: {model_path}\n\n"
        f"Best checkpoint epoch:    {best_combined_epoch}\n"
        f"Best val axon dice:       {best_combined_axon:.4f}\n"
        f"Best val myelin dice:     {best_combined_myel:.4f}\n"
        f"Best combined dice:       {best_combined:.4f}\n"
        f"Best val loss:            {best_combined_loss:.4f}\n\n"
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
        'Model':              str(model_path),
        'Epochs':             n_epochs,
        'Early stopped':      str(n_epochs < epochs),
        'Best checkpoint':    f"epoch {best_combined_epoch}",
        'Best axon dice':     f"{best_combined_axon:.4f}",
        'Best myelin dice':   f"{best_combined_myel:.4f}",
        'Best combined dice': f"{best_combined:.4f}",
        'Best val loss':      f"{best_combined_loss:.4f}",
        'Final train dice':   f"{final['dice_coef'][-1]:.4f}",
        'Final val dice':     f"{final['val_dice_coef'][-1]:.4f}",
        'Final train IoU':    f"{final['iou_coef'][-1]:.4f}",
        'Final val IoU':      f"{final['val_iou_coef'][-1]:.4f}",
        'Training time':      elapsed_str,
        'Patches processed':  total_patches_processed,
        'Patches augmented':  aug_count,
    })

    # ── Prompt model promotion ─────────────────────────────────────────────────
    repo_root   = Path(__file__).resolve().parent.parent
    prod_models = repo_root / "models"
    promote = input(f"\nCopy model to {prod_models}? [Y/n]: ").strip().lower()
    if promote in ('', 'y', 'yes'):
        dest = prod_models / model_path.name
        shutil.copy(str(model_path), str(dest))
        log.success(f"Model promoted to: {dest}")
    else:
        log.info(f"Model not promoted. Find it at: {model_path}")