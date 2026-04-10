"""
train/train.py

Training orchestrator for DeepAxon.
Imports exclusively from train/dataset/ and train/architectures/ — no inline duplicates.
"""

from __future__ import annotations

import sys
import socket
import numpy as np
from pathlib import Path
from datetime import datetime
import cv2

from rich.panel import Panel
from rich.console import Console

from utils.logger import DeepAxonLogger
from utils.version import __version__, __codename__
from utils.helpers import get_model_dir, count_patches, load_config
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

# ─── Augmentation constants ───────────────────────────────────────────────────
GEO_PROB   = _prob_cfg.get("geometric_prob",   0.5)
PHOTO_PROB = _prob_cfg.get("photometric_prob", 0.25)

# ─── Class weights ────────────────────────────────────────────────────────────
_class_weights_cfg = _train_cfg.get("class_weights", [1.5, 1.0, 1.2])


def weighted_dice_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Weighted multiclass Dice loss.
    pred:    (N, C, H, W) raw logits
    target:  (N, H, W)   class indices
    weights: (C,)         per-class weights
    """
    num_classes = pred.shape[1]
    pred_soft   = torch.softmax(pred, dim=1)
    target_one_hot = torch.zeros_like(pred_soft).scatter_(
        1, target.unsqueeze(1), 1.0
    )
    dims   = (0, 2, 3)
    inter  = (pred_soft * target_one_hot).sum(dims)
    union  = (pred_soft + target_one_hot).sum(dims)
    dice   = (2.0 * inter + 1e-6) / (union + 1e-6)
    loss   = 1.0 - dice
    return (weights * loss).sum() / weights.sum()

# ─── Training logger ──────────────────────────────────────────────────────────

class TrainingLogger():
    """
    Logs per-epoch metrics to DeepAxonLogger.
    Stores epoch rows for checkpoint summary at end of training.
    Checkpoint flags are passed in from train_model() where
    checkpoint logic lives — logger only handles display.
    """

    def __init__(self, log: DeepAxonLogger, use_aug: bool):
        self.log        = log
        self.use_aug    = use_aug
        self.epoch_rows = []

    def log_epoch(self, epoch: int, logs: dict, checkpoint_flag: str = ""):
        row = {
            'epoch':         epoch + 1,
            'epoch_time':    logs.get('epoch_time', ''),
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
            f"  Ep {row['epoch']:>4} ({row['epoch_time']}) | "
            f"loss {row['loss']:.3f} | "
            f"dice {row['dice']:.3f} axon {row['dice_axon']:.3f} myel {row['dice_myelin']:.3f} | "
            f"val_dice {row['val_dice']:.3f} vax {row['val_dice_axon']:.3f} vmy {row['val_dice_myel']:.3f}"
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
    images_dir   = Path(images_dir).resolve() / "images"
    masks_dir    = images_dir.parent / "masks"

    cropped_img  = images_dir / "cropped"
    cropped_mask = masks_dir  / "cropped"
    patches_img  = cropped_img  / "patches"
    patches_mask = cropped_mask / "patches"

    log.rule("VERIFYING DATASET")

    if not masks_dir.exists():
        log.warn(f"Masks folder not found: {masks_dir}")
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    imgs  = list(images_dir.rglob('*.tif')) + list(images_dir.rglob('*.tiff')) + list(images_dir.rglob('*.png'))
    masks = list(masks_dir.rglob('*.tif'))  + list(masks_dir.rglob('*.tiff'))  + list(masks_dir.rglob('*.png'))

    img_stems     = {p.stem for p in imgs}
    mask_stems    = {p.stem for p in masks}
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
    log.info(f"Found pairs: {len(matched)} | Missing masks: {len(missing_masks)} | Missing images: {len(missing_imgs)}")

    if missing_masks:
        log.warn(f"Images without masks: {sorted(missing_masks)}")
    if missing_imgs:
        log.warn(f"Masks without images: {sorted(missing_imgs)}")

    # ── Mask quality check ────────────────────────────────────────────────────────
    log.rule("MASK QUALITY CHECK")
    for mask_path in sorted(masks_dir.glob('*.png')) + sorted(masks_dir.glob('*.tif')):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        unexpected = ~np.isin(mask, [0, 127, 128, 255])
        if unexpected.sum() > 0:
            pct = round(unexpected.sum() / mask.size * 100, 2)
            log.warn(f"  {mask_path.name}: {unexpected.sum()} unexpected pixels ({pct}%) — will be thresholded to nearest class")
        else:
            log.info(f"  {mask_path.name}: pixels clean")
        
    return {
        'images_dir':   images_dir,
        'masks_dir':    masks_dir,
        'cropped_img':  cropped_img,
        'cropped_mask': cropped_mask,
        'patches_img':  patches_img,
        'patches_mask': patches_mask,
        'n_pairs':      len(matched),
        'mag':          mag,
    }
  

# ─── Main training function ───────────────────────────────────────────────────

def train_model(
    images_dir: str,
    model_name: str,
    epochs: int,
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
    paths   = prepare_dataset(images_dir, mag, log)

    # ── Preprocess if patches don't exist ─────────────────────────────────────
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

    # ── Verify patch alignment ─────────────────────────────────────────────────
    n_img_p  = count_patches(str(paths['patches_img']))
    n_mask_p = count_patches(str(paths['patches_mask']))
    if n_img_p != n_mask_p:
        raise ValueError(f"Patch count mismatch: {n_img_p} images vs {n_mask_p} masks")

    Console().print(Panel(
        f"✅ Patch alignment verified\nImages: {n_img_p} | Masks: {n_mask_p}",
        expand=False
    ))

    # ── Load training directories ─────────────────────────────────────────────────
    log.rule("LOADING TRAINING DIRECTORIES")
    X_train, Y_train, X_val, Y_val, split_mode, val_stems= load_all_patches(
        str(paths['images_dir']),
        str(paths['masks_dir']),
        log=log,
    )
    log.info(f"Train: {len(X_train)} patches | Val: {len(X_val)} patches | Classes: 3 (background, myelin, axon)")

    Y_train = Y_train.astype(np.int64)
    Y_val   = Y_val.astype(np.int64)

    # ── Augmentation ───────────────────────────────────────────────────────────
    aug_count  = 0
    aug_counts = {}
    if use_aug:
        log.rule("AUGMENTATION")
        X_train, Y_train, aug_count, aug_counts = augment_dataset_np(X_train, Y_train)
        log.success(
            f"Augmented {aug_count}/{len(X_train)} patches "
            f"(geo_prob={GEO_PROB:.2f} photo_prob={PHOTO_PROB:.2f})"
        )
        log.info(
            f"  Geometric  — H-flip: {aug_counts['hflip']}  "
            f"V-flip: {aug_counts['vflip']}  "
            f"Rotation: {aug_counts['rotation']}"
        )
        log.info(
            f"  Photometric — Brightness: {aug_counts['brightness']}  "
            f"Gamma: {aug_counts['gamma']}  "
            f"Noise: {aug_counts['noise']}"
        )
        aug_pct = round(aug_count / len(X_train) * 100, 1)
        log.info(f"  Effective: {aug_count}/{len(X_train)} patches modified ({aug_pct}%)")

    # ── Build model ────────────────────────────────────────────────────────────
    log.rule("BUILDING MODEL")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    
    CLASS_WEIGHTS = torch.tensor(_class_weights_cfg, dtype=torch.float32).to(device)
    _ce_loss      = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

    def loss_fn(pred, target):
        dice = weighted_dice_loss(pred, target, CLASS_WEIGHTS)
        ce   = _ce_loss(pred, target)
        return DICE_WEIGHT * dice + CE_WEIGHT * ce

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=3,
        activation=None,  # raw logits — loss handles softmax internally
    )
    model     = model.to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    log.success(f"Model built: {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE, min_lr=REDUCE_LR_MIN_LR
    )
    

    # ── Training setup summary ─────────────────────────────────────────────────
    log.rule("TRAINING SETUP")
    log.log_dict({
        'Architecture':     'UNet++ (DeepAxon++) — resnet34 encoder',
        'Input size':       '256×256×1',
        'Classes':          '3 (background, myelin, axon)',
        'Class weights':    f"bg={_class_weights_cfg[0]} myelin={_class_weights_cfg[1]} axon={_class_weights_cfg[2]}",
        'Device':           str(device),
        'Train patches':    len(X_train),
        'Val patches':      len(X_val),
        'Test/Train split':  split_mode,
        'Augmentation':  (
            f"ON — geo_prob={GEO_PROB:.2f} photo_prob={PHOTO_PROB:.2f}"
            if use_aug else "OFF"
        ),
        'Loss function': f"Weighted Dice ({DICE_WEIGHT}) + CrossEntropy ({CE_WEIGHT})",
        'Optimizer':     f"Adam lr={LEARNING_RATE}",
        'ReduceLR':      f"patience={REDUCE_LR_PATIENCE}, factor={REDUCE_LR_FACTOR}, min_lr={REDUCE_LR_MIN_LR} — monitors val_loss",
        'EarlyStopping': f"patience={EARLY_STOP_PATIENCE}, min_delta={EARLY_STOP_MIN_DELTA} — monitors axon+myelin dice",
    })

    # ── DataLoaders ────────────────────────────────────────────────────────────
    # PyTorch expects (N, C, H, W) — transpose from (N, H, W, C)
    X_train_t = torch.from_numpy(X_train.transpose(0, 3, 1, 2)).float()  # (N,1,H,W)
    X_val_t   = torch.from_numpy(X_val.transpose(0, 3, 1, 2)).float()
    Y_train_t = torch.from_numpy(Y_train.squeeze(-1)).long()              # (N,H,W)
    Y_val_t   = torch.from_numpy(Y_val.squeeze(-1)).long()

    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   Y_val_t),   batch_size=batch_size, shuffle=False)

    # ── Callbacks setup ────────────────────────────────────────────────────────
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
        'iou_coef':  [], 'val_iou_coef':  [],
        'lr': []
    }

    # ── Base metadata — built once, updated at each checkpoint ────────────────
    _base_meta = {
        # Identity
        'model_name':          model_name,
        'version':             __version__,
        'codename':            __codename__,
        'trained_date':        datetime.now().strftime('%Y-%m-%d'),
        # Architecture
        'architecture':        'UNet++',
        'encoder':             'resnet34',
        'encoder_weights':     'imagenet',
        'in_channels':         1,
        'classes':             ['background', 'myelin', 'axon'],
        'class_weights':       _class_weights_cfg,
        'input_size':          256,
        'activation':          'none (raw logits)',
        # Inference contract
        'normalization':       'L2 axis=1',
        'patch_size':          _config.get('patch_size', {}).get(mag, 256),
        'magnification':       mag,
        # Dataset
        'dataset_path':        str(Path(images_dir).resolve()),
        'split_mode':          split_mode,
        'val_images':          val_stems,
        'n_train_patches':     len(X_train),
        'n_val_patches':       len(X_val),
        # Training config
        'augmentation':        use_aug,
        'geo_prob':            GEO_PROB   if use_aug else None,
        'photo_prob':          PHOTO_PROB if use_aug else None,
        'batch_size':          batch_size,
        'epochs_limit':        epochs,
        'learning_rate':       LEARNING_RATE,
        'dice_weight':         DICE_WEIGHT,
        'ce_weight':           CE_WEIGHT,
        'reduce_lr_patience':  REDUCE_LR_PATIENCE,
        'early_stop_patience': EARLY_STOP_PATIENCE,
        'early_stop_min_delta':EARLY_STOP_MIN_DELTA,
        # Environment
        'gpu':                 torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'torch_version':       str(torch.__version__),
        'python_version':      sys.version.split()[0],
        'hostname':            socket.gethostname(),
    }

    # ── Training loop ──────────────────────────────────────────────────────────
    log.rule("TRAINING")
    for epoch in range(epochs):
        epoch_start = datetime.now()
        
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
            tp, fp, fn, tn = smp.metrics.get_stats(
                preds.argmax(dim=1), yb, mode="multiclass", num_classes=3
            )
            train_tp.append(tp); train_fp.append(fp)
            train_fn.append(fn); train_tn.append(tn)

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
                tp, fp, fn, tn = smp.metrics.get_stats(
                    preds.argmax(dim=1), yb, mode="multiclass", num_classes=3
                )
                val_tp.append(tp); val_fp.append(fp)
                val_fn.append(fn); val_tn.append(tn)

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

        # — Per-class dice (myelin=class 1, axon=class 2) —
        train_dice_myel = smp.metrics.f1_score(train_tp, train_fp, train_fn, train_tn, reduction="none")[:, 1].mean().item()
        train_dice_axon = smp.metrics.f1_score(train_tp, train_fp, train_fn, train_tn, reduction="none")[:, 2].mean().item()
        val_dice_myel   = smp.metrics.f1_score(val_tp,   val_fp,   val_fn,   val_tn,   reduction="none")[:, 1].mean().item()
        val_dice_axon   = smp.metrics.f1_score(val_tp,   val_fp,   val_fn,   val_tn,   reduction="none")[:, 2].mean().item()
        
        # — Log history —
        history['loss'].append(train_loss);      history['val_loss'].append(val_loss)
        history['dice_coef'].append(train_dice); history['val_dice_coef'].append(val_dice)
        history['iou_coef'].append(train_iou);   history['val_iou_coef'].append(val_iou)
        history['lr'].append(current_lr)

        # — Checkpoint and early stopping ──────────────────────────────────────
        # Primary trigger: axon+myelin dice improves meaningfully
        # Tiebreaker:      same combined dice but val_loss improves
        #                  → updates checkpoint, does NOT reset patience
        combined = val_dice_axon + val_dice_myel

        if combined > best_combined + EARLY_STOP_MIN_DELTA:
            best_combined       = combined
            best_combined_epoch = epoch + 1
            best_combined_axon  = val_dice_axon
            best_combined_myel  = val_dice_myel
            best_combined_loss  = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'meta': {
                    **_base_meta,
                    'best_epoch':       epoch + 1,
                    'best_axon_dice':   val_dice_axon,
                    'best_myelin_dice': val_dice_myel,
                    'best_val_loss':    val_loss,
                    'epochs_completed': None,   # updated after loop
                    'early_stopped':    None,   # updated after loop
                }
            }, str(model_path))
            epochs_no_improve = 0
            checkpoint_flag   = " ← CHECKPOINT"

        elif combined >= best_combined and val_loss < best_combined_loss - EARLY_STOP_MIN_DELTA:
            best_combined_loss  = val_loss
            best_combined_epoch = epoch + 1
            torch.save({
                'model_state_dict': model.state_dict(),
                'meta': {
                    **_base_meta,
                    'best_epoch':       epoch + 1,
                    'best_axon_dice':   best_combined_axon,
                    'best_myelin_dice': best_combined_myel,
                    'best_val_loss':    val_loss,
                    'epochs_completed': None,
                    'early_stopped':    None,
                }
            }, str(model_path))
            checkpoint_flag = " ← CHECKPOINT (loss tiebreak)"

        else:
            epochs_no_improve += 1
            checkpoint_flag    = ""

        # — ReduceLROnPlateau —
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            log.print(f"  ReduceLR: {current_lr:.2e} → {new_lr:.2e} (val_loss no improvement for {REDUCE_LR_PATIENCE} epochs)")

        epoch_time     = datetime.now() - epoch_start
        epoch_time_str = f"{int(epoch_time.total_seconds())}s"
        
        # — Epoch log —
        training_logger.log_epoch(epoch, {
            'epoch_time':          epoch_time_str,
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
        
        # — Early stopping —
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            log.info(
                f"Early stopping at epoch {epoch + 1} — "
                f"no improvement in axon+myelin dice for {EARLY_STOP_PATIENCE} epochs"
            )
            break

    # ── Update final fields in checkpoint ─────────────────────────────────────
    n_epochs      = len(history['loss'])
    early_stopped = n_epochs < epochs
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    checkpoint['meta']['epochs_completed'] = n_epochs
    checkpoint['meta']['early_stopped']    = early_stopped
    torch.save(checkpoint, str(model_path))

    # ── Load best weights ──────────────────────────────────────────────────────
    model.load_state_dict(checkpoint['model_state_dict'])

    # ── Checkpoint summary ─────────────────────────────────────────────────────
    training_logger.on_train_end({
        'epoch':    best_combined_epoch,
        'combined': best_combined,
        'axon':     best_combined_axon,
        'myelin':   best_combined_myel,
        'loss':     best_combined_loss,
        'path':     model_path.name,
    })

    # ── Final summary ──────────────────────────────────────────────────────────
    t_elapsed               = datetime.now() - t_start
    elapsed_str             = f"{int(t_elapsed.total_seconds() // 60)}m {int(t_elapsed.total_seconds() % 60)}s"
    final                   = history
    total_patches_processed = n_epochs * len(X_train)

    Console().print(Panel(
        f"Model saved at: {model_path}\n\n"
        f"Best checkpoint epoch:    {best_combined_epoch}\n"
        f"Best val axon dice:       {best_combined_axon:.4f}\n"
        f"Best val myelin dice:     {best_combined_myel:.4f}\n"
        f"Best val loss:            {best_combined_loss:.4f}\n\n"
        f"Final Validation Dice:    {final['val_dice_coef'][-1]:.4f}\n"
        f"Final Validation IoU:     {final['val_iou_coef'][-1]:.4f}\n\n"
        f"Total Training Time:      {elapsed_str}\n"
        f"Epochs completed:         {n_epochs}\n"
        f"Total patches processed:  {total_patches_processed:,}\n",
        title="[bold green]DeepAxon++ Training Complete[/bold green]",
        border_style="green",
        expand=False
    ))

    log.finalize(summary={
        'Model':              str(model_path),
        'GPU':                torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'Epochs':             n_epochs,
        'Early stopped':      str(early_stopped),
        'Best checkpoint':    f"epoch {best_combined_epoch}",
        'Best axon dice':     f"{best_combined_axon:.4f}",
        'Best myelin dice':   f"{best_combined_myel:.4f}",
        'Best val loss':      f"{best_combined_loss:.4f}",
        'Final val dice':     f"{final['val_dice_coef'][-1]:.4f}",
        'Final val IoU':      f"{final['val_iou_coef'][-1]:.4f}",
        'Training time':      elapsed_str,
        'Patches processed':  total_patches_processed,
        'Patches augmented':  aug_count,
    })