"""
train/train.py

Training orchestrator for DeepAxon.
Imports exclusively from train/dataset/ and train/architectures/ — no inline duplicates.
"""

from __future__ import annotations

import sys
import socket
import json
import os
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
from train.unet3plus import UNet3Plus
from utils.metrics import compute_epoch_metrics, compute_all_metrics

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
WEIGHT_DECAY         = _train_cfg.get("weight_decay",         0.01)
DICE_WEIGHT          = _train_cfg.get("dice_weight",          0.5)
CE_WEIGHT            = _train_cfg.get("ce_weight",            0.5)

# ─── Augmentation constants ───────────────────────────────────────────────────
GEO_PROB   = _prob_cfg.get("geometric_prob",   0.5)
PHOTO_PROB = _prob_cfg.get("photometric_prob", 0.25)

# ─── Class weights ────────────────────────────────────────────────────────────
_class_weights_cfg = _train_cfg.get("class_weights", [2.0, 1.0, 1.0])


def weighted_dice_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Weighted multiclass Dice loss.
    pred:    (N, C, H, W) raw logits
    target:  (N, H, W)   class indices
    weights: (C,)         per-class weights
    """
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

# ─────────── Parametric arch/encoder: _ARCH_MAP + build_model() ─────────────
_ARCH_MAP = {
    'unet':           smp.Unet,
    'attention_unet': smp.Unet,
    'unet++':         smp.UnetPlusPlus,
    'unet3+':         UNet3Plus,
    'manet':          smp.MAnet,
    'deeplabv3+':     smp.DeepLabV3Plus,
}                                                             
                                                                     
def build_model(arch: str, encoder: str, device):                   
    """                                                              
    Instantiate a segmentation model by arch/encoder name.           
    arch:    'unet' | 'attention_unet' | 'unet++' | 'unet3+'
             'manet' | 'deeplabv3+'
    encoder: 'resnet34' | 'resnet50'                                 
             'efficientnet-b3' | 'efficientnet-b4'                   
             'densenet121' | 'densenet169'                           
    All combinations valid per segmentation-models-pytorch.          
    encoder_weights: imagenet for all.                               
    """                                                         
    arch_key = arch.lower()                                          
    if arch_key not in _ARCH_MAP:                                    
        raise ValueError(                                            
            f"Unknown arch '{arch}' — must be one of {list(_ARCH_MAP)}" 
        )                                                            
    model_cls = _ARCH_MAP[arch_key]
    kwargs = dict(
        encoder_name=encoder,
        encoder_weights='imagenet',
        in_channels=1,
        classes=3,
        activation=None,
    )
    if arch_key == 'attention_unet':
        kwargs['decoder_attention_type'] = 'scse'
    model = model_cls(**kwargs)                                                   
    return model.to(device)                                          


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
            # Epoch info
            'epoch':         epoch + 1,
            'epoch_time':    logs.get('epoch_time', ''),
            'lr':            logs.get('lr',                  float('nan')),
            # Training metrics
            'loss':          logs.get('loss',                float('nan')),
            'dice':          logs.get('dice_coef',           float('nan')),
            'dice_bg':       logs.get('dice_coef_bg',        float('nan')),
            'dice_axon':     logs.get('dice_coef_axon',      float('nan')),
            'dice_myelin':   logs.get('dice_coef_myelin',    float('nan')),
            # Validation metrics
            'val_loss':      logs.get('val_loss',            float('nan')),
            'val_dice':      logs.get('val_dice_coef',       float('nan')),
            'val_dice_bg':   logs.get('val_dice_coef_bg',    float('nan')),
            'val_dice_axon': logs.get('val_dice_coef_axon',  float('nan')),
            'val_dice_myel': logs.get('val_dice_coef_myelin',float('nan')),
            # Checkpoint
            'checkpoint':    checkpoint_flag,
        }
        self.epoch_rows.append(row)
        self.log.print(
            f"  Epoch{row['epoch']:>4} ({row['epoch_time']}) | learning_rate {row['lr']:.2e}\n"
            f"  TRAIN  loss {row['loss']:.3f} | dice {row['dice']:.3f} | bg {row['dice_bg']:.3f} | ax {row['dice_axon']:.3f} | my {row['dice_myelin']:.3f}\n"
            f"  VAL    loss {row['val_loss']:.3f} | dice {row['val_dice']:.3f} | bg {row['val_dice_bg']:.3f} | ax {row['val_dice_axon']:.3f} | my {row['val_dice_myel']:.3f}  {checkpoint_flag}\n"
        )

    def on_train_end(self, checkpoint_info: dict):
        """
        Write checkpoint summary to log file only.
        Per-epoch detail is already in the live training log lines above.

        checkpoint_info keys: epoch, loss, bg, axon, myelin, iou, path
        """
        if not self.epoch_rows:
            return

        summary = (
            f"Best checkpoint @ epoch {checkpoint_info['epoch']}\n"
            f"  Val loss        : {checkpoint_info['loss']:.3f}\n"
            f"  Val bg dice     : {checkpoint_info['bg']:.3f}\n"
            f"  Val axon dice   : {checkpoint_info['axon']:.3f}\n"
            f"  Val myelin dice : {checkpoint_info['myelin']:.3f}\n"
            f"  Val IoU         : {checkpoint_info['iou']:.3f}\n"
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

    # ── Mask quality check ────────────────────────────────────────────────────────
    log.rule("SOURCE MASK QUALITY")
    mask_paths   = sorted(masks_dir.glob('*.png')) + sorted(masks_dir.glob('*.tif'))
    dirty_masks  = []
    for mask_path in mask_paths:
        mask       = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        unexpected = ~np.isin(mask, [0, 127, 128, 255])
        if unexpected.sum() > 0:
            pct = round(unexpected.sum() / mask.size * 100, 2)
            dirty_masks.append(mask_path.name)
            log.warn(f"  {mask_path.name}: {unexpected.sum()} unexpected pixels ({pct}%) — will be thresholded to nearest class")

    if not dirty_masks:
        log.success(f"No unexpected pixels found in {len(mask_paths)}/{len(mask_paths)} masks")
        
    # ── Dataset verification ────────────────────────────────────────────────────────
    log.rule("DATASET VERIFICATION")

    if not masks_dir.exists():
        log.warn(f"Masks folder not found: {masks_dir}")
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    imgs  = list(images_dir.glob('*.tif')) + list(images_dir.glob('*.tiff')) + list(images_dir.glob('*.png'))
    masks = list(masks_dir.glob('*.tif'))  + list(masks_dir.glob('*.tiff'))  + list(masks_dir.glob('*.png'))    

    img_stems     = {p.stem for p in imgs}
    mask_stems    = {p.stem for p in masks}
    matched       = img_stems & mask_stems
    missing_masks = img_stems - mask_stems
    missing_imgs  = mask_stems - img_stems

    log.info(f"Source images: {len(matched)} pairs | Missing masks: {len(missing_masks)} | Missing images: {len(missing_imgs)}")

    if patches_img.exists():
        n_patches = count_patches(str(patches_img))
        if n_patches > 0:
            log.success(
                f"Existing patches found — {n_patches} patches "
                f"({n_patches // len(matched) if matched else 0} per image) "
                f"— preprocessing will be skipped"
            )
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
        arch: str = 'unet++',                                      
        encoder: str = 'resnet34',                                  
        run_cfg: dict | None = None,                                
    ):

    """
    Full training pipeline.

    Pipeline:
        prepare_dataset → preprocess (if needed) → patch verification
        → load patches → augment → build model → train loop
        → checkpoint summary → finalize

    Checkpointing strategy:
        Trigger       : val_loss improves by > EARLY_STOP_MIN_DELTA
        Early stopping: patience counter reaches EARLY_STOP_PATIENCE epochs
                        without val_loss improvement
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
        log.rule("PATCH VERIFICATION")
        n_img = count_patches(str(paths['patches_img']))
        log.info(f"Existing patches found — skipping preprocessing ({n_img} patches)")

    # ── Verify patch alignment ─────────────────────────────────────────────────
    n_img_p  = count_patches(str(paths['patches_img']))
    n_mask_p = count_patches(str(paths['patches_mask']))
    if n_img_p != n_mask_p:
        raise ValueError(f"Patch count mismatch: {n_img_p} images vs {n_mask_p} masks")
    log.success(f"Patch alignment verified — images: {n_img_p} | masks: {n_mask_p}")

    # ── Load patches ─────────────────────────────────────────────────
    log.rule("LOADING PATCHES")
    X_train, Y_train, X_val, Y_val, split_mode, val_stems = load_all_patches(
        str(paths['images_dir']),
        str(paths['masks_dir']),
        log=log,
        train_stems = run_cfg.get('_train_stems') if run_cfg else None,  
        val_stems   = run_cfg.get('_val_stems')   if run_cfg else None,  
    )

    Y_train = Y_train.astype(np.int64)
    Y_val   = Y_val.astype(np.int64)

    # ── Augmentation ───────────────────────────────────────────────────────────
    aug_count  = 0
    aug_counts = {}
    if use_aug:
        log.rule("AUGMENTATION")
        X_train, Y_train, aug_count, aug_counts = augment_dataset_np(  
            X_train, Y_train,                                           
            aug_params=run_cfg.get('aug_params') if run_cfg else None,  
        )
        aug_pct = round(aug_count / len(X_train) * 100, 1)
        log.success(f"{aug_count}/{len(X_train)} patches modified ({aug_pct}%)")
        log.info(
            f"  Geometric   — H-flip: {aug_counts['hflip']}  |  "
            f"V-flip: {aug_counts['vflip']}  |  "
            f"Rotation: {aug_counts['rotation']}  |  "
            f"Elastic: {aug_counts['elastic']}"                          
        )
        log.info(
            f"  Photometric — Brightness: {aug_counts['brightness']}  |  "
            f"Gamma: {aug_counts['gamma']}  |  "
            f"Noise: {aug_counts['noise']}  |  "
            f"Blur: {aug_counts['blur']}  |  "
            f"CLAHE: {aug_counts['clahe']}  |  "
            f"Contrast stretch: {aug_counts['contrast_stretch']}  |  "  
            f"Erase: {aug_counts['erase']}"                             
        )

    # ── Deterministic seeding — ensures exact reproducibility from result.json ─
    if run_cfg is not None:                                                  
        _seed = run_cfg.get('seed', 42)                                      
        import random                                                         
        random.seed(_seed)                                                    
        np.random.seed(_seed)                                                 
        torch.manual_seed(_seed)                                              
        torch.cuda.manual_seed_all(_seed)                                    
        torch.backends.cudnn.deterministic = True                            
        torch.backends.cudnn.benchmark     = False                           
        log.info(f"Deterministic seeding — seed={_seed}")                   
        
    # ── Build model ────────────────────────────────────────────────────────────
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _weights = run_cfg.get('class_weights', _class_weights_cfg) if run_cfg else _class_weights_cfg
    CLASS_WEIGHTS = torch.tensor(_weights, dtype=torch.float32).to(device)  
    _ce_loss      = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

    def loss_fn(pred, target):
        dice = weighted_dice_loss(pred, target, CLASS_WEIGHTS)
        ce   = _ce_loss(pred, target)
        return DICE_WEIGHT * dice + CE_WEIGHT * ce
    
    model = build_model(arch, encoder, device)
    n_params  = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE, min_lr=REDUCE_LR_MIN_LR
    )
    
    # ── Training setup summary ─────────────────────────────────────────────────
    log.rule("TRAINING SETUP")
    log.log_dict({
        # Architecture
        'Architecture':     f"{arch} — {encoder} encoder, imagenet weights",
        'Parameters':       f"{n_params:,}",
        'Class weights':    f"bg={_weights[0]} myelin={_weights[1]} axon={_weights[2]}",
        'Device':           str(device),
        # Dataset
        'Train patches':    len(X_train),
        'Val patches':      len(X_val),
        'Test/Train split': split_mode,
        # Training
        'Batch size':       batch_size,
        'Epoch limit':      epochs,
        'Augmentation':     f"ON — geo_prob={GEO_PROB:.2f} photo_prob={PHOTO_PROB:.2f}" if use_aug else "OFF",
        'Loss function':    f"Weighted Dice ({DICE_WEIGHT}) + CrossEntropy ({CE_WEIGHT})",
        'Optimizer':        f"AdamW lr={LEARNING_RATE} wd={WEIGHT_DECAY}",
        'ReduceLR':         f"patience={REDUCE_LR_PATIENCE}, factor={REDUCE_LR_FACTOR}, min_lr={REDUCE_LR_MIN_LR} — monitors val_loss",
        'EarlyStopping':    f"patience={EARLY_STOP_PATIENCE}, min_delta={EARLY_STOP_MIN_DELTA} — monitors val_loss",
        'Checkpoint':       "best val_loss",
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
    if run_cfg is not None:                                          
        model_dir = Path(run_cfg['output']['models_dir'])            
    else:                                                            
        model_dir = get_model_dir(images_dir)                       
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}.pt"
    
    # Best checkpoint trackers — all from the same epoch (best val_loss)
    best_val_loss         = float('inf')
    best_val_bg           = 0.0
    best_val_axon         = 0.0
    best_val_myel         = 0.0
    best_val_iou          = 0.0
    best_checkpoint_epoch = 0
    epochs_no_improve     = 0

    training_logger = TrainingLogger(log, use_aug)

    history = {
        'loss':          [],
        'val_loss':      [],
        'dice_coef':     [],
        'val_dice_coef': [],
        'iou_coef':      [],
        'val_iou_coef':  [],
        'lr':            [],
    }

    # ── Base metadata — built once, updated at each checkpoint ────────────────
    _base_meta = {
        # Identity
        'model_name':          model_name,
        'version':             __version__,
        'codename':            __codename__,
        'trained_date':        datetime.now().strftime('%Y-%m-%d'),
        # Architecture
        'architecture':        arch,
        'encoder':             encoder,
        'encoder_weights':     'imagenet',
        'in_channels':         1,
        'classes':             ['background', 'myelin', 'axon'],
        'class_weights':       _weights,
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
        'last_batch_fullness': f"{int((len(X_train) % batch_size) / batch_size * 100) if (len(X_train) % batch_size) > 0 else 100}%",
        'epochs_limit':        epochs,
        'learning_rate':       LEARNING_RATE,
        'dice_weight':         DICE_WEIGHT,
        'ce_weight':           CE_WEIGHT,
        'checkpoint_metric':   'val_loss',
        'reduce_lr_patience':  REDUCE_LR_PATIENCE,
        'early_stop_patience': EARLY_STOP_PATIENCE,
        'early_stop_min_delta':EARLY_STOP_MIN_DELTA,
        'n_images':        run_cfg.get('n_images')    if run_cfg else None,
        'val_fraction':    run_cfg.get('val_fraction') if run_cfg else None,  
        'train_pct':       run_cfg.get('train_pct')   if run_cfg else None,  
        'val_pct':         run_cfg.get('val_pct')     if run_cfg else None,  
        'seed':            run_cfg.get('seed')        if run_cfg else None,  
        'run_id':          run_cfg.get('run_id')      if run_cfg else None,
        # Environment
        'gpu':                 torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'torch_version':       str(torch.__version__),
        'python_version':      sys.version.split()[0],
        'hostname':            socket.gethostname(),
    }

    # ── Training loop ──────────────────────────────────────────────────────────
    _save_checkpoint = run_cfg.get('save_checkpoint', True) if run_cfg else True                                         
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
                preds     = model(xb)
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
        current_lr  = optimizer.param_groups[0]['lr']
        
        train_m = compute_epoch_metrics(train_tp, train_fp, train_fn, train_tn)  
        val_m   = compute_epoch_metrics(val_tp,   val_fp,   val_fn,   val_tn)  
        
        train_dice = train_m['dice_macro']   
        val_dice   = val_m['dice_macro']     
        train_iou  = train_m['iou_macro']    
        val_iou    = val_m['iou_macro']      
        
        # — Per-class dice (background=class 0, myelin=class 1, axon=class 2) —
        train_dice_bg   = train_m['dice_bg'];    train_dice_myel = train_m['dice_myelin'];  train_dice_axon = train_m['dice_axon']  
        val_dice_bg     = val_m['dice_bg'];      val_dice_myel   = val_m['dice_myelin'];    val_dice_axon   = val_m['dice_axon']    
     
        # — Log history —
        history['loss'].append(train_loss);      history['val_loss'].append(val_loss)
        history['dice_coef'].append(train_dice); history['val_dice_coef'].append(val_dice)
        history['iou_coef'].append(train_iou);   history['val_iou_coef'].append(val_iou)
        history['lr'].append(current_lr)

        # — Checkpoint: save on best val_loss —
        if val_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
            best_val_loss         = val_loss
            best_val_bg           = val_dice_bg
            best_val_axon         = val_dice_axon
            best_val_myel         = val_dice_myel
            best_val_iou          = val_iou
            best_checkpoint_epoch = epoch + 1
            
            if _save_checkpoint:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'meta': {
                        **_base_meta,
                        'best_epoch':       epoch + 1,
                        'best_val_loss':    val_loss,
                        'best_bg_dice':     val_dice_bg,
                        'best_axon_dice':   val_dice_axon,
                        'best_myelin_dice': val_dice_myel,
                        'best_val_iou':     val_iou,
                        'epochs_completed': None,
                        'early_stopped':    None,
                    }
                }, str(model_path))
            
            epochs_no_improve = 0
            checkpoint_flag   = " ← CHECKPOINT"

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
            # Epoch info
            'epoch_time':           epoch_time_str,
            'lr':                   current_lr,
            # Training metrics
            'loss':                 train_loss,
            'dice_coef':            train_dice,
            'dice_coef_bg':         train_dice_bg,
            'dice_coef_axon':       train_dice_axon,
            'dice_coef_myelin':     train_dice_myel,
            # Validation metrics
            'val_loss':             val_loss,
            'val_dice_coef':        val_dice,
            'val_dice_coef_bg':     val_dice_bg,
            'val_dice_coef_axon':   val_dice_axon,
            'val_dice_coef_myelin': val_dice_myel,
        }, checkpoint_flag=checkpoint_flag)
        
        # — Early stopping —
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            log.info(
                f"Early stopping at epoch {epoch + 1} — "
                f"no improvement in val_loss for {EARLY_STOP_PATIENCE} epochs"
            )
            break

    # ── Update final fields in checkpoint ─────────────────────────────────────
    n_epochs      = len(history['loss'])
    early_stopped = n_epochs < epochs
    if _save_checkpoint:
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
        checkpoint['meta']['epochs_completed'] = n_epochs
        checkpoint['meta']['early_stopped']    = early_stopped
        torch.save(checkpoint, str(model_path))
        model.load_state_dict(checkpoint['model_state_dict'])   # Load best checkpoint for final evaluation
    
    else:
        log.info(                                                    
            "save_checkpoint=False — post-training evaluation on "  
            f"final epoch weights (epoch {n_epochs}), not best "    
            f"checkpoint (epoch {best_checkpoint_epoch})"           
        )                                                            

    # ── Post-training evaluation — all metrics at best checkpoint ─────────────
    log.rule("POST-TRAINING EVALUATION")
    model.eval()

    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds  = model(xb)
            all_preds.append(preds.argmax(dim=1).cpu())
            all_labels.append(yb.cpu())

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    eval_metrics = compute_all_metrics(all_preds, all_labels, device)  

    log.info(f"Dice      — macro: {eval_metrics['dice_macro']:.4f}  | bg: {eval_metrics['dice_bg']:.4f}  | myelin: {eval_metrics['dice_myelin']:.4f}  | axon: {eval_metrics['dice_axon']:.4f}")
    log.info(f"IoU       — macro: {eval_metrics['iou_macro']:.4f}   | bg: {eval_metrics['iou_bg']:.4f}   | myelin: {eval_metrics['iou_myelin']:.4f}   | axon: {eval_metrics['iou_axon']:.4f}")
    log.info(f"Precision — macro: {eval_metrics['precision_macro']:.4f} | bg: {eval_metrics['precision_bg']:.4f} | myelin: {eval_metrics['precision_myelin']:.4f} | axon: {eval_metrics['precision_axon']:.4f}")
    log.info(f"Recall    — macro: {eval_metrics['recall_macro']:.4f} | bg: {eval_metrics['recall_bg']:.4f} | myelin: {eval_metrics['recall_myelin']:.4f} | axon: {eval_metrics['recall_axon']:.4f}")
    log.info(f"HD95      — macro: {eval_metrics['hd95_macro']:.4f}  | bg: {eval_metrics['hd95_bg']:.4f}  | myelin: {eval_metrics['hd95_myelin']:.4f}  | axon: {eval_metrics['hd95_axon']:.4f}")
    
    # ── Checkpoint summary ─────────────────────────────────────────────────────
    training_logger.on_train_end({
        'epoch':  best_checkpoint_epoch,
        'loss':   best_val_loss,
        'bg':     best_val_bg,
        'axon':   best_val_axon,
        'myelin': best_val_myel,
        'iou':    best_val_iou,
        'path':   model_path.name,
    })

    # ── Final summary ──────────────────────────────────────────────────────────
    t_elapsed               = datetime.now() - t_start
    elapsed_str             = f"{int(t_elapsed.total_seconds() // 60)}m {int(t_elapsed.total_seconds() % 60)}s"
    total_patches_processed = n_epochs * len(X_train)

    log.finalize(summary={
        'Model':             str(model_path),
        'GPU':               torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'Training time':     elapsed_str,
        'Epochs':            f"{n_epochs} (early stopped)" if early_stopped else str(n_epochs),
        'Patches processed': total_patches_processed,
        'Augmented':         aug_count,
        'Best epoch':        best_checkpoint_epoch,
        'Best val_loss':     f"{best_val_loss:.4f}",
        'Dice macro':        f"{eval_metrics['dice_macro']:.4f}",
        'IoU macro':         f"{eval_metrics['iou_macro']:.4f}",
        'Precision macro':   f"{eval_metrics['precision_macro']:.4f}",
        'Recall macro':      f"{eval_metrics['recall_macro']:.4f}",
        'HD95 macro':        f"{eval_metrics['hd95_macro']:.4f}",
        'Dice axon':         f"{eval_metrics['dice_axon']:.4f}",
        'Dice myelin':       f"{eval_metrics['dice_myelin']:.4f}",
        'HD95 axon':         f"{eval_metrics['hd95_axon']:.4f}",
        'HD95 myelin':       f"{eval_metrics['hd95_myelin']:.4f}",
        'HD95 myelin_axon':  f"{eval_metrics['hd95_myelin_axon']:.4f}",
    })
    
    if run_cfg is not None:                                         
        results_dir = Path(run_cfg['output']['results_dir']) / run_cfg['run_id']  
        results_dir.mkdir(parents=True, exist_ok=True)       
              
        result = {
            'run_id':            run_cfg['run_id'],
            'stage':             run_cfg.get('stage', 'sweep'),
            'wave':              run_cfg.get('wave', 1),
            'arch':              arch,
            'encoder':           encoder,
            'class_weights':     _weights,
            'n_images':          run_cfg.get('n_images'),
            'train_pct':         run_cfg.get('train_pct'),
            'val_pct':           run_cfg.get('val_pct'),
            'seed':              run_cfg.get('seed'),
            'augmentation':      use_aug,
            'checkpoint_metric': 'val_loss',
            'best_epoch':        best_checkpoint_epoch,
            'epochs_completed':  n_epochs,
            'early_stopped':     early_stopped,
            'best_val_loss':     round(best_val_loss, 4),
            'train_stems':       run_cfg.get('train_stems', []),
            'val_stems':         run_cfg.get('val_stems',   []),
            'model_path':        str(model_path),
            **eval_metrics,                                          
        }  
                                                                 
        result_path = results_dir / 'result.json'                  
        with open(result_path, 'w') as f:                          
            json.dump(result, f, indent=2)                          
        log.success(f"Result written → {result_path}")             