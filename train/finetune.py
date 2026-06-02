"""
train/finetune.py

Entry point for: python -m train.finetune

Also accessible as a module import from train/.

Domain adaptation via fine-tuning of a pretrained DeepAxon model.
Allows users to adapt an existing model to their images with 5-10 annotated
masks, without training from scratch.

Three freeze strategies:
    --freeze full      Head only (final conv) — same species, minor variation
                       lr=1e-3, patience=20
    --freeze encoder   Decoder + head (recommended default) — different staining/animal
                       lr=1e-4, patience=15
    --freeze none      All layers — very different domain
                       lr=1e-5, patience=10

Prompt order:
    1. Base model selection
    2. Images folder
    3. Magnification (pre-filled from base model metadata)
    4. Freeze strategy
    5. Model name
    6. Epoch limit
    7. Augmentation
    8. Batch size
    9. Log setup + run

Usage:
    python -m train.finetune
"""

import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
import json
from pathlib import Path
from datetime import datetime

import torch
from rich.console import Console
from rich.panel import Panel
from rich.box import DOUBLE
from rich.align import Align
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import DeepAxonLogger
from utils.gpu import setup_gpu_console
from utils.helpers import (
    get_int_input, get_yes_no,
    compute_batch_options, get_model_dir, get_log_dir,
    list_files, load_config, print_panel
)
from utils.version import __version__, __codename__
from train.train import (
    train_model, build_model, weighted_dice_loss,
    TrainingLogger, prepare_dataset,
    REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_MIN_LR,
    EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA,
    WEIGHT_DECAY, DICE_WEIGHT, CE_WEIGHT,
    GEO_PROB, PHOTO_PROB, _class_weights_cfg,
)
from train.dataset.data_loader import load_all_patches
from train.dataset.augment import augment_dataset_np
from utils.metrics import compute_epoch_metrics, compute_all_metrics
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import socket

console  = Console()
has_gpu  = torch.cuda.is_available()

MAG_OPTIONS       = ['40X', '100X']
PATCHES_PER_IMAGE = 63

# ── Freeze strategy defaults ──────────────────────────────────────────────────
FREEZE_STRATEGIES = {
    'full':    {'lr': 1e-3, 'patience': 20, 'label': 'Head only'},
    'encoder': {'lr': 1e-4, 'patience': 15, 'label': 'Decoder + head (recommended)'},
    'none':    {'lr': 1e-5, 'patience': 10, 'label': 'Full fine-tune'},
}

DEFAULT_FINETUNE_EPOCHS = 100


# ── Model discovery ───────────────────────────────────────────────────────────

def find_models(models_dir: Path) -> list[Path]:
    """Find all .pt files in models/ directory."""
    return sorted(models_dir.glob('*.pt'))


def load_base_model_meta(model_path: Path) -> dict:
    """Load metadata from a .pt checkpoint without loading full weights."""
    checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
    return checkpoint.get('meta', {})


# ── Layer freezing ────────────────────────────────────────────────────────────

def freeze_layers(model, strategy: str, log: DeepAxonLogger):
    """
    Apply layer freezing strategy.

    strategy:
        'full'    — freeze encoder + decoder, train head only
        'encoder' — freeze encoder, train decoder + head (default)
        'none'    — train all layers

    Logs trainable parameter count after freezing.
    """
    if strategy == 'full':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.segmentation_head.parameters():
            param.requires_grad = True

    elif strategy == 'encoder':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = True
        for param in model.segmentation_head.parameters():
            param.requires_grad = True

    elif strategy == 'none':
        for param in model.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown freeze strategy '{strategy}' — must be full | encoder | none")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    log.info(
        f"Freeze strategy: {strategy} | "
        f"Trainable: {trainable:,} / {total:,} params | "
        f"Frozen: {frozen:,}"
    )


# ── Fine-tune entry point ─────────────────────────────────────────────────────

def finetune_model(
    base_model_path: Path,
    images_dir:      str,
    model_name:      str,
    epochs:          int,
    batch_size:      int,
    use_aug:         bool,
    freeze:          str,
    log:             DeepAxonLogger,
    mag:             str,
):
    """
    Fine-tune a pretrained DeepAxon model on new images.

    Loads base model weights, applies freeze strategy, then runs the
    standard training loop with fine-tuning learning rate.

    The output .pt file records fine_tuned_from in metadata for lineage
    tracking.
    """
    from utils.helpers import count_patches
    from train.dataset.preprocess import batch_process

    t_start   = datetime.now()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ft_cfg    = FREEZE_STRATEGIES[freeze]
    ft_lr     = ft_cfg['lr']
    ft_pat    = ft_cfg['patience']

    # ── Load base model ───────────────────────────────────────────────────────
    log.rule("BASE MODEL")
    checkpoint = torch.load(str(base_model_path), map_location=device, weights_only=False)
    meta       = checkpoint.get('meta', {})

    arch    = meta.get('architecture', 'unet++')
    encoder = meta.get('encoder',      'resnet34')
    _weights = meta.get('class_weights', _class_weights_cfg)

    log.log_dict({
        'Base model':    base_model_path.name,
        'Architecture':  arch,
        'Encoder':       encoder,
        'Class weights': f"bg={_weights[0]} myelin={_weights[1]} axon={_weights[2]}",
        'Trained date':  meta.get('trained_date', 'unknown'),
        'Base version':  meta.get('version', 'unknown'),
    })

    # ── Prepare dataset ───────────────────────────────────────────────────────
    paths = prepare_dataset(images_dir, mag, log)

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

    n_img_p  = count_patches(str(paths['patches_img']))
    n_mask_p = count_patches(str(paths['patches_mask']))
    if n_img_p != n_mask_p:
        raise ValueError(f"Patch count mismatch: {n_img_p} images vs {n_mask_p} masks")

    log.rule("LOADING PATCHES")
    X_train, Y_train, X_val, Y_val, split_mode, val_stems = load_all_patches(
        str(paths['images_dir']),
        str(paths['masks_dir']),
        log=log,
    )

    Y_train = Y_train.astype(np.int64)
    Y_val   = Y_val.astype(np.int64)

    # ── Augmentation ──────────────────────────────────────────────────────────
    aug_count  = 0
    aug_counts = {}
    if use_aug:
        log.rule("AUGMENTATION")
        X_train, Y_train, aug_count, aug_counts = augment_dataset_np(X_train, Y_train)
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
            f"CLAHE: {aug_counts['clahe']}"
        )

    # ── Build model + load weights ────────────────────────────────────────────
    CLASS_WEIGHTS = torch.tensor(_weights, dtype=torch.float32).to(device)
    _ce_loss      = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

    def loss_fn(pred, target):
        dice = weighted_dice_loss(pred, target, CLASS_WEIGHTS)
        ce   = _ce_loss(pred, target)
        return DICE_WEIGHT * dice + CE_WEIGHT * ce

    model = build_model(arch, encoder, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    log.success(f"Base weights loaded from {base_model_path.name}")

    # ── Apply freeze strategy ─────────────────────────────────────────────────
    log.rule("FINE-TUNE SETUP")
    freeze_layers(model, freeze, log)

    # Only pass trainable params to optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=ft_lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE, min_lr=REDUCE_LR_MIN_LR
    )

    n_params_total     = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log.log_dict({
        'Base model':       base_model_path.name,
        'Freeze strategy':  f"{freeze} — {ft_cfg['label']}",
        'Fine-tune LR':     f"{ft_lr:.0e}",
        'ES patience':      ft_pat,
        'Trainable params': f"{n_params_trainable:,} / {n_params_total:,}",
        'Device':           str(device),
        'Train patches':    len(X_train),
        'Val patches':      len(X_val),
        'Split mode':       split_mode,
        'Batch size':       batch_size,
        'Epoch limit':      epochs,
        'Augmentation':     f"ON — geo_prob={GEO_PROB:.2f} photo_prob={PHOTO_PROB:.2f}" if use_aug else "OFF",
        'Loss function':    f"Weighted Dice ({DICE_WEIGHT}) + CrossEntropy ({CE_WEIGHT})",
        'Checkpoint':       "best val_loss",
    })

    # ── DataLoaders ───────────────────────────────────────────────────────────
    X_train_t = torch.from_numpy(X_train.transpose(0, 3, 1, 2)).float()
    X_val_t   = torch.from_numpy(X_val.transpose(0, 3, 1, 2)).float()
    Y_train_t = torch.from_numpy(Y_train.squeeze(-1)).long()
    Y_val_t   = torch.from_numpy(Y_val.squeeze(-1)).long()

    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   Y_val_t),   batch_size=batch_size, shuffle=False)

    model_dir  = get_model_dir(images_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}.pt"

    # ── Checkpoint metadata ───────────────────────────────────────────────────
    _base_meta = {
        'model_name':          model_name,
        'version':             __version__,
        'codename':            __codename__,
        'trained_date':        datetime.now().strftime('%Y-%m-%d'),
        'fine_tuned_from':     base_model_path.name,          # ← lineage tracking
        'freeze_strategy':     freeze,
        'finetune_lr':         ft_lr,
        'architecture':        arch,
        'encoder':             encoder,
        'encoder_weights':     'imagenet',
        'in_channels':         1,
        'classes':             ['background', 'myelin', 'axon'],
        'class_weights':       _weights,
        'input_size':          256,
        'activation':          'none (raw logits)',
        'normalization':       'L2 axis=1',
        'magnification':       mag,
        'dataset_path':        str(Path(images_dir).resolve()),
        'split_mode':          split_mode,
        'val_images':          val_stems,
        'n_train_patches':     len(X_train),
        'n_val_patches':       len(X_val),
        'augmentation':        use_aug,
        'batch_size':          batch_size,
        'epochs_limit':        epochs,
        'finetune_lr':         ft_lr,
        'dice_weight':         DICE_WEIGHT,
        'ce_weight':           CE_WEIGHT,
        'checkpoint_metric':   'val_loss',
        'gpu':                 torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'torch_version':       str(torch.__version__),
        'python_version':      sys.version.split()[0],
        'hostname':            socket.gethostname(),
    }

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss         = float('inf')
    best_val_bg           = 0.0
    best_val_axon         = 0.0
    best_val_myel         = 0.0
    best_val_iou          = 0.0
    best_checkpoint_epoch = 0
    epochs_no_improve     = 0
    training_logger       = TrainingLogger(log, use_aug)

    log.rule("FINE-TUNING")
    for epoch in range(epochs):
        epoch_start = datetime.now()

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

        train_tp = torch.cat(train_tp); train_fp = torch.cat(train_fp)
        train_fn = torch.cat(train_fn); train_tn = torch.cat(train_tn)

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

        val_tp = torch.cat(val_tp); val_fp = torch.cat(val_fp)
        val_fn = torch.cat(val_fn); val_tn = torch.cat(val_tn)

        train_loss /= len(X_train_t)
        val_loss   /= len(X_val_t)
        current_lr  = optimizer.param_groups[0]['lr']

        train_m = compute_epoch_metrics(train_tp, train_fp, train_fn, train_tn)
        val_m   = compute_epoch_metrics(val_tp,   val_fp,   val_fn,   val_tn)

        train_dice    = train_m['dice_macro'];  val_dice    = val_m['dice_macro']
        train_iou     = train_m['iou_macro'];   val_iou     = val_m['iou_macro']
        train_dice_bg = train_m['dice_bg'];     val_dice_bg = val_m['dice_bg']
        train_dice_myel = train_m['dice_myelin']; val_dice_myel = val_m['dice_myelin']
        train_dice_axon = train_m['dice_axon'];   val_dice_axon = val_m['dice_axon']

        if val_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
            best_val_loss         = val_loss
            best_val_bg           = val_dice_bg
            best_val_axon         = val_dice_axon
            best_val_myel         = val_dice_myel
            best_val_iou          = val_iou
            best_checkpoint_epoch = epoch + 1
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
            checkpoint_flag   = ""

        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            log.print(f"  ReduceLR: {current_lr:.2e} → {new_lr:.2e}")

        epoch_time_str = f"{int((datetime.now() - epoch_start).total_seconds())}s"
        training_logger.log_epoch(epoch, {
            'epoch_time': epoch_time_str, 'lr': current_lr,
            'loss': train_loss, 'dice_coef': train_dice,
            'dice_coef_bg': train_dice_bg, 'dice_coef_axon': train_dice_axon,
            'dice_coef_myelin': train_dice_myel,
            'val_loss': val_loss, 'val_dice_coef': val_dice,
            'val_dice_coef_bg': val_dice_bg, 'val_dice_coef_axon': val_dice_axon,
            'val_dice_coef_myelin': val_dice_myel,
        }, checkpoint_flag=checkpoint_flag)

        if epochs_no_improve >= ft_pat:
            log.info(f"Early stopping at epoch {epoch + 1} — no val_loss improvement for {ft_pat} epochs")
            break

    # ── Finalize ──────────────────────────────────────────────────────────────
    n_epochs      = epoch + 1
    early_stopped = n_epochs < epochs
    checkpoint    = torch.load(str(model_path), map_location=device, weights_only=False)
    checkpoint['meta']['epochs_completed'] = n_epochs
    checkpoint['meta']['early_stopped']    = early_stopped
    torch.save(checkpoint, str(model_path))
    model.load_state_dict(checkpoint['model_state_dict'])

    log.rule("POST-TRAINING EVALUATION")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            all_preds.append(model(xb).argmax(dim=1).cpu())
            all_labels.append(yb.cpu())

    eval_metrics = compute_all_metrics(torch.cat(all_preds), torch.cat(all_labels), device)

    log.info(f"Dice  — macro: {eval_metrics['dice_macro']:.4f} | axon: {eval_metrics['dice_axon']:.4f} | myelin: {eval_metrics['dice_myelin']:.4f} | bg: {eval_metrics['dice_bg']:.4f}")
    log.info(f"HD95  — macro: {eval_metrics['hd95_macro']:.4f} | axon: {eval_metrics['hd95_axon']:.4f} | myelin: {eval_metrics['hd95_myelin']:.4f}")

    training_logger.on_train_end({
        'epoch': best_checkpoint_epoch, 'loss': best_val_loss,
        'bg': best_val_bg, 'axon': best_val_axon,
        'myelin': best_val_myel, 'iou': best_val_iou,
        'path': model_path.name,
    })

    t_elapsed = datetime.now() - t_start
    elapsed_str = f"{int(t_elapsed.total_seconds() // 60)}m {int(t_elapsed.total_seconds() % 60)}s"

    log.finalize(summary={
        'Base model':        base_model_path.name,
        'Fine-tuned model':  str(model_path),
        'Freeze strategy':   f"{freeze} — {ft_cfg['label']}",
        'GPU':               torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'Training time':     elapsed_str,
        'Epochs':            f"{n_epochs} (early stopped)" if early_stopped else str(n_epochs),
        'Best epoch':        best_checkpoint_epoch,
        'Best val_loss':     f"{best_val_loss:.4f}",
        'Dice macro':        f"{eval_metrics['dice_macro']:.4f}",
        'Dice axon':         f"{eval_metrics['dice_axon']:.4f}",
        'Dice myelin':       f"{eval_metrics['dice_myelin']:.4f}",
        'HD95 axon':         f"{eval_metrics['hd95_axon']:.4f}",
        'HD95 myelin':       f"{eval_metrics['hd95_myelin']:.4f}",
    })


# ── Interactive entry point ───────────────────────────────────────────────────

def main():
    setup_gpu_console()

    print_panel(console, Panel(
        Align.center("[bold white]Domain Adaptation via Fine-Tuning[/bold white]"),
        title="[bold cyan]DEEPAXON — FINE-TUNE[/bold cyan]",
        border_style="bright_cyan",
        box=DOUBLE,
        expand=True,
        padding=(1, 4)
    ))

    config    = load_config()
    train_cfg = config.get("training", {})

    # ── Base model selection ──────────────────────────────────────────────────
    models_dir = Path(__file__).resolve().parent.parent / 'models'
    models     = find_models(models_dir)

    if not models:
        console.print(f"[red]No .pt models found in {models_dir}[/red]")
        console.print("Place a pretrained .pt model in the models/ folder and try again.")
        sys.exit(1)

    console.print("\n[bold]Available base models:[/bold]")
    for i, m in enumerate(models, 1):
        meta = load_base_model_meta(m)
        ft_from = meta.get('fine_tuned_from', '')
        lineage = f" (fine-tuned from {ft_from})" if ft_from else ""
        console.print(
            f"  [{i}]  {m.name}{lineage}\n"
            f"       arch={meta.get('architecture','?')} "
            f"encoder={meta.get('encoder','?')} "
            f"mag={meta.get('magnification','?')} "
            f"trained={meta.get('trained_date','?')}"
        )

    while True:
        raw = input(f"\nSelect base model [1-{len(models)}]: ").strip()
        try:
            choice = int(raw)
            if 1 <= choice <= len(models):
                base_model_path = models[choice - 1]
                break
        except ValueError:
            pass
        console.print(f"[red]Please enter a number between 1 and {len(models)}.[/red]")

    base_meta = load_base_model_meta(base_model_path)
    console.print(f"\n[green]✓ Base model: {base_model_path.name}[/green]")

    # ── Images folder ─────────────────────────────────────────────────────────
    while True:
        images_dir = input("\nPath to fine-tuning images folder: ").strip().strip('"')
        if os.path.isdir(images_dir):
            break
        console.print(f"[red]Folder not found: {images_dir}[/red]")

    images_path = Path(images_dir).resolve()
    log_dir     = get_log_dir(str(images_path))
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Magnification — pre-filled from base model ────────────────────────────
    base_mag = base_meta.get('magnification', '')
    if base_mag in MAG_OPTIONS:
        console.print(f"\n[dim]Magnification pre-filled from base model: {base_mag}[/dim]")
        raw = input(f"Press Enter to use {base_mag}, or select [1] 40X  [2] 100X: ").strip()
        if raw == '':
            mag = base_mag
        elif raw in ('1', '2'):
            mag = MAG_OPTIONS[int(raw) - 1]
        else:
            mag = base_mag
    else:
        raw = input("\nSelect imaging magnification — [1] 40X  [2] 100X: ").strip()
        while raw not in ('1', '2'):
            raw = input("Invalid — [1] 40X  [2] 100X: ").strip()
        mag = MAG_OPTIONS[int(raw) - 1]

    # ── Freeze strategy ───────────────────────────────────────────────────────
    t = Text(justify="left")
    t.append("Select fine-tuning strategy:\n\n", style="bold")
    t.append("  [1]  Head only     ", style="white")
    t.append("— final layer only. Same species, minor staining variation.        lr=1e-3\n", style="dim")
    t.append("  [2]  Decoder       ", style="white")
    t.append("— decoder + head. Different staining/animal. (recommended)         lr=1e-4\n", style="green")
    t.append("  [3]  Full          ", style="white")
    t.append("— all layers. Very different domain (species, staining method).     lr=1e-5\n", style="dim")
    print_panel(console, Panel(
        t,
        title="[bold orange1]Fine-Tuning Strategy[/bold orange1]",
        border_style="orange1",
        box=DOUBLE,
        expand=True
    ))

    freeze_map = {'1': 'full', '2': 'encoder', '3': 'none'}
    raw = input("Select [1-3] (press Enter for recommended [2]): ").strip()
    if raw == '':
        raw = '2'
    while raw not in freeze_map:
        raw = input("Invalid — select [1-3]: ").strip()
        if raw == '':
            raw = '2'
    freeze = freeze_map[raw]
    ft_cfg = FREEZE_STRATEGIES[freeze]
    console.print(f"\n[green]✓ Strategy: {freeze} — {ft_cfg['label']} (lr={ft_cfg['lr']:.0e})[/green]")

    # ── Model name ────────────────────────────────────────────────────────────
    base_stem  = base_model_path.stem
    auto_name  = f"{base_stem}_ft_{freeze}_{timestamp}"
    console.print(f"\n[dim]Auto-generated name: {auto_name}[/dim]")
    custom     = input("Enter model name (press Enter to use auto-generated): ").strip()
    model_name = custom if custom else auto_name

    # ── Epoch limit ───────────────────────────────────────────────────────────
    epochs = get_int_input(
        f"Epochs (press Enter for fine-tune default={DEFAULT_FINETUNE_EPOCHS}): ",
        default=DEFAULT_FINETUNE_EPOCHS, min_val=1
    )

    # ── Augmentation ──────────────────────────────────────────────────────────
    use_aug = get_yes_no("Use data augmentation?", default=True)

    # ── Batch size ────────────────────────────────────────────────────────────
    images_sub  = images_path / "images"
    patches_dir = images_sub / 'cropped' / 'patches'
    if patches_dir.exists():
        from utils.helpers import list_files as _lf, count_patches as _cp
        all_patch_files = _lf(str(patches_dir), extensions=('.png', '.tif', '.tiff'))
        n_train_patches = len(all_patch_files)
        n_val_patches   = 0
    else:
        n_imgs          = len(list_files(str(images_sub), extensions=('.tif', '.tiff', '.png')))
        n_train_patches = max(1, int(n_imgs * 0.8)) * PATCHES_PER_IMAGE
        n_val_patches   = max(1, n_imgs - int(n_imgs * 0.8)) * PATCHES_PER_IMAGE

    batch_size, remainder, bs_status = \
        _select_batch_size_simple(n_train_patches, n_val_patches)

    # ── Log setup ─────────────────────────────────────────────────────────────
    log_path = str(log_dir / f"{model_name}_finetune_log.txt")
    log      = DeepAxonLogger(log_path=log_path, program="DeepAxon Fine-Tune")

    log.rule("RUN CONFIGURATION")
    log.log_dict({
        'Mode':           'fine-tune (interactive)',
        'Base model':     base_model_path.name,
        'Freeze':         f"{freeze} — {ft_cfg['label']}",
        'Model name':     model_name,
        'Magnification':  mag,
        'Epoch limit':    epochs,
        'Augmentation':   'ON' if use_aug else 'OFF',
        'Batch size':     batch_size,
        'Images folder':  str(images_path),
        'Log file':       log_path,
    })

    finetune_model(
        base_model_path = base_model_path,
        images_dir      = str(images_path),
        model_name      = model_name,
        epochs          = epochs,
        batch_size      = batch_size,
        use_aug         = use_aug,
        freeze          = freeze,
        log             = log,
        mag             = mag,
    )

    console.print(f"\n[dim]Log saved to: {log_path}[/dim]")


def _select_batch_size_simple(n_train_patches: int, n_val_patches: int) -> tuple[int, int, str]:
    """Simplified batch size selection for fine-tuning — fewer patches expected."""
    opts      = compute_batch_options(n_train_patches, use_gpu=has_gpu)
    menu_items = []

    console.print(f"\n[bold]Batch size — {n_train_patches} training patches:[/bold]")
    for bs, remainder in (opts.get('acceptable') or []):
        idx  = len(menu_items) + 1
        pct  = '' if remainder == 0 else f" ({int(remainder/bs*100)}% last batch)"
        console.print(f"  [{idx}]  bs={bs}{pct}")
        menu_items.append((bs, remainder))

    if not menu_items:
        console.print("  [1]  bs=16")
        menu_items.append((16, n_train_patches % 16))

    custom_idx = len(menu_items) + 1
    console.print(f"  [{custom_idx}]  Custom\n")

    while True:
        raw = input(f"Select [1-{custom_idx}]: ").strip()
        try:
            choice = int(raw)
            if choice == custom_idx:
                bs = int(input("  Enter batch size: ").strip())
                return bs, n_train_patches % bs, 'custom'
            elif 1 <= choice <= len(menu_items):
                bs, rem = menu_items[choice - 1]
                return bs, rem, 'ok'
        except (ValueError, IndexError):
            pass
        console.print("[red]Invalid selection.[/red]")


if __name__ == "__main__":
    main()
