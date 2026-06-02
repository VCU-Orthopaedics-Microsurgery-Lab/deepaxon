"""
train/__main__.py

Entry point for: python -m train

For fine-tuning an existing model on new images:
    python -m train.finetune

Interactive mode (default):
    python -m train
    Handles all user prompts, GPU setup, then calls train.train_model().

Non-interactive mode (sbatch / HPRC):
    python -m train --config train_config.json
    Reads all settings from JSON file — no prompts. Used with train.sbatch.

Prompt order (interactive only):
    1. GPU setup
    2. Images folder
    3. val_ detection automatic
    4. Magnification
    5. Model name
    6. Epoch limit
    7. Augmentation
    8. Batch size menu
    9. Log setup + run
"""

import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
import json
import argparse
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
from train.train import train_model

console  = Console()
has_gpu  = torch.cuda.is_available()

MAG_OPTIONS       = ['40X', '100X']
PATCHES_PER_IMAGE = 63  # at 50% overlap on 1440px images


def _format_bs_line(bs: int, remainder: int, n_patches: int, suffix: str = "") -> str:
    if remainder == 0:
        n_batches = n_patches // bs
        return f"bs={bs:<4} — {n_patches} patches | {n_batches} full batches (perfect fit){suffix}"
    else:
        n_full   = n_patches // bs
        fullness = int(remainder / bs * 100)
        return (
            f"bs={bs:<4} — {n_patches} patches | "
            f"{n_full} full batches + {remainder}/{bs} remainder ({fullness}% full){suffix}"
        )


def _format_trim_line(bs: int, n_dropped: int, pct_dropped: float, n_patches: int) -> str:
    n_kept    = n_patches - n_dropped
    n_batches = n_kept // bs
    return (
        f"bs={bs:<4} — {n_kept} patches | "
        f"{n_batches} full batches ({n_dropped} patches dropped, {pct_dropped}%)"
    )


def _evaluate_custom_bs(bs: int, n_patches: int) -> tuple[str, str]:
    config    = load_config()
    train_cfg = config.get("training", {})
    min_ok    = train_cfg.get("min_last_batch_fullness",    0.80)
    min_warn  = train_cfg.get("danger_last_batch_fullness", 0.15)

    if bs < 1:
        return 'invalid', "Batch size must be ≥ 1."
    if n_patches < bs * 2:
        return 'invalid', f"Need at least {bs * 2} patches for 2 full batches — you have {n_patches}."
    remainder = n_patches % bs
    if remainder == 0:
        return 'perfect', _format_bs_line(bs, 0, n_patches)
    fullness = remainder / bs
    if fullness >= min_ok:
        return 'acceptable', _format_bs_line(bs, remainder, n_patches)
    elif fullness >= min_warn:
        return 'excluded', _format_bs_line(bs, remainder, n_patches)
    else:
        n_kept = n_patches - remainder
        return 'danger', _format_trim_line(bs, remainder, round(remainder/n_patches*100, 1), n_patches)


def select_batch_size(n_train_patches: int, n_val_patches: int, use_gpu: bool) -> tuple[int, int, str]:
    config    = load_config()
    train_cfg = config.get("training", {})
    min_ok    = train_cfg.get("min_last_batch_fullness",    0.80)
    min_warn  = train_cfg.get("danger_last_batch_fullness", 0.15)

    opts      = compute_batch_options(n_train_patches, use_gpu=use_gpu)
    ideal_str = " or ".join(str(b) for b in opts['ideal'])
    menu_items = []

    # ── Device panel ──────────────────────────────────────────────────────────
    t = Text(justify="center")
    t.append(f"{opts['device_label']}\n", style="orange1\n")
    t.append(f"Ideal batch: {ideal_str}\n")
    t.append(f"\nTraining patches: {n_train_patches}  |  Total patches: {n_train_patches + n_val_patches}")
    print_panel(console, Panel(
        t,
        title="[bold orange1]Batch Size Selection[/bold orange1]",
        border_style="orange1",
        box=DOUBLE,
        expand=True
    ))

    # ── Options ───────────────────────────────────────────────────────────────
    if opts['acceptable']:
        console.print(f"[green]✅ Power-of-2 — Acceptable (≥{int(min_ok*100)}% last batch full):[/green]")
        for bs, remainder in opts['acceptable']:
            idx  = len(menu_items) + 1
            line = _format_bs_line(bs, remainder, n_train_patches)
            console.print(f"  [{idx}]  {line}")
            menu_items.append((idx, bs, remainder, 'perfect' if remainder == 0 else 'acceptable'))

    if opts['perfect_fits']:
        console.print(f"\n  [green]✅ Perfect fit — no remainder (non power-of-2, marginally slower):[/green]")
        for bs in opts['perfect_fits']:
            idx  = len(menu_items) + 1
            line = _format_bs_line(bs, 0, n_train_patches)
            console.print(f"  [{idx}]  {line}")
            menu_items.append((idx, bs, 0, 'perfect'))

    if opts['trim']:
        console.print(f"\n  [cyan]✅ Trim to perfect fit (<{int(min_warn*100)}% last batch — drops remainder):[/cyan]")
        for bs, n_dropped, pct_dropped in opts['trim']:
            idx  = len(menu_items) + 1
            line = _format_trim_line(bs, n_dropped, pct_dropped, n_train_patches)
            console.print(f"  [{idx}]  {line}")
            menu_items.append((idx, bs, 0, 'trim'))

    if opts['excluded']:
        excl_str = ", ".join(f"bs={bs} ({pct}%)" for bs, _, pct in opts['excluded'])
        console.print(f"\n  [yellow]⚠  Excluded ({int(min_warn*100)}–{int(min_ok*100)}% last batch): {excl_str}[/yellow]")

    custom_idx = len(menu_items) + 1
    console.print(f"\n  [{custom_idx}]  Enter custom batch size\n")

    while True:
        raw = input(f"Select [1-{custom_idx}]: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            console.print("[red]Please enter a number.[/red]")
            continue

        if choice == custom_idx:
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
                        f"[yellow]  ⚠ Last batch {fullness}% full "
                        f"({int(min_warn*100)}–{int(min_ok*100)}% excluded zone). "
                        f"Strongly recommend choosing a different size.[/yellow]"
                    )
                    if get_yes_no("  Proceed anyway?", default=False):
                        return custom_bs, remainder, 'custom_warn'
                elif status == 'danger':
                    n_dropped = n_train_patches % custom_bs
                    pct       = round(n_dropped / n_train_patches * 100, 1)
                    console.print(
                        f"[red]  ⚠ Last batch <{int(min_warn*100)}% full. "
                        f"{n_dropped} patches ({pct}%) will be dropped.[/red]"
                    )
                    if get_yes_no("  Proceed with trim?", default=False):
                        return custom_bs, 0, 'custom_warn'

        elif 1 <= choice <= len(menu_items):
            _, bs, remainder, status = menu_items[choice - 1]
            return bs, remainder, status
        else:
            console.print(f"[red]Please enter a number between 1 and {custom_idx}.[/red]")


# ── Non-interactive config loader ─────────────────────────────────────────────

def _load_run_config(config_path: str) -> dict:
    """
    Load and validate a train_config.json for non-interactive (sbatch) mode.
    Raises ValueError on missing or invalid fields.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        cfg = json.load(f)

    required = ['images_dir', 'mag', 'epochs', 'augmentation', 'batch_size']
    # Analysis fields — optional for interactive runs, required for Wave 1/2
    analysis_fields = ['arch', 'encoder', 'n_images', 'val_fraction', 'seed', 'run_id']  
    is_analysis_run = any(k in cfg for k in analysis_fields)       
    if is_analysis_run:                                             
        missing_analysis = [k for k in analysis_fields if k not in cfg]  
        if missing_analysis:                                        
            raise ValueError(                                       
                f"Analysis run detected but missing fields: {missing_analysis}"  
            )   
              
    missing  = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"train_config.json missing required fields: {missing}")

    if cfg['mag'] not in MAG_OPTIONS:
        raise ValueError(f"Invalid mag '{cfg['mag']}' — must be one of {MAG_OPTIONS}")
    if not isinstance(cfg['epochs'], int) or cfg['epochs'] < 1:
        raise ValueError(f"'epochs' must be a positive integer, got: {cfg['epochs']}")
    if not isinstance(cfg['batch_size'], int) or cfg['batch_size'] < 1:
        raise ValueError(f"'batch_size' must be a positive integer, got: {cfg['batch_size']}")
    if not os.path.isdir(cfg['images_dir']):
        raise FileNotFoundError(f"images_dir not found: {cfg['images_dir']}")
    return cfg

def main():
    # ── Argument parsing ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="DeepAxon training entry point. "
                    "To fine-tune an existing model: python -m train.finetune"
    )
    parser.add_argument(
        '--config', metavar='FILE',
        help='Path to train_config.json for non-interactive (sbatch) mode'
    )
    args = parser.parse_args()

    print_panel(console, Panel(
        Align.center("[bold white]UNet++ Nerve Segmentation Model Training[/bold white]"),
        title="[bold cyan]DEEPAXON — TRAIN[/bold cyan]",
        border_style="bright_cyan",
        box=DOUBLE,
        expand=True,
        padding=(1, 4)
    ))

    # ── Non-interactive (sbatch) mode ─────────────────────────────────────────
    if args.config:
        console.print(f"[dim]Non-interactive mode — loading config from: {args.config}[/dim]\n")

        run_cfg    = _load_run_config(args.config)
        images_dir = run_cfg['images_dir']
        mag        = run_cfg['mag']
        use_aug    = bool(run_cfg['augmentation'])
        epochs     = int(run_cfg['epochs'])
        batch_size = int(run_cfg['batch_size'])

        images_path = Path(images_dir).resolve()
        
        if 'run_id' in run_cfg:                                          
            model_dir = Path(run_cfg['output']['models_dir'])            
        else:                                                            
            model_dir = get_model_dir(str(images_path))  
        
        if 'run_id' in run_cfg:                                          
            log_dir = Path(run_cfg['output']['logs_dir'])                
        else:                                                            
            log_dir = get_log_dir(str(images_path))
                                 
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if 'run_id' in run_cfg:                                          
            model_name = run_cfg['run_id']                               
        else:                                                            
            model_name = run_cfg.get('model_name', '').strip() or f"{mag.lower()}_{timestamp}"  

        log_path = str(log_dir / f"{model_name}_training_log.txt")
        log      = DeepAxonLogger(log_path=log_path, program="DeepAxon Train")
        
        from train.dataset.split import (
            build_image_manifest, stratified_split, split_summary
        )

        is_analysis = 'run_id' in run_cfg
        if is_analysis:
            manifest = build_image_manifest(
                images_dir   = run_cfg['images_dir'] + '/images',
                ctrl_prefix  = run_cfg.get('ctrl_prefix',  'ctrl_'),
                regen_prefix = run_cfg.get('regen_prefix', 'regen_'),
            )
            train_manifest, val_manifest = stratified_split(
                manifest     = manifest,
                n_images     = run_cfg['n_images'],
                val_fraction = run_cfg['val_fraction'],
                seed         = run_cfg['seed'],
            )
            summary = split_summary(train_manifest, val_manifest)
            run_cfg['train_stems'] = summary['train_stems']
            run_cfg['val_stems']   = summary['val_stems']
            run_cfg['_train_stems'] = summary['train_stems']
            run_cfg['_val_stems']   = summary['val_stems']
            log.rule("STRATIFIED SPLIT")
            log.info(
                f"Split: {summary['train_images']} train "
                f"({summary['train_ctrl']} ctrl / {summary['train_regen']} regen) | "
                f"{summary['val_images']} val "
                f"({summary['val_ctrl']} ctrl / {summary['val_regen']} regen) | "
                f"seed={run_cfg['seed']}"
            )
        
        log.rule("RUN CONFIGURATION")
        log.log_dict({
            'Mode':               'non-interactive (--config)',
            'Config file':        args.config,
            'Model name':         model_name,
            'Magnification':      mag,
            'Epoch limit':        epochs,
            'Augmentation':       'ON' if use_aug else 'OFF',
            'Batch size':         batch_size,
            'Images folder':      str(images_path),
            'Models folder':      str(model_dir),
            'Log file':           log_path,
        })

        train_model(
            images_dir = run_cfg['images_dir'],
            model_name = model_name,  
            epochs     = run_cfg['epochs'],
            batch_size = run_cfg['batch_size'],
            use_aug    = run_cfg['augmentation'],
            log        = log,
            mag        = run_cfg['mag'],
            arch       = run_cfg.get('arch',    'unet++'),              
            encoder    = run_cfg.get('encoder', 'resnet34'),            
            run_cfg    = run_cfg if is_analysis else None,              
        )

        console.print(f"\n[dim]Log saved to: {log_path}[/dim]")
        return

    # ── Interactive mode ──────────────────────────────────────────────────────
    setup_gpu_console()

    # ── Images folder ─────────────────────────────────────────────────────────
    while True:
        images_dir = input("Input the path to the training images folder: ").strip().strip('"')
        if os.path.isdir(images_dir):
            break
        console.print(f"[red]Folder not found: {images_dir} — please try again.[/red]")

    images_path = Path(images_dir).resolve()
    images_sub  = images_path / "images"
    
    model_dir   = get_model_dir(str(images_path))                   
    log_dir     = get_log_dir(str(images_path))                     
    log_dir.mkdir(parents=True, exist_ok=True)                      

    console.print("[yellow]Val set: auto-detected from val_ prefixed images (or 20% random split if none found)[/yellow]")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Load config ───────────────────────────────────────────────────────────
    config    = load_config()
    train_cfg = config.get("training", {})
    aug_cfg   = config.get("augmentation", {})
    prob_cfg  = aug_cfg.get("probabilities", {})
    param_cfg = aug_cfg.get("parameters", {})

    geo_prob   = prob_cfg.get("geometric_prob",  0.5)
    photo_prob = prob_cfg.get("photometric_prob", 0.25)

    # ── Magnification ─────────────────────────────────────────────────────────
    raw = input("\nSelect imaging magnification — [1] 40X  [2] 100X: ").strip()
    while raw not in ('1', '2'):
        raw = input("Invalid — [1] 40X  [2] 100X: ").strip()
    mag = MAG_OPTIONS[int(raw) - 1]

    # ── Model name ────────────────────────────────────────────────────────────
    auto_name  = f"{mag.lower()}_{timestamp}"
    console.print(f"\n[dim]Auto-generated name: {auto_name}[/dim]")
    custom     = input("Enter model name (press Enter to use auto-generated): ").strip()
    model_name = custom if custom else auto_name

    # ── Epoch limit ───────────────────────────────────────────────────────────
    default_epochs = train_cfg.get("epochs", 200)
    epochs = get_int_input(
        f"Epochs (press Enter for default={default_epochs}): ",
        default=default_epochs, min_val=1
    )

    # ── Augmentation ──────────────────────────────────────────────────────────
    expected_pct = round((1 - (1 - geo_prob)**3 * (1 - photo_prob)**3) * 100, 1)
    t = Text(justify="center")
    t.append(f"Geometric prob: {geo_prob:.2f}    |    Photometric prob: {photo_prob:.2f}\n", style="orange1")
    t.append(
        f"Flips, rotation ±{param_cfg.get('rotation_deg', 15)}°, "
        f"brightness {param_cfg.get('brightness_range', [0.8, 1.2])}, "
        f"gamma {param_cfg.get('gamma_range', [0.7, 1.4])}, "
        f"noise σ={param_cfg.get('noise_sigma', 0.02)}\n"
    )
    t.append(f"Expected augmentation rate: ~{expected_pct}% of patches", style="orange1")
    print_panel(console, Panel(
        t,
        title="[bold orange1]Data Augmentation Settings[/bold orange1]",
        border_style="orange1",
        box=DOUBLE,
        expand=True
    ))
    use_aug = get_yes_no("Use data augmentation?", default=False)

    # ── Batch size menu ───────────────────────────────────────────────────────
    patches_dir = images_sub / 'cropped' / 'patches'
    if patches_dir.exists():
        all_patch_files = list_files(str(patches_dir), extensions=('.png', '.tif', '.tiff'))
        n_train_patches = len([
            f for f in all_patch_files
            if not f.stem.startswith('val_')
        ])
        n_val_patches = len([
            f for f in all_patch_files
            if f.stem.startswith('val_')
        ])
    else:
        n_train_images = len([
            f for f in list_files(str(images_sub), extensions=('.tif', '.tiff', '.png', '.bmp'))
            if not Path(f).stem.lower().startswith('val_')
        ])
        n_val_images = len([
            f for f in list_files(str(images_sub), extensions=('.tif', '.tiff', '.png', '.bmp'))
            if Path(f).stem.lower().startswith('val_')
        ])
        n_train_patches = n_train_images * PATCHES_PER_IMAGE
        n_val_patches   = n_val_images   * PATCHES_PER_IMAGE

    batch_size, remainder, bs_status = select_batch_size(n_train_patches, n_val_patches, use_gpu=has_gpu)
    
    # ── Log setup ─────────────────────────────────────────────────────────────
    log_path = str(log_dir / f"{model_name}_training_log.txt")
    log      = DeepAxonLogger(log_path=log_path, program="DeepAxon Train")

    if remainder == 0:
        bs_log = f"{batch_size} (perfect fit)"
    else:
        bs_log = (
            f"{batch_size} "
            f"(last batch {remainder}/{batch_size} = {int(remainder/batch_size*100)}% full)"
        )
    if bs_status == 'custom_warn':
        bs_log += " [user override — outside recommended zones]"

    log.rule("RUN CONFIGURATION")
    log.log_dict({
        'Model name':         model_name,
        'Magnification':      mag,
        'Epoch limit':        epochs,
        'Augmentation':       'ON' if use_aug else 'OFF',
        'Batch size':         bs_log,
        'Est. train patches': f"~{n_train_patches}",
        'Images folder':      str(images_path),
        'Models folder':      str(model_dir),
        'Log file':           log_path,
    })

    # ── Run training ──────────────────────────────────────────────────────────
    train_model(
        images_dir=str(images_path),
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        use_aug=use_aug,
        log=log,
        mag=mag,
    )

    console.print(f"\n[dim]Log saved to: {log_path}[/dim]")


if __name__ == "__main__":
    main()
