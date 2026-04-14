"""
segment/segment.py

Core segmentation logic for DeepAxon.
- Position-aware Hann window blending (9-position grid)
- 50% overlap patching
- Center crop
- Weak CLAHE for contrast standardisation (configurable)
- Per-patch normalisation matching original training pipeline (L2 axis=1)
- BGW output (Black=background, Grey=myelin, White=axon)
"""

from __future__ import annotations

import csv
import time
import numpy as np
import cv2
from pathlib import Path
from patchify import patchify
import torch
import segmentation_models_pytorch as smp

from utils.resize import resize_img, get_image_resolution
from utils.logger import DeepAxonLogger
from utils.helpers import load_config, list_files, center_crop, get_hann_compatible_step
from torch.serialization import add_safe_globals
from torch.torch_version import TorchVersion

add_safe_globals([TorchVersion])


# ─── Hann window (position-aware) ────────────────────────────────────────────
# 9-position grid: corners (0,2,6,8), edges (1,3,5,7), center (4)
# Each position gets an asymmetric taper weight so that edge and corner patches
# only blend toward the image interior, not toward the image boundary.
# _hann_fn maps pixel index → weight in [0, 1] using a cosine taper.

def _hann_fn(x):
    return (1 - np.cos(2 * np.pi * x / 255)) / 2


def get_pos(shape, i, j):
    i_max = shape[0] - 1
    j_max = shape[1] - 1
    if   i == 0     and j == 0:     return 0
    elif i == 0     and j == j_max: return 2
    elif i == i_max and j == 0:     return 6
    elif i == i_max and j == j_max: return 8
    elif i == 0:                    return 1
    elif i == i_max:                return 7
    elif j == 0:                    return 3
    elif j == j_max:                return 5
    else:                           return 4


def hann_window(pos, patch_size=256):
    half = patch_size // 2
    i, j = np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='ij')
    c1 = (i <= half) & (j <= half)
    c2 = (i > half)  & (j <  half)
    c3 = (i <  half) & (j >  half)
    c4 = ~c1 & ~c2 & ~c3
    s  = np.zeros((patch_size, patch_size), dtype=float)
    hi = _hann_fn(i.astype(float))
    hj = _hann_fn(j.astype(float))
    if pos == 0:
        s[c1]=1;             s[c2]=hi[c2];          s[c3]=hj[c3];          s[c4]=hi[c4]*hj[c4]
    elif pos == 1:
        s[c1]=hj[c1];        s[c2]=hi[c2]*hj[c2];   s[c3]=hj[c3];          s[c4]=hi[c4]*hj[c4]
    elif pos == 2:
        s[c1]=hj[c1];        s[c2]=hi[c2]*hj[c2];   s[c3]=1;               s[c4]=hi[c4]
    elif pos == 3:
        s[c1]=hi[c1];        s[c2]=hi[c2];           s[c3]=hi[c3]*hj[c3];   s[c4]=hi[c4]*hj[c4]
    elif pos == 4:
        s[c1]=hi[c1]*hj[c1]; s[c2]=hi[c2]*hj[c2];   s[c3]=hi[c3]*hj[c3];   s[c4]=hi[c4]*hj[c4]
    elif pos == 5:
        s[c1]=hi[c1]*hj[c1]; s[c2]=hi[c2]*hj[c2];   s[c3]=hi[c3];          s[c4]=hi[c4]
    elif pos == 6:
        s[c1]=hi[c1];        s[c2]=1;                s[c3]=hi[c3]*hj[c3];   s[c4]=hj[c4]
    elif pos == 7:
        s[c1]=hi[c1]*hj[c1]; s[c2]=hj[c2];          s[c3]=hi[c3]*hj[c3];   s[c4]=hj[c4]
    elif pos == 8:
        s[c1]=hi[c1]*hj[c1]; s[c2]=hj[c2];          s[c3]=hi[c3];          s[c4]=1
    return s


# ─── Recolor ──────────────────────────────────────────────────────────────────

def recolor(pred):
    """
    Map class indices to BGW pixel values.

    BGW contract: 0=background(black), 1→128=myelin(grey), 2→255=axon(white)
    morphometrics/morphometrics.py inRange() thresholds depend on this mapping.
    Update both files if class assignments ever change.
    """
    out = np.zeros(pred.shape, dtype=np.uint8)
    out[pred == 1] = 128
    out[pred == 2] = 255
    return np.stack([out, out, out], axis=-1)


# ─── CLAHE ────────────────────────────────────────────────────────────────────

def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Apply weak CLAHE for contrast standardisation.
    Parameters loaded from config.json.
    """
    config    = load_config()
    clahe_cfg = config.get("clahe", {})
    clip_limit = clahe_cfg.get("clip_limit", 1.5)
    tile_size  = clahe_cfg.get("tile_grid_size", [8, 8])
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tuple(tile_size)
    )
    return clahe.apply(img)


# ─── Model loader ─────────────────────────────────────────────────────────────

def load_model(
    model_path: str,
    device: torch.device,
    log: DeepAxonLogger = None
) -> tuple:
    """
    Load a DeepAxon .pt model file.
    Supports both legacy (raw state_dict) and v5 (dict with meta) formats.

    Returns:
        (model, meta) — meta is {} for legacy models
    """
    model_path = Path(model_path).resolve()
    checkpoint = torch.load(
        str(model_path),
        map_location=device,
        weights_only=True
    )

    # Detect format
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            meta = checkpoint.get('meta', {})
        else:
            # assume raw state_dict
            state_dict = checkpoint
            meta = {}
    else:
        state_dict = checkpoint
        meta = {}

    encoder_name = meta.get('encoder', 'resnet34')

    if 'encoder' not in meta:
        if log:
            log.warn("Missing encoder in meta → defaulting to resnet34")
    
    # Build model from metadata if available, else use defaults
    model = smp.UnetPlusPlus(
        encoder_name    = encoder_name,
        encoder_weights = None,
        in_channels     = meta.get('in_channels', 1),
        classes         = len(meta.get('classes', ['background', 'myelin', 'axon'])),
        activation      = None,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Log metadata
    if log and meta:
        log.rule("LOADING MODEL")
        log.info(f"Model      : {model_path.name}")
        log.info(
            f"Trained    : {meta.get('trained_date', '?')} on "
            f"{meta.get('gpu', '?')} ({meta.get('hostname', '?')})"
        )
        log.info(
            f"Mag        : {meta.get('magnification', '?')} | "
            f"Patch: {meta.get('patch_size', '?')}px"
        )
        log.info(
            f"Best epoch : {meta.get('best_epoch', '?')} | "
            f"Axon dice: {meta.get('best_axon_dice', float('nan')):.4f} | "
            f"Myelin dice: {meta.get('best_myelin_dice', float('nan')):.4f}"
        )
    elif log:
        log.rule("LOADING MODEL")
        log.info(f"Model      : {model_path.name}")
        log.warn("Legacy format — no embedded metadata")

    return model, meta


# ─── Segment single image (CROPPED VERSION) ─────────────────────────────────────────────────────

# def segment_image(img_path, model, patch_size=256, log=None, use_clahe=False):
#     t0   = time.time()
#     step = get_hann_compatible_step(patch_size)  # 50% overlap required by Hann blending

#     # Resize
#     img = resize_img(img_path, is_mask=False)

#     # Center crop
#     img_crop       = center_crop(img, patch_size)
#     crop_h, crop_w = img_crop.shape[:2]

#     # CLAHE — applied to full cropped image before patchifying if enabled
#     img_to_patch = apply_clahe(img_crop) if use_clahe else img_crop

#     # Patchify (still uint8 at this point)
#     patches        = patchify(img_to_patch, (patch_size, patch_size), step=step)
#     n_rows, n_cols = patches.shape[:2]

#     pred_img = np.zeros((crop_h, crop_w), dtype=float)

#     for i in range(n_rows):
#         for j in range(n_cols):
#             patch = patches[i, j, :, :]
#             # L2 normalisation along axis=1 — matches original DeepAxon training pipeline.
#             # Must stay in sync with train/dataset/data_loader.py load_patches() normalization.
#             # Do NOT change without retraining all models.
#             norms = np.linalg.norm(patch, axis=1, keepdims=True)
#             norms = np.where(norms == 0, 1, norms)
#             patch = patch / norms
#             patch = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
#             patch = patch.to(next(model.parameters()).device)
#             with torch.no_grad():
#                 pred = model(patch)              # (1, 3, H, W) logits
#             pred = pred.argmax(dim=1).squeeze(0) # (H, W)
#             pred = pred.cpu().numpy()

#             pos  = get_pos((n_rows, n_cols), i, j)
#             hann = hann_window(pos, patch_size)
#             adj  = pred * hann

#             i_start = i * step
#             j_start = j * step
#             pred_img[i_start:i_start + patch_size, j_start:j_start + patch_size] += adj

#     pred_img = np.round(pred_img).astype(int)
#     result   = recolor(pred_img)
#     elapsed  = time.time() - t0
#     return result, elapsed, crop_h, crop_w

# ─── Segment single image (PADDED VERSION) ─────────────────────────────────────────────────────

def segment_image(img_path, model, patch_size=256, log=None, use_clahe=False):
    t0   = time.time()
    step = get_hann_compatible_step(patch_size)

    # Resize
    img = resize_img(img_path, is_mask=False)
    orig_h, orig_w = img.shape[:2]

    # # ── Center crop (alternative to padding) ──────────────────────────────
    # img_proc = center_crop(img, patch_size)

    # ── Pad to next multiple of patch_size ────────────────────────────────
    pad_h    = (patch_size - orig_h % patch_size) % patch_size
    pad_w    = (patch_size - orig_w % patch_size) % patch_size
    img_proc = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    proc_h, proc_w = img_proc.shape[:2]

    # CLAHE — applied before patchifying if enabled
    img_to_patch = apply_clahe(img_proc) if use_clahe else img_proc

    # Patchify
    patches        = patchify(img_to_patch, (patch_size, patch_size), step=step)
    n_rows, n_cols = patches.shape[:2]

    pred_img = np.zeros((proc_h, proc_w), dtype=float)

    for i in range(n_rows):
        for j in range(n_cols):
            patch = patches[i, j, :, :]
            norms = np.linalg.norm(patch, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            patch = patch / norms
            patch = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
            patch = patch.to(next(model.parameters()).device)
            with torch.no_grad():
                pred = model(patch)
            pred = pred.argmax(dim=1).squeeze(0)
            pred = pred.cpu().numpy()

            pos  = get_pos((n_rows, n_cols), i, j)
            hann = hann_window(pos, patch_size)
            adj  = pred * hann

            i_start = i * step
            j_start = j * step
            pred_img[i_start:i_start + patch_size, j_start:j_start + patch_size] += adj

    pred_img = np.round(pred_img).astype(int)

    # Crop back to original dimensions before recoloring
    pred_img = pred_img[:orig_h, :orig_w]

    result  = recolor(pred_img)
    elapsed = time.time() - t0
    return result, elapsed, orig_h, orig_w

# ─── Save Overlays ────────────────────────────────────────────────────────
def save_qc_sheet(
    tiff_dir: Path,
    output_dir: Path,
    images: list,
    seg_suffix: str,
    model_name: str,
    log: DeepAxonLogger = None
):
    """
    Generate a QC sheet showing original / segmented / overlay for each image.
    Saved as {nerve_name}_qc_sheet.png in a QC/ subfolder of the nerve directory.

    Layout: one row per image, three columns — original | segmented | overlay
    Overlay: axon=pink, myelin=orange, background=transparent
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend — no display needed
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        if log:
            log.warn("matplotlib not available — skipping QC sheet")
        return

    config     = load_config()
    qc_folder  = config.get("qc_folder", "QC")
    nerve_name = tiff_dir.parent.name
    qc_dir     = tiff_dir.parent / qc_folder
    qc_dir.mkdir(parents=True, exist_ok=True)

    n_images = len(images)
    if n_images == 0:
        return

    fig, axes = plt.subplots(n_images, 3, figsize=(24, 8 * n_images))
    if n_images == 1:
        axes = [axes]  # ensure iterable rows

    fig.suptitle(
        f"{nerve_name}  |  model: {model_name}",
        fontsize=14, fontweight='bold', y=1.0
    )

    col_titles = ["Original", "Segmented", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=11, fontweight='bold', pad=8)

    for row, img_path in enumerate(images):
        seg_path = output_dir / f"{img_path.stem}{seg_suffix}.tif"

        # Load original
        orig = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if orig is None:
            continue
        orig = resize_img(str(img_path), is_mask=False)

        # Load segmented
        if not seg_path.exists():
            continue
        seg = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
        if seg is None:
            continue

        # Resize original to match seg dimensions
        if orig.shape != seg.shape:
            orig = cv2.resize(orig, (seg.shape[1], seg.shape[0]))

        # Build overlay — RGB
        overlay = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB).astype(np.float32)
        alpha   = 0.45

        # Axon → pink (255, 105, 180)
        axon_mask = seg == 255
        overlay[axon_mask] = (
            overlay[axon_mask] * (1 - alpha) +
            np.array([255, 105, 180], dtype=np.float32) * alpha
        )

        # Myelin → orange (255, 160, 100)
        myelin_mask = seg == 128
        overlay[myelin_mask] = (
            overlay[myelin_mask] * (1 - alpha) +
            np.array([255, 160, 100], dtype=np.float32) * alpha
        )
        overlay = overlay.astype(np.uint8)

        # Plot
        axes[row][0].imshow(orig, cmap='gray', vmin=0, vmax=255)
        axes[row][0].set_ylabel(img_path.name, fontsize=7, rotation=0,
                                labelpad=120, va='center')
        axes[row][0].axis('off')

        axes[row][1].imshow(seg, cmap='gray', vmin=0, vmax=255)
        axes[row][1].axis('off')

        axes[row][2].imshow(overlay)
        axes[row][2].axis('off')

    # Legend on last row overlay panel
    legend = [
        mpatches.Patch(color=(1.0, 0.41, 0.71), label='Axon'),
        mpatches.Patch(color=(1.0, 0.63, 0.39), label='Myelin'),
    ]
    axes[-1][2].legend(handles=legend, loc='lower right', fontsize=9, framealpha=0.7)

    plt.tight_layout()

    out_path = qc_dir / f"{nerve_name}_qc_sheet.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)

    if log:
        log.success(f"QC sheet saved → QC/{nerve_name}_qc_sheet.png")

# ─── Segment directory ────────────────────────────────────────────────────────

def segment_dir(tiff_dir, output_dir, model, mag, log, timing_csv=None, model_name='unknown'):
    config         = load_config()
    patch_size     = config.get("patch_size", {}).get(mag, 256)
    seg_suffix     = config.get("segmented_suffix", "_segmented")
    step           = get_hann_compatible_step(patch_size)
    overlap_pct    = 100 - int(step / patch_size * 100)

    tiff_dir   = Path(tiff_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_files(str(tiff_dir), extensions=('.tif', '.tiff'))
    if not images:
        log.warn(f"No TIFF images found in {tiff_dir}")
        return

    log.rule(f"SEGMENTING {tiff_dir.parent.name} / {tiff_dir.name}")
    clahe_cfg  = config.get("clahe", {})
    clahe_on   = clahe_cfg.get("enabled", False)
    timing_on  = config.get("timing", False)
    logging_cfg = config.get("logging", {})
    logging_on  = logging_cfg.get("segment", False) if isinstance(logging_cfg, dict) else bool(logging_cfg)

    log.info(
        f"Found {len(images)} image(s) | patch={patch_size}px | "
        f"step={step}px ({overlap_pct}% overlap) | "
        f"CLAHE={'ON' if clahe_on else 'OFF'} | "
        f"Logging={'ON' if logging_on else 'OFF'} | "
        f"Timing={'ON' if timing_on else 'OFF'}"
    )

    # Resolution check
    resolutions = {}
    for img_path in images:
        try:
            w, h = get_image_resolution(str(img_path))
            resolutions[img_path.name] = (w, h)
        except Exception as e:
            log.warn(f"Could not read resolution for {img_path.name}: {e}")

    unique_res = set(resolutions.values())
    if len(unique_res) > 1:
        log.warn("Resolution mismatch detected:")
        for name, res in resolutions.items():
            log.warn(f"  {name}: {res[0]}x{res[1]}")
        log.warn("Continuing with mismatched resolutions.")
    else:
        res    = list(unique_res)[0] if unique_res else ('?', '?')
        log.info(f"Original resolution: {res[0]}x{res[1]} px")

    timing_rows = []
    success     = 0
    failed      = 0

    for img_path in images:
        stem     = img_path.stem
        out_path = output_dir / f"{stem}{seg_suffix}.tif"
        res_str  = "x".join(str(x) for x in resolutions.get(img_path.name, ('?', '?')))

        try:
            mask, elapsed, crop_h, crop_w = segment_image(
                str(img_path), model,
                patch_size=patch_size,
                log=log,
                use_clahe=clahe_on
            )
            log.success(f"{img_path.name} [{crop_w}x{crop_h}] -> {elapsed:.1f}s")
            cv2.imwrite(str(out_path), mask)
            timing_rows.append({
                'image': img_path.name, 'resolution': res_str,
                'crop_size': f"{crop_w}x{crop_h}", 'patch_size': patch_size,
                'time_s': f"{elapsed:.2f}", 'status': 'ok'
            })
            success += 1
            
        except Exception as e:
            log.error(f"{img_path.name} - FAILED: {e}")
            timing_rows.append({
                'image': img_path.name, 'resolution': res_str,
                'crop_size': '', 'patch_size': patch_size,
                'time_s': '', 'status': f'FAILED: {e}'
            })
            failed += 1

    if timing_csv:
        with open(timing_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'image', 'resolution', 'crop_size', 'patch_size', 'time_s', 'status'
            ])
            writer.writeheader()
            writer.writerows(timing_rows)
            
    log.success(f"Done - {success} succeeded, {failed} failed")
    
    save_qc_sheet(
        tiff_dir=tiff_dir,
        output_dir=output_dir,
        images=images,
        seg_suffix=seg_suffix,
        model_name=model_name,
        log=log
    )