"""
segment/segment.py

Core segmentation logic for DeepAxon.
- Position-aware Hann window blending (9-position grid)
- 50% overlap patching
- Center crop
- Weak CLAHE for contrast standardisation (configurable)
- Per-patch normalisation matching original training pipeline (keras.utils.normalize, axis=1)
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

from utils.resize import resize_img, get_image_resolution
from utils.logger import DeepAxonLogger
from utils.helpers import load_config, list_files, center_crop


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
    s = np.zeros((patch_size, patch_size), dtype=float)
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
    config = load_config()
    clahe_cfg = config.get("clahe", {})
    clip_limit = clahe_cfg.get("clip_limit", 1.5)
    tile_size = clahe_cfg.get("tile_grid_size", [8, 8])
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tuple(tile_size)
    )
    return clahe.apply(img)


# ─── Segment single image ─────────────────────────────────────────────────────

def segment_image(img_path, model, patch_size=256, cropped_dir=None, log=None, use_clahe=False):
    t0 = time.time()
    step = patch_size // 2  # 50% overlap

    # Resize
    img = resize_img(img_path, is_mask=False)

    # Center crop
    img_crop = center_crop(img, patch_size)
    crop_h, crop_w = img_crop.shape[:2]
    
    #SAVE (OVERWRITE ORIGINAL) img_crop saved at img_path

    # CLAHE — applied to full cropped image before patchifying if enabled
    if use_clahe:
        img_to_patch = apply_clahe(img_crop)
    else:
        img_to_patch = img_crop

    # Patchify (still uint8 at this point)
    patches = patchify(img_to_patch, (patch_size, patch_size), step=step)
    n_rows, n_cols = patches.shape[:2]

    pred_img = np.zeros((crop_h, crop_w), dtype=float)

    for i in range(n_rows):
        for j in range(n_cols):
            patch = patches[i, j, :, :]
            # L2 normalisation along axis=1 — matches original DeepAxon training pipeline (v1 train.py).
            # Must stay in sync with train/data/data_loader.py load_patches() normalization.
            # Do NOT change without retraining all models.
            norms = np.linalg.norm(patch, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # avoid division by zero
            patch = patch / norms
            patch = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            patch = patch.to(next(model.parameters()).device)
            model.eval()
            with torch.no_grad():
                pred = model(patch)                     # (1, 3, H, W) logits
            pred = pred.argmax(dim=1).squeeze(0)        # (H, W)
            pred = pred.cpu().numpy()

            pos = get_pos((n_rows, n_cols), i, j)
            hann = hann_window(pos, patch_size)
            adj = pred * hann

            i_start = i * step
            j_start = j * step
            pred_img[i_start:i_start + patch_size, j_start:j_start + patch_size] += adj

    pred_img = np.round(pred_img).astype(int)
    result = recolor(pred_img)

    elapsed = time.time() - t0
    return result, elapsed, crop_w


# ─── Segment directory ────────────────────────────────────────────────────────

def segment_dir(tiff_dir, output_dir, model, mag, log, timing_csv=None):
    config = load_config()
    patch_size     = config.get("patch_size", {}).get(mag, 256)
    cropped_folder = config.get("cropped_folder", "Cropped")
    seg_suffix     = config.get("segmented_suffix", "_segmented")

    tiff_dir   = Path(tiff_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir = tiff_dir.parent / cropped_folder

    images = list_files(str(tiff_dir), extensions=('.tif', '.tiff'))
    if not images:
        log.warn(f"No TIFF images found in {tiff_dir}")
        return

    log.rule(f"SEGMENTING {tiff_dir.name}")
    clahe_cfg  = config.get("clahe", {})
    clahe_on   = clahe_cfg.get("enabled", False)
    logging_on = config.get("logging", False)
    timing_on  = config.get("timing", False)
    log.info(
        f"Found {len(images)} image(s) | patch_size={patch_size}px | "
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
        crop_w = (1440 // patch_size) * patch_size
        crop_h = (1024 // patch_size) * patch_size
        log.info(f"Original resolution: {res[0]}x{res[1]} px")
        log.info(f"Center crop: {crop_w}x{crop_h} px")

    timing_rows = []
    success = 0
    failed  = 0

    for img_path in images:
        stem     = img_path.stem
        out_path = output_dir / f"{stem}{seg_suffix}.tif"
        res_str  = "x".join(str(x) for x in resolutions.get(img_path.name, ('?', '?')))

        try:
            mask, elapsed, crop_w = segment_image(
                str(img_path), model,
                patch_size=patch_size,
                cropped_dir=str(cropped_dir),
                log=log,
                use_clahe=clahe_on
            )
            cv2.imwrite(str(out_path), mask)
            log.success(f"{img_path.name} [{res_str}] -> {elapsed:.1f}s")
            timing_rows.append({
                'image': img_path.name, 'resolution': res_str,
                'crop_width': crop_w, 'patch_size': patch_size,
                'time_s': f"{elapsed:.2f}", 'status': 'ok'
            })
            success += 1
        except Exception as e:
            log.error(f"{img_path.name} - FAILED: {e}")
            timing_rows.append({
                'image': img_path.name, 'resolution': res_str,
                'crop_width': '', 'patch_size': patch_size,
                'time_s': '', 'status': f'FAILED: {e}'
            })
            failed += 1

    if timing_csv:
        with open(timing_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'image', 'resolution', 'crop_width', 'patch_size', 'time_s', 'status'
            ])
            writer.writeheader()
            writer.writerows(timing_rows)

    log.rule()
    log.success(f"Done - {success} succeeded, {failed} failed")