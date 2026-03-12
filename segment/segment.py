"""
segment/segment.py

Core segmentation logic for DeepAxon.
- Position-aware Hann window blending (9-position grid)
- 50% overlap patching
- Center crop
- Weak CLAHE for contrast standardisation (configurable)
- Per-patch normalisation matching original training pipeline
- BGW output (Black=background, Grey=axon, White=myelin)
- Saves cropped image to Cropped\ subfolder
"""

from __future__ import annotations

import csv
import time
import numpy as np
import cv2
from pathlib import Path
from patchify import patchify

from utils.resize import resize_img
from utils.console import DeepAxonLogger
from utils.helpers import load_config


# ─── Hann window (position-aware) ────────────────────────────────────────────

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
    """BGW: 0=black, 1=grey(128), 2=white(255)"""
    out = np.zeros(pred.shape, dtype=np.uint8)
    out[pred == 1] = 128
    out[pred == 2] = 255
    return np.stack([out, out, out], axis=-1)


# ─── Normalise (per-patch, matching original training pipeline) ───────────────

def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """
    Per-patch min-max normalisation to [0, 1].
    Matches keras.utils.normalize() behaviour from original code.
    """
    p_min = patch.min()
    p_max = patch.max()
    if p_max - p_min > 0:
        return (patch - p_min) / (p_max - p_min)
    return np.zeros_like(patch, dtype=np.float32)


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


# ─── Center crop ──────────────────────────────────────────────────────────────

def center_crop(img, patch_size):
    h, w = img.shape[:2]
    crop_h = (h // patch_size) * patch_size
    crop_w = (w // patch_size) * patch_size
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    return img[start_h:start_h + crop_h, start_w:start_w + crop_w]


# ─── Segment single image ─────────────────────────────────────────────────────

def segment_image(img_path, model, patch_size=256, cropped_dir=None, log=None):
    t0 = time.time()
    step = patch_size // 2  # 50% overlap

    # Resize
    img = resize_img(img_path, is_mask=False)

    # Center crop
    img_crop = center_crop(img, patch_size)
    crop_h, crop_w = img_crop.shape[:2]

    # Save cropped image
    if cropped_dir:
        Path(cropped_dir).mkdir(parents=True, exist_ok=True)
        stem = Path(img_path).stem
        cv2.imwrite(str(Path(cropped_dir) / f"{stem}_cropped.tif"), img_crop)

    # CLAHE — applied to full cropped image before patchifying
    img_clahe = apply_clahe(img_crop)

    # Patchify (still uint8 at this point)
    patches = patchify(img_clahe, (patch_size, patch_size), step=step)
    n_rows, n_cols = patches.shape[:2]

    pred_img = np.zeros((crop_h, crop_w), dtype=float)

    for i in range(n_rows):
        for j in range(n_cols):
            patch = patches[i, j].astype(np.float32)

            # Per-patch normalisation — matches original training pipeline
            patch_norm = normalize_patch(patch)
            patch_input = np.expand_dims(patch_norm, axis=(0, -1))  # (1, H, W, 1)

            pred = model.predict(patch_input, verbose=0)
            pred_cls = np.argmax(pred, axis=-1)[0]  # (H, W)

            pos = get_pos((n_rows, n_cols), i, j)
            hann = hann_window(pos, patch_size)
            adj = pred_cls * hann

            i_start = i * step
            j_start = j * step
            pred_img[i_start:i_start + patch_size, j_start:j_start + patch_size] += adj

    result = recolor(np.round(pred_img).astype(int))
    elapsed = time.time() - t0
    return result, elapsed, crop_w


# ─── Segment directory ────────────────────────────────────────────────────────

def segment_dir(tiff_dir, output_dir, model, mag, log, timing_csv=None):
    from utils.resize import get_image_resolution
    from utils.helpers import list_files

    config = load_config()
    patch_size = config.get("patch_size", {}).get(mag, 256)
    cropped_folder = config.get("cropped_folder", "Cropped")

    tiff_dir = Path(tiff_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir = tiff_dir.parent / cropped_folder

    images = list_files(str(tiff_dir), extensions=('.tif', '.tiff'))
    if not images:
        log.warn(f"No TIFF images found in {tiff_dir}")
        return

    log.rule(f"SEGMENTING {tiff_dir.name}")
    log.info(f"Found {len(images)} image(s) | patch_size={patch_size}px | overlap=50% | CLAHE=ON")

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
        if input("Continue anyway? [y/N]: ").strip().lower() != 'y':
            log.warn("Aborted.")
            return
    else:
        res = list(unique_res)[0] if unique_res else ('?', '?')
        log.info(f"Original resolution: {res[0]}x{res[1]} px")
        crop_w = (1440 // patch_size) * patch_size
        crop_h = (1024 // patch_size) * patch_size
        log.info(f"Center crop: {crop_w}x{crop_h} px")

    timing_rows = []
    success = 0
    failed = 0

    for img_path in images:
        stem = img_path.stem
        out_path = output_dir / f"{stem}_segmented.tif"
        res_str = "x".join(str(x) for x in resolutions.get(img_path.name, ('?', '?')))

        try:
            mask, elapsed, crop_w = segment_image(
                str(img_path), model,
                patch_size=patch_size,
                cropped_dir=str(cropped_dir),
                log=log
            )
            cv2.imwrite(str(out_path), cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
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
