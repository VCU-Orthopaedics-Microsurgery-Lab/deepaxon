"""
train/dataset/data_loader.py

Load preprocessed patches from disk into numpy arrays for training.

v5_analysis branch — val_ prefix mode removed.

Two load modes:
    Manifest mode (primary):
        train_stems + val_stems passed from split.py via run_cfg.
        Used by all Wave 1/2/3 analysis runs.
        Phenotype-balanced, seed-reproducible splits.

    Random split mode (fallback only):
        No stems passed — auto-split using val_fraction.
        Not used in analysis runs — retained as emergency fallback.
"""

from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split

from utils.helpers import list_files


# ─── Mask encoding ────────────────────────────────────────────────────────────

def base_label(arr: np.ndarray) -> np.ndarray:
    """
    Remap mask pixel values to class indices.
    Pixel value → class index:
        0         → 0 (background)
        127, 128  → 1 (myelin)
        255       → 2 (axon)
    """
    out = np.zeros_like(arr, dtype=np.float32)
    out[arr == 127] = 1
    out[arr == 128] = 1
    out[arr == 255] = 2
    return out


# ─── File list loader (module-level — shared by all modes) ───────────────────

def _load_file_list(files: list, is_mask: bool) -> np.ndarray:
    """
    Load a list of patch file paths into a (N, H, W, 1) float32 array.
    Images: L2 normalization axis=1.
    Masks:  base_label() class index remapping.
    """
    patches = []
    for f in sorted(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Could not load patch, skipping: {f}")
            continue
        patches.append(img)

    if not patches:
        raise ValueError(f"No patches loaded from {len(files)} files — all failed to read")

    arr = np.expand_dims(np.array(patches, dtype=np.float32), axis=-1)

    if is_mask:
        return base_label(arr)

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return arr / norms


# ─── Stem recovery ────────────────────────────────────────────────────────────

def _patch_stem(f: Path) -> str:
    """
    Recover source image stem from patch filename.
    Drops the last _RRCC segment.
    ctrl_animal1_0304.png → ctrl_animal1
    """
    return '_'.join(Path(f).stem.split('_')[:-1])


def _patches_to_image_stems(files: list) -> list[str]:
    """
    Extract ordered unique source image stems from a list of patch files.
    """
    seen  = set()
    stems = []
    for f in files:
        stem = _patch_stem(f)
        if stem not in seen:
            seen.add(stem)
            stems.append(stem)
    return stems


# ─── Main loader ──────────────────────────────────────────────────────────────

def load_all_patches(
    img_dir:      str,
    mask_dir:     str,
    val_fraction: float = 0.2,
    log=None,
    train_stems:  list[str] | None = None,
    val_stems:    list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, list]:
    """
    Load train and validation patches into numpy arrays.

    Mode 1 — Manifest (primary, analysis runs):
        train_stems and val_stems passed from split.py.
        Patches filtered by source image stem.
        split_mode returned: 'manifest'

    Mode 2 — Random split (fallback only):
        No stems passed.
        Auto-split by val_fraction with fixed seed 42.
        split_mode returned: 'random_Npct'

    Args:
        img_dir:      path to images_dir (parent of cropped/patches/)
        mask_dir:     path to masks_dir  (parent of cropped/patches/)
        val_fraction: val fraction for random split fallback (default 0.2)
        log:          DeepAxonLogger instance (optional)
        train_stems:  list of source image stems for train set (manifest mode)
        val_stems:    list of source image stems for val set   (manifest mode)

    Returns:
        (X_train, Y_train, X_val, Y_val, split_mode, val_stems)
        All arrays (N, H, W, 1) float32.
    """
    img_dir  = Path(img_dir)
    mask_dir = Path(mask_dir)

    patches_img  = img_dir  / 'cropped' / 'patches'
    patches_mask = mask_dir / 'cropped' / 'patches'

    all_img_files  = list_files(str(patches_img),  extensions=('.png', '.tif', '.tiff'))
    all_mask_files = list_files(str(patches_mask), extensions=('.png', '.tif', '.tiff'))

    if not all_img_files:
        raise FileNotFoundError(f"No image patches found in {patches_img}")
    if not all_mask_files:
        raise FileNotFoundError(f"No mask patches found in {patches_mask}")

    # ── Mode 1 — Manifest ─────────────────────────────────────────────────────
    if train_stems is not None and val_stems is not None:
        train_stems_set = set(train_stems)
        val_stems_set   = set(val_stems)

        train_img_files  = [f for f in all_img_files  if _patch_stem(f) in train_stems_set]
        train_mask_files = [f for f in all_mask_files if _patch_stem(f) in train_stems_set]
        val_img_files    = [f for f in all_img_files  if _patch_stem(f) in val_stems_set]
        val_mask_files   = [f for f in all_mask_files if _patch_stem(f) in val_stems_set]

        if not train_img_files:
            raise ValueError(
                f"Manifest mode: no patches found for train stems — "
                f"first 3: {train_stems[:3]}"
            )
        if not val_img_files:
            raise ValueError(
                f"Manifest mode: no patches found for val stems — "
                f"first 3: {val_stems[:3]}"
            )

        X_train = _load_file_list(train_img_files,  is_mask=False)
        Y_train = _load_file_list(train_mask_files, is_mask=True)
        X_val   = _load_file_list(val_img_files,    is_mask=False)
        Y_val   = _load_file_list(val_mask_files,   is_mask=True)

        if log:
            n_total = len(train_stems) + len(val_stems)
            eff_pct = round(len(val_stems) / n_total * 100, 1)
            log.info(f"Split mode:  manifest (stratified, phenotype-balanced)")
            log.info(f"Train: {len(train_stems)} images → {len(X_train)} patches")
            log.info(f"Val:   {len(val_stems)} images → {len(X_val)} patches ({eff_pct}%)")
            log.info(f"Val images: {', '.join(val_stems)}")

        return X_train, Y_train, X_val, Y_val, 'manifest', val_stems

    # ── Mode 2 — Random split fallback ────────────────────────────────────────
    if log:
        log.warn(
            "No manifest stems provided — falling back to random split. "
            "This should not occur in analysis runs."
        )

    X = _load_file_list(all_img_files,  is_mask=False)
    Y = _load_file_list(all_mask_files, is_mask=True)

    if len(X) != len(Y):
        raise ValueError(f"Patch count mismatch: {len(X)} images vs {len(Y)} masks")

    indices = list(range(len(all_img_files)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_fraction, random_state=42
    )

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val,   Y_val   = X[val_idx],   Y[val_idx]

    if log:
        val_files   = [all_img_files[i] for i in val_idx]
        train_files = [all_img_files[i] for i in train_idx]
        val_stems_r   = _patches_to_image_stems(val_files)
        train_stems_r = _patches_to_image_stems(train_files)
        n_total = len(val_stems_r) + len(train_stems_r)
        eff_pct = round(len(val_stems_r) / n_total * 100, 1)
        log.info(f"Split mode:  random_{int(val_fraction*100)}pct (seed=42)")
        log.info(
            f"Train: {len(train_stems_r)} images → {len(X_train)} patches  |  "
            f"Val: {len(val_stems_r)} images → {len(X_val)} patches ({eff_pct}%)"
        )
        log.info(f"Val images: {', '.join(val_stems_r)}")

    return X_train, Y_train, X_val, Y_val, f'random_{int(val_fraction*100)}pct', val_stems_r