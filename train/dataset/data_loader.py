"""
train/dataset/data_loader.py

Load preprocessed patches from disk into numpy arrays for training.
Supports val_ prefix mode and random split mode.

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
        127, 128  → 1 (myelin / grey)
        255       → 2 (axon / white)
    """
    pixel_to_class = {0: 0, 127: 1, 128: 1, 255: 2}
    return np.vectorize(lambda x: pixel_to_class.get(int(x), 0))(arr).astype(np.float32)


# ─── Patch loading ────────────────────────────────────────────────────────────

def load_patches(patches_dir: str, is_mask: bool = False) -> np.ndarray:
    """
    Load all patches from a directory into a numpy array.
    Images: L2 normalization axis=1. Masks: base_label() remapping.
    Returns (N, H, W, 1) float32.
    """
    files = list_files(patches_dir, extensions=('.png', '.tif', '.tiff'))
    if not files:
        raise FileNotFoundError(f"No patch files found in {patches_dir}")

    patches = []
    for f in sorted(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Could not load patch, skipping: {f}")
            continue
        patches.append(img)

    arr = np.array(patches, dtype=np.float32)
    arr = np.expand_dims(arr, axis=-1)

    if is_mask:
        arr = base_label(arr)
    else:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        arr   = arr / norms

    return arr


# ── Recover source image stems from patch filenames ──────────────────────────
def _patches_to_image_stems(files: list) -> list[str]:
    """
    Extract unique source image stems from patch filenames.
    Drops the row/col suffix (last _XXXX segment).

    img_10_0304.png     → img_10
    val_img_11_0404.png → val_img_11
    """
    seen  = set()
    stems = []
    for f in files:
        stem = '_'.join(Path(f).stem.split('_')[:-1])  # drop last _RRCC segment
        if stem not in seen:
            seen.add(stem)
            stems.append(stem)
    return sorted(stems)
# ─────────────────────────────────────────────────────────────────────────────


def load_all_patches(
    img_train_dir: str,
    mask_train_dir: str,
    val_fraction: float = 0.2,
    log=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Load train and validation patches.

    val_ prefix mode (auto-detected):
        Patches prefixed with val_ → val set.
        All other patches → train set.

    Random split mode (no val_ patches found):
        Auto-split using val_fraction (default 0.2).

    Returns:
        (X_train, Y_train, X_val, Y_val, split_mode)
        Arrays are (N, H, W, 1) float32. split_mode is 'val_prefix' or 'random_Npct'.
    """
    img_train_dir  = Path(img_train_dir)
    mask_train_dir = Path(mask_train_dir)

    patches_img  = img_train_dir  / 'cropped' / 'patches'
    patches_mask = mask_train_dir / 'cropped' / 'patches'

    all_img_files  = list_files(str(patches_img),  extensions=('.png', '.tif', '.tiff'))
    all_mask_files = list_files(str(patches_mask), extensions=('.png', '.tif', '.tiff'))

    val_img_files = [f for f in all_img_files if f.stem.startswith('val_')]

    # ── val_ prefix mode ─────────────────────────────────────────────────────
    if val_img_files:
        train_img_files  = [f for f in all_img_files  if not f.stem.startswith('val_')]
        train_mask_files = [f for f in all_mask_files if not f.stem.startswith('val_')]
        val_mask_files   = [f for f in all_mask_files if f.stem.startswith('val_')]

        def _load_file_list(files, is_mask):
            patches = []
            for f in sorted(files):
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"[WARN] Could not load patch, skipping: {f}")
                    continue
                patches.append(img)
            arr = np.expand_dims(
                np.array(patches, dtype=np.float32), axis=-1
            )
            if is_mask:
                return base_label(arr)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return arr / norms

        X_train = _load_file_list(train_img_files,  is_mask=False)
        Y_train = _load_file_list(train_mask_files, is_mask=True)
        X_val   = _load_file_list(val_img_files,    is_mask=False)
        Y_val   = _load_file_list(val_mask_files,   is_mask=True)

        if log:
            train_stems = _patches_to_image_stems(train_img_files)
            val_stems   = _patches_to_image_stems(val_img_files)
            n_total     = len(train_stems) + len(val_stems)
            eff_pct     = round(len(val_stems) / n_total * 100, 1)
            log.info(
                f"Train: {len(train_stems)} images → {len(X_train)} patches  |  "
                f"Val: {len(val_stems)} images → {len(X_val)} patches ({eff_pct}%)"
            )
            log.info(f"Val images: {', '.join(val_stems)}")

        return X_train, Y_train, X_val, Y_val, 'val_prefix', val_stems

    # ── Random split mode ─────────────────────────────────────────────────────
    X = load_patches(str(patches_img),  is_mask=False)
    Y = load_patches(str(patches_mask), is_mask=True)

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
        val_stems   = _patches_to_image_stems(val_files)
        train_stems = _patches_to_image_stems(train_files)
        n_total     = len(val_stems) + len(train_stems)
        eff_pct     = round(len(val_stems) / n_total * 100, 1)
        log.info(
            f"Train: {len(train_stems)} images → {len(X_train)} patches  |  "
            f"Val: {len(val_stems)} images → {len(X_val)} patches ({eff_pct}%)"
        )
        log.info(f"Val images: {', '.join(val_stems)}")

    return X_train, Y_train, X_val, Y_val, f'random_{int(val_fraction*100)}pct', val_stems