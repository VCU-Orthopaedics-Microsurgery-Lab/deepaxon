"""
train/dataset/data_loader.py

Load preprocessed patches from disk into numpy arrays for training.

Supports two dataset split modes:
    Phenotype mode — explicit train/regen/, train/control/, val/regen/, val/control/
                     folders. User controls split. val_fraction ignored.
                     Warns if regen/control patch counts are imbalanced.
    Flat mode      — single train/ and val/ folders. val_fraction used to
                     auto-split if no val folder present.

Mode is detected by presence of regen/ and control/ subfolders.
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
    Matches original DeepAxon training pipeline (v1 train.py).

    Pixel value → class index:
        0         → 0 (background)
        127, 128  → 1 (myelin / grey)
        255       → 2 (axon / white)

    Args:
        arr: (N, H, W, 1) float32 mask array with raw pixel values

    Returns:
        (N, H, W, 1) float32 array with values 0, 1, or 2
    """
    pixel_to_class = {0: 0, 127: 1, 128: 1, 255: 2}
    return np.vectorize(lambda x: pixel_to_class.get(int(x), 0))(arr).astype(np.float32)


# ─── Patch loading ────────────────────────────────────────────────────────────

def load_patches(patches_dir: str, is_mask: bool = False) -> np.ndarray:
    """
    Load all patches from a directory into a numpy array.

    For images: L2 normalization along axis=1 — matches original DeepAxon
        training pipeline (v1 train.py). Must stay in sync with
        segment/segment.py segment_image() normalization.
        Do NOT change to /= 255.0 — existing models were trained with L2.
    For masks: remaps pixel values to class indices via base_label().

    Returns:
        (N, H, W, 1) float32 array
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
    arr = np.expand_dims(arr, axis=-1)  # (N, H, W, 1)

    if is_mask:
        arr = base_label(arr)
    else:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        arr   = arr / norms

    return arr


def _load_phenotype_patches(
    base_dir: Path,
    is_mask: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load patches from regen/ and control/ subfolders separately.

    Returns:
        (regen_patches, control_patches) — both (N, H, W, 1) float32
    """
    regen_dir   = base_dir / 'regen'   / 'cropped' / 'patches'
    control_dir = base_dir / 'control' / 'cropped' / 'patches'

    regen   = load_patches(str(regen_dir),   is_mask=is_mask)
    control = load_patches(str(control_dir), is_mask=is_mask)

    return regen, control


def _check_phenotype_balance(
    n_regen: int,
    n_control: int,
    split_name: str,
    warn_threshold: float = 0.2
) -> str | None:
    """
    Check regen/control patch balance. Returns warning string if imbalanced.
    warn_threshold: fractional difference above which a warning is issued.
    e.g. 0.2 = warn if counts differ by more than 20%.
    """
    if n_regen == 0 or n_control == 0:
        return f"{split_name}: one phenotype has 0 patches — check folder structure."
    total    = n_regen + n_control
    pct_diff = abs(n_regen - n_control) / total
    if pct_diff > warn_threshold:
        majority   = 'regen'   if n_regen > n_control else 'control'
        minority   = 'control' if n_regen > n_control else 'regen'
        pct_regen  = int(n_regen   / total * 100)
        pct_ctrl   = int(n_control / total * 100)
        return (
            f"{split_name}: imbalanced — {n_regen} regen ({pct_regen}%) vs "
            f"{n_control} control ({pct_ctrl}%). "
            f"Model may skew toward {majority}."
        )
    return None


# ─── Split mode detection ─────────────────────────────────────────────────────

def detect_split_mode(images_train_dir: Path) -> str:
    """
    Detect dataset split mode based on folder structure.

    Returns:
        'phenotype' if regen/ and control/ subfolders both exist
        'flat'      otherwise
    """
    regen_exists   = (images_train_dir / 'regen').exists()
    control_exists = (images_train_dir / 'control').exists()
    if regen_exists and control_exists:
        return 'phenotype'
    return 'flat'


# ─── Main loader ──────────────────────────────────────────────────────────────

def load_all_patches(
    img_train_dir: str,
    mask_train_dir: str,
    img_val_dir: str = None,
    mask_val_dir: str = None,
    val_fraction: float = 0.2,
    split_mode: str = 'flat',
    log=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train and validation patches.

    Phenotype mode:
        Loads from train/regen/cropped/patches/ and train/control/cropped/patches/.
        Concatenates both phenotypes. Warns if imbalanced.
        Val loaded from val/regen/ and val/control/ — val_fraction ignored.

    Flat mode:
        Loads from a single patches directory.
        If val directory exists and has images, uses it directly.
        Otherwise auto-splits using val_fraction.

    Args:
        img_train_dir:  Path to images train directory
        mask_train_dir: Path to masks train directory
        img_val_dir:    Path to images val directory (optional)
        mask_val_dir:   Path to masks val directory (optional)
        val_fraction:   Fraction for auto-split in flat mode (ignored in phenotype mode)
        split_mode:     'phenotype' or 'flat'
        log:            DeepAxonLogger instance for balance warnings

    Returns:
        (X_train, Y_train, X_val, Y_val) — all (N, H, W, 1) float32
        Masks are class indices (0/1/2).
    """
    img_train_dir  = Path(img_train_dir)
    mask_train_dir = Path(mask_train_dir)

    # ── Phenotype mode ────────────────────────────────────────────────────────
    if split_mode == 'phenotype':

        # Train
        X_regen,   Y_regen   = _load_phenotype_patches(img_train_dir,  is_mask=False), \
                                _load_phenotype_patches(mask_train_dir, is_mask=True)
        X_ctrl,    Y_ctrl    = X_regen[1], Y_regen[1]
        X_regen,   Y_regen   = X_regen[0], Y_regen[0]

        # Balance check — train
        warn = _check_phenotype_balance(len(X_regen), len(X_ctrl), 'Train')
        if warn:
            if log:
                log.warn(warn)
            else:
                print(f"[WARN] {warn}")

        X_train = np.concatenate([X_regen, X_ctrl], axis=0)
        Y_train = np.concatenate([Y_regen, Y_ctrl], axis=0)

        # Shuffle train so regen/control aren't in blocks
        idx     = np.random.permutation(len(X_train))
        X_train = X_train[idx]
        Y_train = Y_train[idx]

        # Val
        img_val_path  = Path(img_val_dir)  if img_val_dir  else None
        mask_val_path = Path(mask_val_dir) if mask_val_dir else None

        if img_val_path and mask_val_path:
            X_val_regen, Y_val_regen = _load_phenotype_patches(img_val_path,  is_mask=False), \
                                       _load_phenotype_patches(mask_val_path, is_mask=True)
            X_val_ctrl,  Y_val_ctrl  = X_val_regen[1], Y_val_regen[1]
            X_val_regen, Y_val_regen = X_val_regen[0], Y_val_regen[0]

            # Balance check — val
            warn_val = _check_phenotype_balance(len(X_val_regen), len(X_val_ctrl), 'Val')
            if warn_val:
                if log:
                    log.warn(warn_val)
                else:
                    print(f"[WARN] {warn_val}")

            X_val = np.concatenate([X_val_regen, X_val_ctrl], axis=0)
            Y_val = np.concatenate([Y_val_regen, Y_val_ctrl], axis=0)

            # Shuffle val
            idx_v = np.random.permutation(len(X_val))
            X_val = X_val[idx_v]
            Y_val = Y_val[idx_v]
        else:
            raise FileNotFoundError(
                "Phenotype mode requires val/regen/ and val/control/ folders. "
                "None found."
            )

        return X_train, Y_train, X_val, Y_val

    # ── Flat mode ─────────────────────────────────────────────────────────────
    patches_img  = img_train_dir  / 'cropped' / 'patches'
    patches_mask = mask_train_dir / 'cropped' / 'patches'

    X = load_patches(str(patches_img),  is_mask=False)
    Y = load_patches(str(patches_mask), is_mask=True)

    if len(X) != len(Y):
        raise ValueError(
            f"Patch count mismatch: {len(X)} images vs {len(Y)} masks"
        )

    # Check for pre-split val set
    has_val = (
        img_val_dir and mask_val_dir and
        Path(img_val_dir).exists() and
        any(p.suffix.lower() in ('.png', '.tif', '.tiff')
            for p in Path(img_val_dir).iterdir())
    )

    if has_val:
        X_val = load_patches(img_val_dir,  is_mask=False)
        Y_val = load_patches(mask_val_dir, is_mask=True)
        return X, Y, X_val, Y_val
    else:
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=val_fraction, random_state=42
        )
        return X_train, Y_train, X_val, Y_val