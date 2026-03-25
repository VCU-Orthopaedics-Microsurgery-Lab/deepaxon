"""
train/data/data_loader.py

Load preprocessed patches from disk into numpy arrays for training.
"""

from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path
from keras.utils import normalize
from sklearn.model_selection import train_test_split

from utils.helpers import list_files


def base_label(arr: np.ndarray) -> np.ndarray:
    """
    Remap mask pixel values to class indices.
    Matches original DeepAxon training pipeline (v1 train.py).

    Pixel value → class index:
        0         → 0 (background)
        127, 128  → 1 (axon / grey)
        255       → 2 (myelin / white)

    Args:
        arr: (N, H, W, 1) float32 mask array with raw pixel values

    Returns:
        (N, H, W, 1) float32 array with values 0, 1, or 2
    """
    pixel_to_class = {0: 0, 127: 1, 128: 1, 255: 2}
    return np.vectorize(lambda x: pixel_to_class.get(int(x), 0))(arr).astype(np.float32)


def load_patches(patches_dir: str, is_mask: bool = False) -> np.ndarray:
    """
    Load all patches from a directory into a numpy array.

    For images: normalizes using keras.utils.normalize(arr, axis=1) —
        L2 normalization along the width axis, matching the original
        DeepAxon training pipeline (v1 train.py).
    For masks: loads raw pixel values and remaps to class indices
        via base_label() — call to_categorical() in train.py before model.fit().

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
        # Normalize along axis=1 (width) to match original training pipeline.
        # keras.utils.normalize performs L2 normalization per row.
        # Do NOT change to /= 255.0 — existing models were trained with this method.
        arr = normalize(arr, axis=1)

    return arr


def load_all_patches(
    img_patches_dir: str,
    mask_patches_dir: str,
    val_img_patches_dir: str = None,
    val_mask_patches_dir: str = None,
    val_fraction: float = 0.1
) -> tuple:
    """
    Load train and validation patches.

    If val directories are provided and populated, uses them directly.
    Otherwise splits from train set using val_fraction.

    Returns:
        (X_train, Y_train, X_val, Y_val)
        Masks are class indices (0/1/2) — apply to_categorical() in train.py
        before passing to model.fit().
    """
    X = load_patches(img_patches_dir, is_mask=False)
    Y = load_patches(mask_patches_dir, is_mask=True)

    if len(X) != len(Y):
        raise ValueError(
            f"Patch count mismatch: {len(X)} images vs {len(Y)} masks"
        )

    # Check for pre-split val set — filter to image files only
    has_val = (
        val_img_patches_dir and val_mask_patches_dir and
        Path(val_img_patches_dir).exists() and
        any(p.suffix.lower() in ('.png', '.tif', '.tiff')
            for p in Path(val_img_patches_dir).iterdir())
    )

    if has_val:
        X_val = load_patches(val_img_patches_dir, is_mask=False)
        Y_val = load_patches(val_mask_patches_dir, is_mask=True)
        return X, Y, X_val, Y_val
    else:
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=val_fraction, random_state=42
        )
        return X_train, Y_train, X_val, Y_val