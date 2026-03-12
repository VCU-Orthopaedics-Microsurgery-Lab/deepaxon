"""
train/data/data_loader.py

Load preprocessed patches from disk into numpy arrays for training.
"""

from __future__ import annotations

import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split

from utils.helpers import list_files


def load_patches(patches_dir: str, is_mask: bool = False) -> np.ndarray:
    """
    Load all patches from a directory into a numpy array.

    Returns:
        (N, H, W, 1) float32 normalised array
    """
    files = list_files(patches_dir, extensions=('.png', '.tif', '.tiff'))
    if not files:
        raise FileNotFoundError(f"No patch files found in {patches_dir}")

    patches = []
    for f in sorted(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        patches.append(img)

    arr = np.array(patches, dtype=np.float32)
    arr = np.expand_dims(arr, axis=-1)  # (N, H, W, 1)

    if not is_mask:
        arr /= 255.0

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
    """
    X = load_patches(img_patches_dir, is_mask=False)
    Y = load_patches(mask_patches_dir, is_mask=True)

    if len(X) != len(Y):
        raise ValueError(
            f"Patch count mismatch: {len(X)} images vs {len(Y)} masks"
        )

    # Check for pre-split val set
    has_val = (
        val_img_patches_dir and val_mask_patches_dir and
        Path(val_img_patches_dir).exists() and
        any(Path(val_img_patches_dir).iterdir())
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
