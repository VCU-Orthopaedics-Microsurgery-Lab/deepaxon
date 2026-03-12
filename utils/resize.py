"""
utils/resize.py

Image resize utility. Resizes microscopy images from acquisition resolution
(2048×2880) to working resolution (1024×1440) with greyscale conversion.
Used by both train/data/preprocess.py and segment/segment.py.
"""

from __future__ import annotations

import cv2
import numpy as np


TARGET_SIZE = (1440, 1024)  # (width, height) for cv2


def resize_img(img_path: str, is_mask: bool = False) -> np.ndarray:
    """
    Load and resize an image to TARGET_SIZE (1440×1024).

    For regular images: converts to greyscale, resizes with INTER_AREA.
    For masks: loads as greyscale, resizes with INTER_NEAREST to preserve labels.

    Args:
        img_path: path to the image file
        is_mask:  if True, use nearest-neighbour interpolation

    Returns:
        np.ndarray of shape (1024, 1440) dtype uint8

    Raises:
        ValueError: if the image cannot be read
    """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # Convert to greyscale if multi-channel
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Only resize if not already target size
    h, w = img.shape[:2]
    if (w, h) == TARGET_SIZE:
        return img

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    img = cv2.resize(img, TARGET_SIZE, interpolation=interp)
    return img


def get_image_resolution(img_path: str) -> tuple[int, int]:
    """
    Return (width, height) of an image without fully loading it.
    Uses PIL for speed — does not decode pixel data.
    """
    from PIL import Image
    with Image.open(img_path) as img:
        return img.size  # (width, height)
