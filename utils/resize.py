"""
utils/resize.py

Image resize utility. Resizes microscopy images from acquisition resolution
(2880×2048) to working resolution (1440×1024) with greyscale conversion.
Used by both train/data/preprocess.py and segment/segment.py.

Interpolation methods:
  Images: INTER_LANCZOS4 — high quality downsampling, matches v1 pipeline
  Masks:  INTER_NEAREST  — preserves exact label values (0/128/255), no blending
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


TARGET_SIZE = (1440, 1024)  # (width, height) for cv2


def resize_img(img_path: str, is_mask: bool = False) -> np.ndarray:
    """
    Load and resize an image to TARGET_SIZE (1440×1024).

    For images: INTER_LANCZOS4 — high quality downsampling matching v1 pipeline.
    For masks:  INTER_NEAREST  — preserves label values exactly, no blending.

    Args:
        img_path: path to the image file
        is_mask:  if True, use nearest-neighbour interpolation

    Returns:
        np.ndarray of shape (1024, 1440) dtype uint8

    Raises:
        ValueError: if the image cannot be read
    """
    img = cv2.imread(img_path, 0)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # Only resize if not already target size
    h, w = img.shape[:2]
    if (w, h) == TARGET_SIZE:
        return img

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LANCZOS4
    img = cv2.resize(img, TARGET_SIZE, interpolation=interpolation)
    return img


def get_image_resolution(img_path: str) -> tuple[int, int]:
    """
    Return (width, height) of an image without fully loading it.
    Uses PIL for speed — does not decode pixel data.
    """
    with Image.open(img_path) as img:
        return img.size  # (width, height)