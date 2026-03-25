"""
train/data/augment.py

On-the-fly data augmentation for DeepAxon training.
All augmentations are applied to image/mask pairs consistently.
Mask augmentations: geometric only (flip, rotate).
Image-only augmentations: brightness, gamma, noise.
All parameters are configurable via config.json augmentation block.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple

from utils.helpers import load_config


def augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
    prob: float = 0.25
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random augmentations to an image/mask pair.

    Args:
        image: (H, W, 1) float32 normalised [0,1]
        mask:  (H, W, C) float32 one-hot or (H, W) int
        prob:  probability of each augmentation being applied

    Returns:
        Augmented (image, mask) pair
    """
    config  = load_config()
    aug_cfg = config.get("augmentation", {})

    rotation_deg      = aug_cfg.get("rotation_deg",       15)
    brightness_range  = aug_cfg.get("brightness_range",   [0.8, 1.2])
    brightness_offset = aug_cfg.get("brightness_offset",  [-0.1, 0.1])
    gamma_range       = aug_cfg.get("gamma_range",        [0.7, 1.4])
    noise_sigma       = aug_cfg.get("noise_sigma",        0.02)

    rng = np.random.default_rng()

    # ── Horizontal flip ───────────────────────────────────────────────────────
    if rng.random() < prob:
        image = np.fliplr(image)
        mask  = np.fliplr(mask)

    # ── Vertical flip ─────────────────────────────────────────────────────────
    if rng.random() < prob:
        image = np.flipud(image)
        mask  = np.flipud(mask)

    # ── Small rotation ────────────────────────────────────────────────────────
    if rng.random() < prob:
        angle  = rng.uniform(-rotation_deg, rotation_deg)
        h, w   = image.shape[:2]
        M      = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image  = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        mask   = cv2.warpAffine(mask,  M, (w, h),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_REFLECT_101)

    # ── Brightness / contrast (image only) ───────────────────────────────────
    if rng.random() < prob:
        alpha = rng.uniform(brightness_range[0],  brightness_range[1])
        beta  = rng.uniform(brightness_offset[0], brightness_offset[1])
        image = np.clip(image * alpha + beta, 0.0, 1.0)

    # ── Gamma correction (image only) ─────────────────────────────────────────
    if rng.random() < prob:
        gamma = rng.uniform(gamma_range[0], gamma_range[1])
        image = np.power(np.clip(image, 0.0, 1.0), gamma)

    # ── Gaussian noise (image only) ──────────────────────────────────────────
    if rng.random() < prob:
        noise = rng.normal(0, noise_sigma, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)

    return image.astype(np.float32), mask.astype(np.float32)


def augment_dataset_np(
    images: np.ndarray,
    masks: np.ndarray,
    prob: float = 0.25
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Augment a full dataset in-place, replacing each sample with its
    augmented version.

    Args:
        images: (N, H, W, 1) float32
        masks:  (N, H, W, C) float32
        prob:   per-augmentation probability

    Returns:
        (augmented images, augmented masks, count augmented)
    """
    aug_images = images.copy()
    aug_masks  = masks.copy()
    count      = 0

    for i in range(len(images)):
        aug_img, aug_mask    = augment_pair(images[i], masks[i], prob)
        aug_images[i]        = aug_img
        aug_masks[i]         = aug_mask
        count               += 1

    return aug_images, aug_masks, count