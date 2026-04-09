"""
train/data/augment.py

On-the-fly data augmentation for DeepAxon training.
All augmentations are applied to image/mask pairs consistently.

Geometric augmentations (flip, rotate) — applied to BOTH image and mask.
Photometric augmentations (brightness, gamma, noise) — applied to IMAGE ONLY.

Separate probabilities for geometric vs photometric augmentations:
    geometric_prob  (default 0.5) — higher, nerves have no inherent orientation
    photometric_prob (default 0.25) — lower, imaging is already standardized

All parameters configurable via config.json augmentation block.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple

from utils.helpers import load_config


def augment_pair(
    image: np.ndarray,
    mask:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Apply random augmentations to an image/mask pair.

    Probabilities and parameters read from config.json augmentation block.

    Args:
        image: (H, W, 1) float32 normalised [0,1]
        mask:  (H, W, 1) int class indices or (H, W, C) one-hot

    Returns:
        (augmented image, augmented mask, list of applied augmentation names)
    """
    config  = load_config()
    aug_cfg   = config.get("augmentation", {})
    prob_cfg  = aug_cfg.get("probabilities", {})
    param_cfg = aug_cfg.get("parameters", {})

    geo_prob          = prob_cfg.get("geometric_prob",    0.5)
    photo_prob        = prob_cfg.get("photometric_prob",  0.25)
    rotation_deg      = param_cfg.get("rotation_deg",     15)
    brightness_range  = param_cfg.get("brightness_range", [0.8, 1.2])
    brightness_offset = param_cfg.get("brightness_offset",[-0.1, 0.1])
    gamma_range       = param_cfg.get("gamma_range",      [0.7, 1.4])
    noise_sigma       = param_cfg.get("noise_sigma",      0.02)

    rng     = np.random.default_rng()
    applied = []

    # ── Geometric — applied to BOTH image and mask ────────────────────────────

    if rng.random() < geo_prob:
        image = np.fliplr(image)
        mask  = np.fliplr(mask)
        applied.append('hflip')

    if rng.random() < geo_prob:
        image = np.flipud(image)
        mask  = np.flipud(mask)
        applied.append('vflip')

    if rng.random() < geo_prob:
        angle = rng.uniform(-rotation_deg, rotation_deg)
        h, w  = image.shape[:2]
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h),
                       flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)[..., np.newaxis]
        mask  = cv2.warpAffine(mask, M, (w, h),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_REFLECT_101)[..., np.newaxis]
        applied.append(f'rot{angle:.1f}')

    # ── Photometric — applied to IMAGE ONLY ───────────────────────────────────

    if rng.random() < photo_prob:
        alpha = rng.uniform(brightness_range[0],  brightness_range[1])
        beta  = rng.uniform(brightness_offset[0], brightness_offset[1])
        image = np.clip(image * alpha + beta, 0.0, 1.0)
        applied.append('brightness')

    if rng.random() < photo_prob:
        gamma = rng.uniform(gamma_range[0], gamma_range[1])
        image = np.power(np.clip(image, 0.0, 1.0), gamma)
        applied.append('gamma')

    if rng.random() < photo_prob:
        noise = rng.normal(0, noise_sigma, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)
        applied.append('noise')

    return image.astype(np.float32), mask.astype(np.float32), applied


def augment_dataset_np(
    images: np.ndarray,
    masks:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int, dict]:
    """
    Augment a full dataset, replacing each sample with its augmented version.

    Args:
        images: (N, H, W, 1) float32
        masks:  (N, H, W, 1) int class indices

    Returns:
        (augmented images, augmented masks, n_augmented, aug_counts)

        n_augmented: number of patches that received at least one augmentation
        aug_counts:  dict of per-augmentation application counts
                     keys: hflip, vflip, rotation, brightness, gamma, noise
    """
    aug_images = images.copy()
    aug_masks  = masks.copy()

    n_augmented = 0
    aug_counts  = {
        'hflip': 0, 'vflip': 0, 'rotation': 0,
        'brightness': 0, 'gamma': 0, 'noise': 0
    }

    for i in range(len(images)):
        aug_img, aug_mask, applied = augment_pair(images[i], masks[i])

        aug_images[i] = aug_img
        aug_masks[i]  = aug_mask

        if applied:
            n_augmented += 1
            for a in applied:
                key = 'rotation' if a.startswith('rot') else a
                aug_counts[key] += 1

    return aug_images, aug_masks, n_augmented, aug_counts