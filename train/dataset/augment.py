"""
train/dataset/augment.py

On-the-fly data augmentation for DeepAxon training.
All augmentations are applied to image/mask pairs consistently.

Geometric augmentations (flip, rotate, elastic) — applied to BOTH image and mask.
Photometric augmentations (brightness, gamma, noise, blur, CLAHE) — applied to IMAGE ONLY.

Two modes:

    Config mode (Wave 1, interactive):
        aug_params=None → reads from config.json augmentation block.
        Uses shared geo_prob / photo_prob for all aug types.

    Parametric mode (Wave 2):
        aug_params dict passed per job from run_cfg['aug_params'].
        Each aug type has its own probability and intensity parameters.
        All probabilities default to 0.0 — only the swept aug type fires.

New aug types added for Wave 2 sweep:
    gaussian_blur   — simulates focus variation (image only)
    elastic         — tissue deformation during sectioning (image + mask)
    clahe           — contrast enhancement variation (image only)
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import gaussian_filter

from utils.helpers import load_config


# ─── Augment pair ─────────────────────────────────────────────────────────────

def augment_pair(
    image:      np.ndarray,
    mask:       np.ndarray,
    aug_params: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Apply random augmentations to an image/mask pair.

    Args:
        image:      (H, W, 1) float32 normalised [0,1]
        mask:       (H, W, 1) int class indices
        aug_params: optional per-job parameter dict from run_cfg['aug_params'].
                    If None, reads from config.json (Wave 1 / interactive mode).

    Returns:
        (augmented image, augmented mask, list of applied augmentation names)
    """
    rng     = np.random.default_rng()
    applied = []

    # ── Resolve parameters ────────────────────────────────────────────────────
    if aug_params is not None:
        # Parametric mode — Wave 2
        hflip_prob        = aug_params.get('hflip_prob',        0.0)
        vflip_prob        = aug_params.get('vflip_prob',        0.0)
        rotation_prob     = aug_params.get('rotation_prob',     0.0)
        rotation_deg      = aug_params.get('rotation_deg',      15)
        brightness_prob   = aug_params.get('brightness_prob',   0.0)
        brightness_scale  = aug_params.get('brightness_scale',  [0.8, 1.2])
        brightness_offset = aug_params.get('brightness_offset', [-0.1, 0.1])
        gamma_prob        = aug_params.get('gamma_prob',        0.0)
        gamma_range       = aug_params.get('gamma_range',       [0.7, 1.4])
        noise_prob        = aug_params.get('noise_prob',        0.0)
        noise_sigma       = aug_params.get('noise_sigma',       0.02)
        blur_prob         = aug_params.get('blur_prob',         0.0)
        blur_sigma        = aug_params.get('blur_sigma',        1.0)
        elastic_prob      = aug_params.get('elastic_prob',      0.0)
        elastic_alpha     = aug_params.get('elastic_alpha',     20)
        elastic_sigma     = aug_params.get('elastic_sigma',     8)
        clahe_prob        = aug_params.get('clahe_prob',        0.0)
        clahe_clip        = aug_params.get('clahe_clip',        1.5)
        clahe_tile        = aug_params.get('clahe_tile',        16)
    else:
        # Config mode — Wave 1 / interactive
        config    = load_config()
        aug_cfg   = config.get("augmentation", {})
        prob_cfg  = aug_cfg.get("probabilities", {})
        param_cfg = aug_cfg.get("parameters", {})

        geo_prob   = prob_cfg.get("geometric_prob",   0.5)
        photo_prob = prob_cfg.get("photometric_prob", 0.25)

        hflip_prob        = geo_prob
        vflip_prob        = geo_prob
        rotation_prob     = geo_prob
        rotation_deg      = param_cfg.get("rotation_deg",      15)
        brightness_prob   = photo_prob
        brightness_scale  = param_cfg.get("brightness_range",  [0.8, 1.2])
        brightness_offset = param_cfg.get("brightness_offset", [-0.1, 0.1])
        gamma_prob        = photo_prob
        gamma_range       = param_cfg.get("gamma_range",       [0.7, 1.4])
        noise_prob        = photo_prob
        noise_sigma       = param_cfg.get("noise_sigma",       0.02)
        # New aug types default OFF in config mode
        blur_prob         = 0.0
        blur_sigma        = 1.0
        elastic_prob      = 0.0
        elastic_alpha     = 20
        elastic_sigma     = 8
        clahe_prob        = 0.0
        clahe_clip        = 1.5
        clahe_tile        = 16

    # ── Geometric — applied to BOTH image and mask ────────────────────────────

    if rng.random() < hflip_prob:
        image = np.fliplr(image)
        mask  = np.fliplr(mask)
        applied.append('hflip')

    if rng.random() < vflip_prob:
        image = np.flipud(image)
        mask  = np.flipud(mask)
        applied.append('vflip')

    if rng.random() < rotation_prob:
        angle = rng.uniform(-rotation_deg, rotation_deg)
        h, w  = image.shape[:2]
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101)[..., np.newaxis]
        mask  = cv2.warpAffine(mask.squeeze(-1), M, (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_REFLECT_101)[..., np.newaxis]
        applied.append(f'rot{angle:.1f}')

    if rng.random() < elastic_prob:
        image, mask = _elastic_deform(image, mask, elastic_alpha, elastic_sigma, rng)
        applied.append('elastic')

    # ── Photometric — applied to IMAGE ONLY ───────────────────────────────────

    if rng.random() < brightness_prob:
        alpha = rng.uniform(brightness_scale[0],  brightness_scale[1])
        beta  = rng.uniform(brightness_offset[0], brightness_offset[1])
        image = np.clip(image * alpha + beta, 0.0, 1.0)
        applied.append('brightness')

    if rng.random() < gamma_prob:
        gamma = rng.uniform(gamma_range[0], gamma_range[1])
        image = np.power(np.clip(image, 0.0, 1.0), gamma)
        applied.append('gamma')

    if rng.random() < noise_prob:
        noise = rng.normal(0, noise_sigma, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)
        applied.append('noise')

    if rng.random() < blur_prob:
        image = _gaussian_blur(image, blur_sigma)
        applied.append('blur')

    if rng.random() < clahe_prob:
        image = _apply_clahe(image, clahe_clip, int(clahe_tile))
        applied.append('clahe')

    return image.astype(np.float32), mask.astype(np.float32), applied


# ─── New aug type implementations ─────────────────────────────────────────────

def _elastic_deform(
    image: np.ndarray,
    mask:  np.ndarray,
    alpha: float,
    sigma: float,
    rng:   np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Elastic deformation — applied identically to image and mask.
    Displacement field smoothed by sigma, scaled by alpha.
    Preserves ground truth spatial relationships.

    image: (H, W, 1) float32
    mask:  (H, W, 1) float32
    """
    h, w = image.shape[:2]

    dx = gaussian_filter(rng.standard_normal((h, w)), sigma) * alpha
    dy = gaussian_filter(rng.standard_normal((h, w)), sigma) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)

    image_d = cv2.remap(
        image.squeeze(-1), map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )[..., np.newaxis]

    mask_d = cv2.remap(
        mask.squeeze(-1), map_x, map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT_101
    )[..., np.newaxis]

    return image_d, mask_d


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian blur — image only.
    Simulates focus variation between imaging sessions.

    image: (H, W, 1) float32
    """
    blurred = cv2.GaussianBlur(
        image.squeeze(-1),
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )
    return blurred[..., np.newaxis]


def _apply_clahe(image: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    """
    CLAHE contrast enhancement — image only.
    Simulates inter-session contrast variation.

    image: (H, W, 1) float32 [0,1]
    """
    img_uint8 = (image.squeeze(-1) * 255).astype(np.uint8)
    clahe     = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size)
    )
    enhanced  = clahe.apply(img_uint8)
    return (enhanced.astype(np.float32) / 255.0)[..., np.newaxis]


# ─── Dataset augmentation ─────────────────────────────────────────────────────

def augment_dataset_np(
    images:     np.ndarray,
    masks:      np.ndarray,
    aug_params: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, int, dict]:
    """
    Augment a full dataset, replacing each sample with its augmented version.

    Args:
        images:     (N, H, W, 1) float32
        masks:      (N, H, W, 1) int class indices
        aug_params: optional per-job parameter dict (Wave 2).
                    If None, reads from config.json (Wave 1 / interactive).

    Returns:
        (augmented images, augmented masks, n_augmented, aug_counts)

        n_augmented: number of patches that received at least one augmentation
        aug_counts:  dict of per-augmentation application counts
    """
    aug_images = images.copy()
    aug_masks  = masks.copy()

    n_augmented = 0
    aug_counts  = {
        'hflip': 0, 'vflip': 0, 'rotation': 0,
        'brightness': 0, 'gamma': 0, 'noise': 0,
        'blur': 0, 'elastic': 0, 'clahe': 0,
    }

    for i in range(len(images)):
        aug_img, aug_mask, applied = augment_pair(
            images[i], masks[i], aug_params=aug_params
        )
        aug_images[i] = aug_img
        aug_masks[i]  = aug_mask

        if applied:
            n_augmented += 1
            for a in applied:
                key = 'rotation' if a.startswith('rot') else a
                if key in aug_counts:
                    aug_counts[key] += 1

    return aug_images, aug_masks, n_augmented, aug_counts
