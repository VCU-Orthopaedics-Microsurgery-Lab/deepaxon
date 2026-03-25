"""
train/data/preprocess.py

Image preprocessing pipeline for DeepAxon training.
resize -> center crop -> save cropped image -> patchify (50% overlap) -> save patches
"""

from __future__ import annotations

import cv2
from pathlib import Path
from patchify import patchify

from utils.resize import resize_img
from utils.console import DeepAxonLogger
from utils.helpers import load_config, list_files


def center_crop(img, patch_size):
    h, w = img.shape[:2]
    crop_h = (h // patch_size) * patch_size
    crop_w = (w // patch_size) * patch_size
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    return img[start_h:start_h + crop_h, start_w:start_w + crop_w]


def process_single_image(
    img_path,
    patches_dir,
    cropped_dir,
    patch_size=256,
    is_mask=False,
    log=None
):
    """
    Process a single image: resize -> center crop -> save cropped -> patchify (50% overlap) -> save patches.

    Resize interpolation:
      Images: INTER_LANCZOS4 (via resize_img is_mask=False)
      Masks:  INTER_NEAREST  (via resize_img is_mask=True) — preserves label values

    Returns number of patches saved.
    """
    step = patch_size // 2  # 50% overlap

    img      = resize_img(img_path, is_mask=is_mask)
    img_crop = center_crop(img, patch_size)
    crop_h, crop_w = img_crop.shape[:2]

    # Save cropped image
    stem     = Path(img_path).stem
    crop_ext = '.tif' if not is_mask else '.png'
    Path(cropped_dir).mkdir(parents=True, exist_ok=True)
    crop_out = Path(cropped_dir) / f"{stem}_cropped{crop_ext}"
    cv2.imwrite(str(crop_out), img_crop)

    # Patchify with 50% overlap
    Path(patches_dir).mkdir(parents=True, exist_ok=True)
    patches = patchify(img_crop, (patch_size, patch_size), step=step)
    n_rows, n_cols = patches.shape[:2]

    count = 0
    for i in range(n_rows):
        for j in range(n_cols):
            patch     = patches[i, j]
            out_path  = Path(patches_dir) / f"{stem}_{i:02d}{j:02d}.png"
            cv2.imwrite(str(out_path), patch)
            count += 1

    if log:
        h_orig, w_orig = img.shape[:2]
        log.info(
            f"  {'[MASK]' if is_mask else '[IMG] '} {stem} | "
            f"Resized: ({h_orig}x{w_orig}) -> Cropped: ({crop_h}x{crop_w}) -> {count} patches"
        )

    return count


def batch_process(images_dir, masks_dir, patches_img_dir, patches_mask_dir,
                  cropped_img_dir, cropped_mask_dir, mag, log):
    """
    Process all images and masks.
    Validates that every image has a corresponding mask before processing.
    Returns (n_img_patches, n_mask_patches).
    """
    config     = load_config()
    patch_size = config.get("patch_size", {}).get(mag, 256)

    images = list_files(images_dir, extensions=('.tif', '.tiff'))
    masks  = list_files(masks_dir,  extensions=('.tif', '.tiff', '.png'))

    # Validate image/mask pairing before processing
    img_stems  = {Path(p).stem for p in images}
    mask_stems = {Path(p).stem for p in masks}
    missing    = img_stems - mask_stems
    if missing:
        raise ValueError(
            f"Missing masks for {len(missing)} image(s): {sorted(missing)}"
        )

    log.rule("IMAGE PROCESSING")
    log.info(f"Patch size: {patch_size}px | Overlap: 50%")
    total_img = 0
    for img_path in images:
        total_img += process_single_image(
            str(img_path), patches_img_dir, cropped_img_dir,
            patch_size=patch_size, is_mask=False, log=log
        )

    log.rule("MASK PROCESSING")
    total_mask = 0
    for mask_path in masks:
        total_mask += process_single_image(
            str(mask_path), patches_mask_dir, cropped_mask_dir,
            patch_size=patch_size, is_mask=True, log=log
        )

    return total_img, total_mask