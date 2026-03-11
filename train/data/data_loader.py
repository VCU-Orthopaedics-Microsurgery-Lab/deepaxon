# train/data/data_loader.py
"""
Load and verify dataset image–mask pairs, and load preprocessed patches into memory.
"""

import os
import cv2
import re
import numpy as np
from ..utils.helpers import list_files
from ..utils.console_utils import info, warn, success

VALID_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif")


# ---------------------- Dataset Paths ---------------------- #
def get_dataset_paths(training_dir, test_fraction=0.3):
    """
    Return train/validation split of original image and mask paths.
    Ensures 1:1 matching by basename.

    Args:
        training_dir (str): Path containing 'images' and 'masks' subfolders
        test_fraction (float): Fraction for validation set

    Returns:
        train_images, val_images, train_masks, val_masks (lists of paths)
    """
    images_dir = os.path.join(training_dir, "images")
    masks_dir = os.path.join(training_dir, "masks")

    image_files = list_files(images_dir, extensions=VALID_IMAGE_EXTS)
    mask_files = list_files(masks_dir, extensions=VALID_IMAGE_EXTS)

    # Natural sort
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    image_files.sort(key=natural_sort_key)
    mask_files.sort(key=natural_sort_key)

    def basename_no_ext(f):
        return os.path.splitext(os.path.basename(f))[0]

    # Match images to masks
    matched_images, matched_masks = [], []
    mask_basenames = {basename_no_ext(f): f for f in mask_files}

    for img_file in image_files:
        img_name = basename_no_ext(img_file)
        if img_name in mask_basenames:
            matched_images.append(img_file)
            matched_masks.append(mask_basenames[img_name])
        else:
            warn(f"Skipping {img_file} — no matching mask found.")

    # Train/validation split
    n = len(matched_images)
    if n == 0:
        raise ValueError(f"No matching image-mask pairs found in {training_dir}.")

    split_idx = int(n * (1 - test_fraction))
    train_images, val_images = matched_images[:split_idx], matched_images[split_idx:]
    train_masks, val_masks = matched_masks[:split_idx], matched_masks[split_idx:]

    info("Data directory loaded")
    return train_images, val_images, train_masks, val_masks

# =====================================================================
# PATCH LOADING
# =====================================================================
def load_patches(patch_dir, is_mask=False):
    """
    Read all preprocessed patch PNGs under patch_dir into a numpy array.
    Returns array of shape (num_patches, h, w) and total count.
    """
    files = [os.path.join(patch_dir, f) for f in os.listdir(patch_dir)
             if os.path.isfile(os.path.join(patch_dir, f)) and not f.startswith('.')]
    files.sort()

    if not files:
        warn(f"No patch files found in {patch_dir}")
        return np.array([]), 0

    # Load images
    dtype = np.uint8 if is_mask else np.float32
    patches = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files], dtype=dtype)

    # Ensure 3D array (num_patches, h, w)
    if patches.ndim == 2:
        patches = patches[..., np.newaxis]

    info(f"Loaded {len(patches)} {'mask' if is_mask else 'image'} patches from {patch_dir}")
    return patches, len(patches)


def load_all_patches(img_patch_dir, mask_patch_dir):
    img_patches, n_img = load_patches(img_patch_dir, is_mask=False)
    mask_patches, n_mask = load_patches(mask_patch_dir, is_mask=True)

    if n_img != n_mask:
        warn(f"Patch count mismatch: {n_img} images vs {n_mask} masks")

    return img_patches, mask_patches, n_img