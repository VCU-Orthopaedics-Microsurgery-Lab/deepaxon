# train/utils/cli_interactive.py
"""
-------------------------------- DEEPAXON --------------------------------
Interactive CLI to train a DeepAxon++ segmentation model.
Combines dataset preparation, patch loading, augmentation, and training.
"""

import os
import shutil
import random
import re
import numpy as np
from rich.console import Console
from rich.panel import Panel

from train import train
from train.utils.helpers import (
    get_training_dir,
    get_model_dir,
    get_int_input,
    get_float_input,
    compute_aug_prob,
    compute_batch_size,
    list_files,
    count_patches
)
from train.utils.console_utils import header, info, success, warn
from train.data.preprocess import batch_process
from train.data.data_loader import load_all_patches
from train.data.augment import augment_dataset_np

console = Console()
VALID_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif")


# -------------------------- Dataset Utilities --------------------------- #
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def ensure_val_split(training_dir, test_fraction):
    """Ensure validation split exists; move fraction of train if missing."""
    train_img_dir = os.path.join(training_dir, "images", "train")
    val_img_dir = os.path.join(training_dir, "images", "val")
    train_mask_dir = os.path.join(training_dir, "masks", "train")
    val_mask_dir = os.path.join(training_dir, "masks", "val")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)

    # skip if val folder exists with images
    if os.path.exists(val_img_dir) and os.path.exists(val_mask_dir) and list_files(val_img_dir, VALID_IMAGE_EXTS):
        info("[INFO] User validation folder found → skipping split.")
        return

    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)

    train_images = list_files(train_img_dir, VALID_IMAGE_EXTS)
    if not train_images:
        raise RuntimeError("No training images found — cannot create validation split.")

    n_val = max(1, int(len(train_images) * test_fraction))
    selected = random.sample(train_images, n_val)

    for img_path in selected:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_candidates = [f for f in os.listdir(train_mask_dir) if os.path.splitext(f)[0] == base_name]
        if not mask_candidates:
            raise FileNotFoundError(f"[ERROR] Mask missing for image '{base_name}'.")
        mask_path = os.path.join(train_mask_dir, mask_candidates[0])

        shutil.move(img_path, os.path.join(val_img_dir, os.path.basename(img_path)))
        shutil.move(mask_path, os.path.join(val_mask_dir, os.path.basename(mask_path)))

    success(f"[SUCCESS] Created validation split → {n_val} paired images + masks moved.")


def prepare_dataset(images, masks, split_name, patch_size=256):
    """Resize → crop → patch if patches don't exist"""
    if not images:
        warn(f"No images found for {split_name}. Skipping preprocessing.")
        return
    parent_dir = os.path.dirname(images[0])
    patch_dir = os.path.join(parent_dir, "cropped", "patches")
    if not os.path.exists(patch_dir) or count_patches(patch_dir) == 0:
        info(f"Processing {split_name} folder: resize → crop → patch")
        batch_process(images, masks, patch_size=patch_size)
        print()
    else:
        info(f"{split_name} folder patches already exist, skipping preprocessing.")
        print()


def load_patches(train_or_val_dir):
    """Load image and mask patches into memory and convert masks to float32"""
    img_dir = os.path.join(train_or_val_dir, "images", "cropped", "patches")
    mask_dir = os.path.join(train_or_val_dir, "masks", "cropped", "patches")
    patches, masks_arr, patch_count = load_all_patches(img_dir, mask_dir)

    # Ensure mask dtype is float32 0-1
    if np.issubdtype(masks_arr.dtype, np.integer):
        masks_arr = masks_arr.astype(np.float32) / 255.0
    else:
        masks_arr = masks_arr.astype(np.float32)
        if masks_arr.max() > 1.0:
            masks_arr /= 255.0

    # add channel dim if missing
    if patches.ndim == 3:
        patches = patches[..., np.newaxis]
    if masks_arr.ndim == 3:
        masks_arr = masks_arr[..., np.newaxis]

    return patches, masks_arr, patch_count


def summarize_dataset(train_masks, val_masks):
    """Print dataset summary including mask pixel values"""
    def _mask_pixel_values(masks):
        unique_vals = set()
        for mask in masks:
            unique_vals.update(np.unique(mask))
        return sorted(unique_vals)

    train_mask_values = _mask_pixel_values(train_masks)
    val_mask_values = _mask_pixel_values(val_masks)

    console.print(Panel(
        f"[INFO] Dataset ready\n"
        f"Train masks: {len(train_masks)} | pixel values: {train_mask_values}\n"
        f"Val   masks: {len(val_masks)} | pixel values: {val_mask_values}",
        title="DATASET SUMMARY"
    ))


def get_batches(images, masks, batch_size=8, augment=False):
    """Yield batches with optional augmentation"""
    n = len(images)
    for i in range(0, n, batch_size):
        batch_imgs = images[i:i+batch_size]
        batch_masks = masks[i:i+batch_size]
        if augment:
            batch_imgs, batch_masks, _ = augment_dataset_np(batch_imgs, batch_masks)
        yield batch_imgs.astype(np.float32), batch_masks.astype(np.float32)


# -------------------------- Main CLI --------------------------- #
def run_interactive():
    header("DeepAxon++ Interactive DeepLearning Model Trainer")

    training_dir = get_training_dir()
    model_dir = get_model_dir()
    model_name = input("Input the name of the model: ").strip() or "default_model"

    epochs = get_int_input("Epochs", 200, min_value=1)
    test_fraction = get_float_input("Fraction of dataset for testing (0-1)", 0.3, min_value=0.0)
    use_aug = input("Use data augmentation? [y/N]: ").strip().lower() in ["y", "yes"]
    aug_prob, aug_lines = compute_aug_prob(use_aug)

    info(aug_lines[0])
    for line in aug_lines[1:]:
        print(f"  {line.strip()}")
    print()

    # ---------------- Step 1: Dataset Split ---------------- #
    ensure_val_split(training_dir, test_fraction)

    # ---------------- Step 2: Load original image/mask paths ---------------- #
    def get_sorted_files(subdir):
        return sorted(list_files(subdir, VALID_IMAGE_EXTS), key=natural_sort_key)

    train_img_dir = os.path.join(training_dir, "images", "train")
    train_mask_dir = os.path.join(training_dir, "masks", "train")
    val_img_dir = os.path.join(training_dir, "images", "val")
    val_mask_dir = os.path.join(training_dir, "masks", "val")

    train_images = get_sorted_files(train_img_dir)
    train_masks = get_sorted_files(train_mask_dir)
    val_images = get_sorted_files(val_img_dir)
    val_masks = get_sorted_files(val_mask_dir)

    # ---------------- Step 3: Preprocess ---------------- #
    prepare_dataset(train_images, train_masks, "train")
    prepare_dataset(val_images, val_masks, "val")

    # ---------------- Step 4: Load patches ---------------- #
    train_patches, train_masks_arr, train_patch_count = load_patches(os.path.join(training_dir, "train"))
    val_patches, val_masks_arr, val_patch_count = load_patches(os.path.join(training_dir, "val"))

    info(f"Dataset ready → {train_patch_count} train patches, {val_patch_count} val patches")

    summarize_dataset(train_masks_arr, val_masks_arr)

    # ---------------- Step 5: Batch size ---------------- #
    recommended_batch, _, _ = compute_batch_size(len(train_patches))
    info(f"Recommended batch size: {recommended_batch}")

    while True:
        custom_input = input("Enter custom batch size or press Enter to use recommended: ").strip()
        if not custom_input:
            batch_size = recommended_batch
            break
        try:
            batch_size = int(custom_input)
            _, remainder, efficiency = compute_batch_size(len(train_patches), desired_batch=batch_size)
            confirm = input(f"Confirm batch size {batch_size}? [y/N]: ").strip().lower()
            if confirm in ("y", "yes"):
                break
        except ValueError:
            warn("Invalid input. Must be an integer.")

    # ---------------- Step 6: Train ---------------- #
    train.train_model(
        training_dir=training_dir,
        model_dir=model_dir,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        test_fraction=test_fraction,
        augment=use_aug,
    )

    success("Training session completed successfully.")