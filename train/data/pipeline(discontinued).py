# train/data/pipeline.py
"""
High-level DataPipeline for DeepAxon
1) Load original dataset
2) Preprocess: resize → crop → patch
3) Load patches into RAM
4) Yield batches for training with optional augmentation

Folder Structure:

training/
├─ images/
│  ├─ train/
│  └─ val/        <-- optional, auto-created if missing
├─ masks/
│  ├─ train/
│  └─ val/        <-- optional, auto-created if missing
"""

import os
import shutil
import random
import numpy as np
import re

from .data_loader import load_all_patches
from .preprocess import batch_process
from .augment import augment_dataset_np
from ..utils.console_utils import info, success, warn
from ..utils.helpers import list_files, count_patches

VALID_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif")


class DataPipeline:
    def __init__(self, training_dir, test_fraction=0.3, patch_size=256, augment=True):
        self.training_dir = training_dir
        self.test_fraction = test_fraction
        self.patch_size = patch_size
        self.augment = augment

        # 1. Ensure validation folders exist and move fraction of files if missing
        self._ensure_val_folders()

        # 2. Natural sort
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split(r"(\d+)", s)]

        # 3. Collect original image/mask paths
        self.train_images = sorted(
            list_files(os.path.join(training_dir, "images", "train"), extensions=VALID_IMAGE_EXTS),
            key=natural_sort_key
        )
        self.train_masks = sorted(
            list_files(os.path.join(training_dir, "masks", "train"), extensions=VALID_IMAGE_EXTS),
            key=natural_sort_key
        )
        self.val_images = sorted(
            list_files(os.path.join(training_dir, "images", "val"), extensions=VALID_IMAGE_EXTS),
            key=natural_sort_key
        )
        self.val_masks = sorted(
            list_files(os.path.join(training_dir, "masks", "val"), extensions=VALID_IMAGE_EXTS),
            key=natural_sort_key
        )

        # 4. Preprocess datasets (resize → crop → patch)
        self._prepare_dataset(self.train_images, self.train_masks, "train")
        self._prepare_dataset(self.val_images, self.val_masks, "val")

        # 5. Load patches with data_loader.load_all_patches()
        self.train_patches, self.train_masks_arr, self.train_patch_count = \
            load_all_patches(
                os.path.join(training_dir, "images", "train", "cropped", "patches"),
                os.path.join(training_dir, "masks",  "train", "cropped", "patches")
            )

        self.val_patches, self.val_masks_arr, self.val_patch_count = \
            load_all_patches(
                os.path.join(training_dir, "images", "val", "cropped", "patches"),
                os.path.join(training_dir, "masks",  "val", "cropped", "patches")
            )

        # 6. Validation sanity check message
        info(f"Dataset ready → {self.train_patch_count} train patches, "
             f"{self.val_patch_count} val patches")
        info(f"Sanity check → train images: {self.train_patches.shape}, "
             f"train masks: {self.train_masks_arr.shape}, "
             f"dtype: {self.train_patches.dtype}/{self.train_masks_arr.dtype}")
        info(f"Sanity check → val images: {self.val_patches.shape}, "
             f"val masks: {self.val_masks_arr.shape}, "
             f"dtype: {self.val_patches.dtype}/{self.val_masks_arr.dtype}")


    # =====================================================================
    # VALIDATION SPLIT
    # =====================================================================
    def _ensure_val_folders(self):
        """Ensure validation folders exist; create paired split if missing."""
        train_img_dir = os.path.join(self.training_dir, "images", "train")
        val_img_dir = os.path.join(self.training_dir, "images", "val")
        train_mask_dir = os.path.join(self.training_dir, "masks", "train")
        val_mask_dir = os.path.join(self.training_dir, "masks", "val")

        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(train_mask_dir, exist_ok=True)

        # If val folders exist and not empty, skip
        if os.path.exists(val_img_dir) and os.path.exists(val_mask_dir) and list_files(val_img_dir, VALID_IMAGE_EXTS):
            info("[INFO] User validation folder found → skipping split.")
            return

        # Create val folders
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(val_mask_dir, exist_ok=True)

        train_images = list_files(train_img_dir, VALID_IMAGE_EXTS)
        if not train_images:
            raise RuntimeError("No training images found — cannot create validation split.")

        n_val = max(1, int(len(train_images) * self.test_fraction))
        selected = random.sample(train_images, n_val)

        for img_path in selected:
            base_name = os.path.splitext(os.path.basename(img_path))[0]  # remove extension

            # find mask in train_mask_dir by basename only
            mask_candidates = [
                f for f in os.listdir(train_mask_dir)
                if os.path.splitext(f)[0] == base_name
            ]
            if not mask_candidates:
                raise FileNotFoundError(f"[ERROR] Mask missing for image '{base_name}'.")

            mask_path = os.path.join(train_mask_dir, mask_candidates[0])  # take first match

            # Move image
            shutil.move(img_path, os.path.join(val_img_dir, os.path.basename(img_path)))
            # Move mask
            shutil.move(mask_path, os.path.join(val_mask_dir, os.path.basename(mask_path)))

        success(f"[SUCCESS] Created validation split → {n_val} paired images + masks moved.")

    
    # =====================================================================
    # PREPROCESSING (resize → crop → patch)
    # =====================================================================
    def _prepare_dataset(self, images, masks, split_name):
        """Run preprocessing if cropped/patch folders do not exist"""
        if not images:
            warn(f"No images found for {split_name}. Skipping preprocessing.")
            return
        
        parent_dir = os.path.dirname(images[0])
        patch_dir = os.path.join(parent_dir, "cropped", "patches")
        
        if not os.path.exists(patch_dir) or count_patches(patch_dir) == 0:
            info(f"Processing {split_name} folder: resize → crop → patch")
            batch_process(images, masks, patch_size=self.patch_size)
            print()  # blank line
            
        else:
            info(f"{split_name} folder patches already exist, skipping preprocessing.")
            print()
            

    # =====================================================================
    # BATCH GENERATOR
    # =====================================================================
    def get_batches(self, batch_size=8, training=True):
        """Yield batches with optional augmentation and automatic dtype handling."""
        if training:
            patches = self.train_patches
            masks   = self.train_masks_arr
        else:
            patches = self.val_patches
            masks   = self.val_masks_arr

        n = len(patches)
        for i in range(0, n, batch_size):
            batch_imgs  = patches[i:i+batch_size]
            batch_masks = masks[i:i+batch_size]

            # augmentation only on training split
            if self.augment and training:
                batch_imgs, batch_masks, _ = augment_dataset_np(batch_imgs, batch_masks)

            # add channel dimension if missing
            if batch_imgs.ndim == 3:
                batch_imgs = batch_imgs[..., np.newaxis]
            if batch_masks.ndim == 3:
                batch_masks = batch_masks[..., np.newaxis]

            # --- auto-detect mask dtype ---
            if np.issubdtype(batch_masks.dtype, np.integer):
                batch_masks = batch_masks.astype(np.uint8)
            else:
                batch_masks = batch_masks.astype(np.float32)
                # normalize if values > 1
                if batch_masks.max() > 1.0:
                    batch_masks /= 255.0

            yield batch_imgs.astype(np.float32), batch_masks