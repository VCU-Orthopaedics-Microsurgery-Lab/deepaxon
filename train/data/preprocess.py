# train/data/preprocess.py
"""
Image preprocessing for DeepAxon
Steps:
1) Resize originals to standard size
2) Center-crop to multiples of patch size (save to parentfolder/cropped)
3) Split into patches (save to parentfolder/cropped/patches)
"""

import os
import cv2
import numpy as np
from patchify import patchify
from ..utils.console_utils import info, warn, rule
from rich.console import Console
from rich.panel import Panel

console = Console()


# ------------------------------ Single Image Processing ----------------------- #
def process_single_image(path, patch_size=256, target_shape=(1024, 1440)):
    image_name = os.path.splitext(os.path.basename(path))[0]
    parent_dir = os.path.dirname(path)
    is_mask = "mask" in parent_dir.lower()

    # Load image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    orig_shape = img.shape[:2]  # store H, W before any resizing
    
    # Ensure image is 2D (grayscale) for processing
    if img.ndim == 3:
        if img.shape[2] == 3 or img.shape[2] == 4:  # RGB or RGBA
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 1:  # single-channel 3D
            img = img[:, :, 0]

    # Resize if needed
    if img.shape != target_shape:
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=interp)

    # Masks: enforce pixel values
    if is_mask:
        img = np.where((img >= 126) & (img <= 129), 127, img)
        img = np.where(img > 200, 255, img)
        img = np.where(img < 50, 0, img)

    # Crop to multiple of patch_size
    h, w = img.shape
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    cropped = img[top:top+new_h, left:left+new_w]

    # Save cropped
    cropped_dir = os.path.join(parent_dir, "cropped")
    os.makedirs(cropped_dir, exist_ok=True)
    cv2.imwrite(os.path.join(cropped_dir, os.path.basename(path)), cropped)

    # Patchify
    patch_dir = os.path.join(cropped_dir, "patches")
    os.makedirs(patch_dir, exist_ok=True)
    patches = patchify(cropped, (patch_size, patch_size), step=patch_size)

    patch_count = 0
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j]
            if patch.ndim == 3 and patch.shape[0] == 1:
                patch = patch[0]
            cv2.imwrite(os.path.join(patch_dir, f"{image_name}_{i}{j}.png"), patch)
            patch_count += 1

    # Pixel info for masks
    pixel_info = ""
    if is_mask:
        all_pixel_vals = np.unique(np.array([p.ravel() for row in patches for p in row]).flatten())
        pixel_info = f" | pixel values: {list(all_pixel_vals)}"

    # Display panel
    line1 = f"Original: {orig_shape} → Resized: {img.shape[:2]} → Cropped: {cropped.shape}"
    line2 = f"Patches created: {patch_count} total{pixel_info}"
    console.print(Panel(f"{line1}\n{line2}", title=f"[IMG] {image_name}", expand=False))


# ------------------------------ Batch Processing ------------------------------ #
def batch_process(images, masks, patch_size=256, target_shape=(1024, 1440)):
    """Process all images and masks in dataset"""
    rule("IMAGE PROCESSING START")
    for img_path in images:
        process_single_image(img_path, patch_size=patch_size, target_shape=target_shape)

    rule("MASK PROCESSING START")
    for mask_path in masks:
        process_single_image(mask_path, patch_size=patch_size, target_shape=target_shape)
