#!/usr/bin/env python3
"""
DeepAxon (AUGMENTED)
Train a UNet++ (DeepAxon++) segmentation model with CPU-based patch augmentation,
ReduceLROnPlateau and EarlyStopping callbacks, and a human-readable training log.

Expected folder structure:
training/
  ├── images/
  └── masks/
  (image and mask filenames must match, e.g. image1.png and image1_mask.png)

Main steps:
  1) Resize originals to standard size
  2) Center-crop to multiples of patch size (save to parentfolder/cropped)
  3) Split into patches  (save to parentfolder/cropped/patches and then load into RAM for further training)
  4) Apply lightweight CPU augmentations (images only for intensity; masks get geometry)
  5) Train UNet++ with LR scheduling + early stopping
  6) Save model and write a human-friendly log

Usage:
  Call train_model(training_dir, model_path, model_name, ...)
"""

# ------------------------------ Standard Libraries ------------------------------------ #
import os    
import re                              
import random   
from datetime import datetime                     

# ------------------------------ Third-party Libraries --------------------------------- #
import cv2                              # Computer vision (OpenCV) 
import numpy as np                      # Numerical operations
from patchify import patchify           # Split images into smaller patches

#--------------------------------Rich Console (CLI Output)-------------------------------#
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()                      # Initialize global console

# ------------------------------ GPU Selection ----------------------------------------- #
# Ask whether to try GPU; otherwise force CPU (works fine on CPU too).
use_gpu = input("Use GPU acceleration if available? [y/N]: ").strip().lower()

if use_gpu not in ["y", "yes"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"       # Force CPU-only mode
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"        # Suppress CUDA warnings, keep critical error message
    console.print(Panel.fit("[bold yellow]Running DeepAxon on CPU only[/bold yellow]", border_style="red"))
else:
    console.print(Panel.fit("[bold green]Attempting to use GPU if available...[/bold green]", border_style="green"))

# -------------------- Keras / Sklearn Imports  /  Tensorflow --------------------------- #
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from sklearn.model_selection import train_test_split

# ------------------------------ Local Imports ----------------------------------------- #
from model import deepaxon_plusplus_model               # UNET++ model architecture definition
from resize import resize_img                           # Custom function for standard resizing


# ====================================================================================== #
#                                    DICE & IOU Metrics                                  #
# ====================================================================================== #
"""
train using CE + Dice loss.

track both Dice + IoU on training batches.

monitor validation IoU for callbacks like early stopping and LR reduction.

report Dice + IoU for both sets at the end.
"""
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def iou_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def combined_loss(y_true, y_pred):
    bce = CategoricalCrossentropy()(y_true, y_pred)
    dsc = dice_loss(y_true, y_pred)
    return bce + dsc

# ====================================================================================== #
#                                   UTILITY HELPERS                                      #
# ====================================================================================== #
def count_files(folder_path):
    """
    Count non-hidden files in a folder.
    """
    return len([f for f in os.listdir(folder_path) 
                if os.path.isfile(os.path.join(folder_path, f)) and not f.startswith('.')])
    
# ------------------------------ Dataset Loaders -------------------------------------- #
def verify_and_load_dataset(images_dir, masks_dir, img_ext=".tif", mask_ext=".png"):
    """
    Verify and load image–mask pairs by filename matching.
    Accepts any common image format (.png, .jpg, .jpeg, .tif, .tiff, .bmp, .gif).
    Ensures only 1:1 pairs are returned and prints a dataset summary.

    Returns:
        matched_images: list of valid image paths
        matched_masks: list of valid mask paths
    """
    console.rule("[bold cyan]VERIFYING IMAGE–MASK PAIRS[/bold cyan]")

    VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif")

    # ---------- Helper: list valid files ----------
    def list_valid_files(folder):
        return [
            f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
            and not f.startswith(".")
            and f.lower().endswith(VALID_EXTS)
        ]

    # ---------- Helper: natural sort ----------
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    # ---------- Get and sort image/mask files ----------
    image_files = list_valid_files(images_dir)
    mask_files  = list_valid_files(masks_dir)

    image_files.sort(key=natural_sort_key)
    mask_files.sort(key=natural_sort_key)

    # ---------- Basename matching ----------
    def get_basename_no_ext(f):
        return os.path.splitext(f)[0]

    image_basenames = [get_basename_no_ext(f) for f in image_files]
    mask_basenames  = [get_basename_no_ext(f) for f in mask_files]

    matched_images, matched_masks = [], []

    for img_file, img_name in zip(image_files, image_basenames):
        if img_name in mask_basenames:
            mask_file = next(f for f in mask_files if get_basename_no_ext(f) == img_name)
            matched_images.append(os.path.join(images_dir, img_file))
            matched_masks.append(os.path.join(masks_dir, mask_file))
        else:
            console.print(f"[yellow]⚠ Skipping {img_file} — no matching mask found.[/yellow]")

    # ---------- Verify alignment ----------
    missing_masks = [n for n in image_basenames if n not in mask_basenames]
    missing_images = [n for n in mask_basenames if n not in image_basenames]

    # ---------- Summary panel ----------
    summary_panel = Panel.fit(
        f"[bold white]Found pairs:[/bold white] [cyan]{len(matched_images)}[/cyan]\n"
        f"[bold white]Missing masks:[/bold white] {len(missing_masks)}\n"
        f"[bold white]Missing images:[/bold white] {len(missing_images)}",
        title="[bold magenta]Dataset Verification[/bold magenta]",
        border_style="magenta",
    )
    console.print(summary_panel)

    return matched_images, matched_masks

def verify_patches(image_patch_dir, mask_patch_dir):
    console.rule("[bold yellow]VERIFYING PATCH ALIGNMENT[/bold yellow]")

    img_patches = sorted(
        [f for f in os.listdir(image_patch_dir) if f.endswith(".png") and not f.startswith(".")]
    )
    mask_patches = sorted(
        [f for f in os.listdir(mask_patch_dir) if f.endswith(".png") and not f.startswith(".")]
    )

    if len(img_patches) == len(mask_patches):
        console.print(
            Panel.fit(
                f"[bold green]✅ Patch alignment verified[/bold green]\n"
                f"[white]Images:[/white] {len(img_patches)} | [white]Masks:[/white] {len(mask_patches)}",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel.fit(
                f"[bold red]⚠ Patch mismatch detected![/bold red]\n"
                f"[white]Images:[/white] {len(img_patches)} | [white]Masks:[/white] {len(mask_patches)}",
                border_style="red",
            )
        )

    return len(img_patches), len(mask_patches)


# ====================================================================================== #
#                                  IMAGE PROCESSING                                      #
# ====================================================================================== #
def crop_center_array(img_array, patch_size=256):
    """
    Center-crop a numpy array so BOTH dimensions are exact multiples of patch_size.

    Example:
        If resized image is 1024×1440 and patch_size=256:
            New height = 1024 (already multiple)
            New width  = 1024 (1440 cropped to 1024)
    """
    height, width = img_array.shape[:2]
    new_height = (height // patch_size) * patch_size
    new_width  = (width  // patch_size) * patch_size
    top = (height - new_height) // 2
    left = (width  - new_width)  // 2
    cropped = img_array[top:top+new_height, left:left+new_width]
    return cropped

def patch_array_and_save(img_array, save_dir, image_name, patch_size=256):
    """
    Split an array into patches and save as PNG files. Returns (patch_count, num_channels).
    """
    os.makedirs(save_dir, exist_ok=True)
    patches = patchify(img_array, (patch_size, patch_size), step=patch_size)
    patch_count = 0

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch_array = patches[i, j]
            if patch_array.ndim == 3 and patch_array.shape[0] == 1:
                patch_array = patch_array[0]
            patch_path = os.path.join(save_dir, f"{image_name}_{i}{j}.png")
            cv2.imwrite(patch_path, patch_array)
            patch_count += 1

    num_channels = img_array.shape[2] if img_array.ndim == 3 else 1
    #console.print(f"patch {i},{j} shape {patch_array.shape} dtype={patch_array.dtype}")
    return patch_count, num_channels
    
def process_single_image(path, patch_size=256):
    """
    Resize, crop, save cropped, and split into patches. Shows a small panel per image.
    """
    image_name = os.path.splitext(os.path.basename(path))[0]
    parent_dir = os.path.dirname(path)

    original_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    orig_shape = original_img.shape
    orig_channels = 1 if len(orig_shape) == 2 else orig_shape[2]

    resized_img = resize_img(path, is_mask=True if "mask" in parent_dir.lower() else False)
    resized_shape = resized_img.shape
    resized_channels = 1 if len(resized_shape) == 2 else resized_shape[2]

    # Flatten single-channel 3D arrays (HxWx1)
    if resized_img.ndim == 3 and resized_img.shape[2] == 1:
        resized_img = resized_img[:, :, 0]

    cropped_img = crop_center_array(resized_img, patch_size=patch_size)
    cropped_shape = cropped_img.shape
    cropped_channels = 1 if len(cropped_shape) == 2 else cropped_shape[2]

    cropped_dir = os.path.join(parent_dir, "cropped")
    os.makedirs(cropped_dir, exist_ok=True)
    cropped_path = os.path.join(cropped_dir, os.path.basename(path))
    cv2.imwrite(cropped_path, cropped_img)

    patch_dir = os.path.join(cropped_dir, "patches")
    patch_count, patch_channels = patch_array_and_save(cropped_img, patch_dir, image_name, patch_size)

    # Render all in one Rich panel
    panel = Panel.fit(
        "\n".join([
            f"Original: {orig_shape} → {orig_channels} ch",
            f"Resized:  {resized_shape} → {resized_channels} ch",
            f"Cropped:  {cropped_shape} → {cropped_channels} ch",
            f"Patches:  {patch_count} total → {patch_channels} ch each"
        ]),
        title=f"[bold cyan][IMG][/bold cyan] {image_name}",
        border_style="cyan"
    )
    console.print(panel)
    
def batch_patch(images, masks, patch_size=256):
    """
    Apply full processing to ALL images AND masks (resizing/cropping/patching).
    """
    console.rule("[bold yellow]IMAGE PROCESSING START[/bold yellow]")
    for p in images:
        process_single_image(p, patch_size=patch_size)
        
    console.rule("[bold yellow]MASK PROCESSING START[/bold yellow]")
    for p in masks:
        process_single_image(p, patch_size=patch_size)
    
def get_images(patch_path: str) -> np.ndarray:
    """
    Read all image files under 'patch_path' (sorted) and return as a numpy array.
    
    :param patch_path: A path (string or object) 
    
    :returns: Numpy Array; all images in the patches folder
    """
    #go to every list in the patch folder and append it to the list
    files = [os.path.join(patch_path, f) for f in os.listdir(patch_path)
                   if os.path.isfile(os.path.join(patch_path, f)) and not f.startswith('.')]
    files.sort()
    return np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files])

def base_label(train_masks):
    """
    Convert grayscale mask pixels into class IDs (0,1,2).
    Any unknown value defaults to background (0).
    
    :param train_masks: Numpy Array; training mask images
    
    :returns: Numpy Array; training mask images with values being 0,1,2 instead of whatever colors it was
    """
    # Ensure integer mask values
    train_masks = np.round(train_masks).astype(np.uint8)
    out = np.zeros_like(train_masks, dtype=np.uint8)
    out[train_masks == 127] = 1
    out[train_masks == 128] = 1
    out[train_masks == 255] = 2
    return out


# ===================================================================================== #
#                                   AUGMENTATION                                        #
# ===================================================================================== #
def augment_dataset_np(X,
                       Y,
                       aug_prob=0.25,
                       p_flip_h=0.5,
                       p_flip_v=0.5,
                       p_rotate=0.3,
                       p_brightness=0.25,
                       p_gamma=0.2,
                       p_noise=0.1):
    """
    Perform lightweight, on-the-fly CPU augmentations on (X, Y) batches.

    Augmentation strategy:
    - A global gatekeeper (`aug_prob`) determines whether any augmentation is applied to a given patch.
    - If augmentation is triggered, each augmentation type is applied independently with its own probability:
        - Random horizontal and vertical flips
        - Small random rotation (-10° to +10°) 
            - Masks use nearest-neighbor interpolation
            - Images use linear interpolation
        - Small random brightness and contrast adjustments (applied to images only)
        - Small random gamma adjustment (applied to images only)
        - Small additive Gaussian noise (applied to images only)
        
    Masks only get geometric transforms. Returns (X_out, Y_out, flags).
    """
    X_out, Y_out, flags = [], [], []
    
    for img, mask in zip(X, Y):
        img_out, mask_out = img.copy(), mask.copy()
        augmented = False
        
        # Decide if augmentation occurs
        if random.random() < aug_prob:

            # Random horizontal flip
            if random.random() < p_flip_h:
                img_out = np.flip(img_out, axis=1)
                mask_out = np.flip(mask_out, axis=1)
                augmented = True

            # Random vertical flip
            if random.random() < p_flip_v:
                img_out = np.flip(img_out, axis=0)
                mask_out = np.flip(mask_out, axis=0)
                augmented = True

            # Random small rotation (-10 to +10 degrees)
            if random.random() < p_rotate:
                angle = random.uniform(-10, 10)
                M = cv2.getRotationMatrix2D((img_out.shape[1]/2, img_out.shape[0]/2), angle, 1)

                # Image: linear interpolation
                img_out = cv2.warpAffine(img_out, M, (img_out.shape[1], img_out.shape[0]),
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

                # Mask: nearest-neighbor, safe cast & clip
                mask_out = mask_out.astype(np.uint8)
                mask_out = cv2.warpAffine(mask_out, M, (mask_out.shape[1], mask_out.shape[0]),
                                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
                mask_out = np.round(mask_out).astype(np.uint8)
                mask_out = np.clip(mask_out, 0, 2)
                augmented = True

            # Random brightness / contrast jitter (images only)
            if random.random() < p_brightness:
                alpha = random.uniform(0.95, 1.05)  # contrast
                beta = random.uniform(-0.05, 0.05) # brightness in 0–1 scale
                img_out = np.clip(img_out * alpha + beta, 0.0, 1.0)
                augmented = True

            # Random Gamma jitter (images only)
            if random.random() < p_gamma:
                gamma = random.uniform(0.95, 1.05)
                img_out = np.clip(img_out ** gamma, 0.0, 1.0)
                augmented = True

            # Random tiny Gaussian noise (images only)
            if random.random() < p_noise:
                noise = np.random.normal(0, 0.01, img_out.shape)  # small noise in 0–1 range
                img_out = np.clip(img_out + noise, 0.0, 1.0)
                augmented = True
            
        # If image/mask is 2D (H x W), add channel dimension
        if img_out.ndim == 2:
            img_out = np.expand_dims(img_out, axis=-1)  # shape: H x W x 1
        if mask_out.ndim == 2:
            mask_out = np.expand_dims(mask_out, axis=-1)

        # Always append, whether augmented or not
        X_out.append(img_out)
        Y_out.append(mask_out)
        flags.append(augmented)

    return np.array(X_out), np.array(Y_out), np.array(flags)


# ===================================================================================== #
#                               DATA AUGMENTATION TOGGLE                                #
# ===================================================================================== #

use_aug = input("Use data augmentation? [y/N]: ").strip().lower() in ["y", "yes"]

# Set default augmentation probability if enabled
if use_aug:
    aug_prob = 0.25  # <-- adjust default here if needed
    print("\n Data augmentation ENABLED")
    print(f"   • Global augmentation probability: {aug_prob}")
    print("   • Random flips, small rotations, brightness/gamma jitter, light noise\n")
else:
    aug_prob = 0.0
    print("\n Data augmentation DISABLED")
    print("   • Model will train on original, unmodified patches only\n")
    
# ===================================================================================== #
#                                  EPOCH LOG CALLBACK                                   #
# ===================================================================================== #

class TrainingLogger(Callback):
    """
    Unified logger that combines:
      - Epoch metrics (loss, accuracy, val_loss, val_accuracy, LR)
      - Augmentation stats (total_aug, avg_per_batch, min_batch, max_batch)
      - When LR reduction is triggered LR will be displayed in bold red
    
    Writes a neatly formatted table to both console and file:
    (EXAMPLE)
    Epoch |    Loss |   Acc | Val_Loss | Val_Acc |    LR    | Aug(total/avg/min/max)
    -------------------------------------------------------------------------------
        1 |  1.1780 | 0.671 |  1.4003  | 0.4071  | 1.00e-03 | 76 / 5.43 / 3 / 8
    """
    def __init__(self, log_file_path, hyperparams, aug_counts_list, steps_per_epoch):
        super().__init__()
        self.console = Console()
        self.log_file_path = log_file_path
        self.hyperparams = hyperparams
        self.aug_counts_list = aug_counts_list
        self.steps_per_epoch = steps_per_epoch
        self.prev_lr = None                     # track last LR for comparison
        self.table = Table(
            title="[bold blue]DeepAxon++ Training Progress[/bold blue]",
            header_style="bold cyan",
            show_lines=False
        )
        for header in ["Epoch", "Loss", "Acc", "Val_Loss", "Val_Acc", "LR", "Aug (T/A/Mi/Ma)"]:
            self.table.add_column(header, justify="center")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        try:
            lr = float(self.model.optimizer._decayed_lr(tf.float32).numpy())
        except Exception:
            lr = float('nan')

        # Detect LR change
        lr_changed = (self.prev_lr is not None and abs(lr - self.prev_lr) > 1e-12)
        self.prev_lr = lr

        # Only color the LR if it changed (no other formatting affected)
        lr_str = f"[bold red]{lr:.2e}[/bold red]" if lr_changed else f"{lr:.2e}"

        start = epoch * self.steps_per_epoch
        end = min(start + self.steps_per_epoch, len(self.aug_counts_list))
        counts = self.aug_counts_list[start:end]

        if len(counts) > 0:
            total_aug = sum(counts)
            avg_aug = np.mean(counts)
            min_aug = np.min(counts)
            max_aug = np.max(counts)
            aug_str = f"{total_aug:.0f}/{avg_aug:.2f}/{min_aug:.0f}/{max_aug:.0f}"
        else:
            aug_str = "-/-/-/-"

        self.table.add_row(
            str(epoch + 1),
            f"{logs.get('loss', float('nan')):.4f}",
            f"{logs.get('accuracy', float('nan')):.4f}",
            f"{logs.get('val_loss', float('nan')):.4f}",
            f"{logs.get('val_accuracy', float('nan')):.4f}",
            lr_str,
            aug_str
        )
        # write plain numeric LR to log file (no rich markup)
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            if epoch == 0:
                f.write("\nEpoch | Loss | Acc | Val_Loss | Val_Acc | LR | Aug(T/A/Mi/Ma)\n")
                f.write("-" * 75 + "\n")
            f.write(
                f"{epoch + 1:5d} | "
                f"{logs.get('loss', float('nan')):7.4f} | "
                f"{logs.get('accuracy', float('nan')):5.3f} | "
                f"{logs.get('val_loss', float('nan')):8.5f} | "
                f"{logs.get('val_accuracy', float('nan')):6.4f} | "
                f"{lr:8.2e} | {aug_str}\n"
            )

    def on_train_end(self, logs=None):
        self.console.rule("[bold blue]TRAINING SUMMARY TABLE[/bold blue]")
        self.console.print(self.table)
            
            
# ===================================================================================== #
#                             MAIN MODEL TRAINER                                        #
# ===================================================================================== #
def train_model(training_dir,
                model_path,
                model_name,
                epochs=200,
                test_fraction=0.3,
                img_ext=".tif",
                mask_ext=".png",
                patch_size=256):
    """
    Train and save DeepAxon++ model with learning rate scheduler, optional early stopping, and human-readable logs.
    Augmentations are applied dynamically per batch during training.
    
    :param training_dir: Folder containing 'images/' and 'masks/'.
    :param model_path: Folder to save trained model.
    :param model_name: Filename for saved model.
    :param epochs: Maximum number of epochs to train.
    :param test_fraction: Fraction of dataset to reserve for testing.
    :param img_ext: Image file extension (default ".tif").
    :param mask_ext: Mask file extension (default ".png").
    
    :returns: Trained Keras model
    """
    # List to store number of patches and augmentations per batch
    augmented_counts_per_batch = []
    patches_per_batch = [] 
    
    # -------------------- Resolve folders -------------------- #
    images_dir = os.path.join(training_dir, "images")
    masks_dir = os.path.join(training_dir, "masks")

    # -------------------- Load dataset ----------------------- #
    images, masks = verify_and_load_dataset(images_dir, masks_dir, img_ext, mask_ext)

    # -------------------- Crop & patch ----------------------- #
    batch_patch(images, masks, patch_size=patch_size)

    # Patch folders where patches were written
    image_patch_path = os.path.join(images_dir, "cropped", "patches")
    mask_patch_path  = os.path.join(masks_dir, "cropped", "patches")
    verify_patches(image_patch_path, mask_patch_path)

    # -------------------- Load patches into RAM -------------- #
    X = get_images(image_patch_path)  # returns (n, H, W)
    Y = base_label(get_images(mask_patch_path))

    #print("X dtype, shape, min/max:", X.dtype, X.shape, X.min(), X.max())
    #print("Y dtype, shape, unique values:", Y.dtype, Y.shape, np.unique(Y)[:20])

    # Ensure 4D tensors (n_samples, H, W, 1)
    if X.ndim == 3:
        X = np.expand_dims(X, axis=-1)
    if Y.ndim == 3:
        Y = np.expand_dims(Y, axis=-1)
        
    # -------------------- Normalize images -------------------- #
    X = X.astype(np.float32) / 255.0  # normalize training + validation

    # -------------------- Train / test split ----------------- #
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_fraction, random_state=0)
    
    # ────────────────────────────────────── RECOMMEND BATCH SIZE ───────────────────────────────────── #
    def recommend_batch_size(num_patches):
        """
        Recommend an efficient batch size that minimizes leftover patches,
        preferring smaller batch sizes when multiple divide evenly.

        Args:
            num_patches (int): Number of training patches.

        Returns:
            tuple: (recommended_batch_size, remainder, efficiency)
        """
        possible_sizes = [4, 8, 16, 32, 64]
        perfect_divisors = [s for s in possible_sizes if num_patches % s == 0]

        if perfect_divisors:
            # Prefer the smallest batch size that divides evenly
            recommended = min(perfect_divisors)
            remainder = 0
        else:
            # Otherwise pick the one with the smallest remainder
            remainder = float('inf')
            recommended = possible_sizes[0]
            for s in possible_sizes:
                r = num_patches % s
                if r < remainder:
                    remainder = r
                    recommended = s

        efficiency = (1 - (remainder / num_patches)) * 100

        return recommended, remainder, efficiency

    # ────────────────────────────────────── COMPUTE AND PROMPT USER ───────────────────────────────────── #
    train_patches = len(X_train)
    val_patches = len(X_test)

    recommended_batch_size, remainder, efficiency = recommend_batch_size(train_patches)

    console.rule("[bold green]TRAINING SETUP[/bold green]")
    console.print(f"📦 Total training patches: [bold cyan]{train_patches}[/bold cyan]")
    console.print(f"🧪 Validation patches:     [bold cyan]{val_patches}[/bold cyan]")
    console.print(f"💡 Recommended batch size: [bold yellow]{recommended_batch_size}[/bold yellow]")
    console.print(f"🔢 Batches per epoch:      [bold magenta]{train_patches // recommended_batch_size}[/bold magenta]")
    console.print(f"📈 Batch efficiency:       [bold green]{efficiency:.1f}%[/bold green]\n")
    
    if remainder == 0:
        console.print(f"💡 [italic]Perfect fit — no leftover patches with batch size {recommended_batch_size}.[/italic]")
    else:
        console.print(
            f"💡 [italic]Recommended batch size minimizes leftover patches "
            f"({remainder} unused if {recommended_batch_size} chosen).[/italic]"
        )

    # Prompt user with proper override logic
    use_recommended = input(f"Use recommended batch size ({recommended_batch_size})? [Y/n]: ").strip().lower()

    if use_recommended in ["", "y", "yes"]:
        batch_size = recommended_batch_size
    else:
        while True:
            custom_input = input("Enter custom batch size (positive integer): ").strip()
            try:
                batch_size = int(custom_input)
                if batch_size <= 0:
                    raise ValueError

                # Recalculate metrics for the custom batch size
                remainder = train_patches % batch_size
                efficiency = (1 - (remainder / train_patches)) * 100
                batches_per_epoch = train_patches // batch_size

                console.rule("[bold yellow]CUSTOM BATCH SIZE SUMMARY[/bold yellow]")
                console.print(f"💪 Custom batch size: [bold yellow]{batch_size}[/bold yellow]")
                console.print(f"🔢 Batches per epoch: [bold magenta]{batches_per_epoch}[/bold magenta]")
                console.print(f"📈 Batch efficiency:  [bold green]{efficiency:.1f}%[/bold green]")
                console.print(f"🧩 Leftover patches:  [bold cyan]{remainder}[/bold cyan]\n")

                confirm = input("Confirm use of custom batch size? [Y/n]: ").strip().lower()
                if confirm in ["", "y", "yes"]:
                    break
                else:
                    console.print("[italic yellow]Re-enter custom batch size...[/italic yellow]")
            except ValueError:
                print("⚠️ Invalid input. Please enter a positive integer.")

    console.print(f"✅ Using batch size: [bold green]{batch_size}[/bold green]\n")  
    
    # -------------------- One-hot encode test masks ---------- #
    y_test_flat = np.squeeze(y_test, axis=-1)           # -> (N, H, W)
    y_test_cat  = to_categorical(y_test_flat, num_classes=3)  # -> (N, H, W, 3)

    # -----------------AUG Data Generator --------------------- #
    augmented_counts_per_batch = []
    
    # Soft coded for now, once finalized can integrate into model.py with tensorflow
    def train_generator(X, Y, batch_size, augmented_counts_per_batch, patches_list, **aug_kwargs):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        while True:
            np.random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                X_batch, Y_batch = X[batch_idx], Y[batch_idx]
                
                # Apply augmentation on-the-fly
                X_aug, Y_aug, flags = augment_dataset_np(X_batch, Y_batch, aug_prob=aug_prob)
                
                # Record augmented count for this batch
                augmented_counts_per_batch.append(np.sum(flags))
            
                # One-hot encode masks
                Y_aug_squeezed = np.squeeze(Y_aug, axis=-1)        # -> (batch, H, W)
                Y_aug_cat = to_categorical(Y_aug_squeezed, num_classes=3)  # -> (batch, H, W, 3)
                patches_list.append(len(batch_idx))
                yield X_aug, Y_aug_cat

    train_gen = train_generator(X_train, y_train,
                                batch_size=batch_size,
                                augmented_counts_per_batch=augmented_counts_per_batch,
                                patches_list=patches_per_batch)
    
    steps_per_epoch = int(np.ceil(X_train.shape[0] / batch_size))   # sum = # patches augmented in this batch
    
    # -------------------- Build model ------------------------ #
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = X_train.shape[1:4]
    model = deepaxon_plusplus_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=combined_loss,
        metrics=[dice_coef, iou_coef]
    )

    # ---------------------- Callbacks ------------------------ #
    # Reduce learning rate when validation loss plateaus
    """ lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',     # Track validation loss
        factor=0.5,             # Reduce LR by half when plateau occurs
        patience=15,            # Wait 15 epochs without improvement before reducing LR
        min_delta=0.001,        # Minimum change required 0.001
        cooldown=5,             # Wait 5 epochs after reduction before restarting patience
        verbose=1, 
        min_lr=1e-6             # Don't go below this learning rate
    ) """

    """ # Early stopping to prevent overfitting and restore best weights
    early_stop = EarlyStopping(
        monitor='val_loss',     # Stop when validation loss stops improving
        patience=40,            # Wait 40 epochs without improvement before stopping
        min_delta=0.001,         # Same noise filter as for LR reduction
        restore_best_weights=True, 
        verbose=1
    ) """
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_iou_coef',     # Track validation IoU instead of val_loss
        factor=0.5,                 # Reduce LR by half when plateau occurs
        patience=15,                # Wait 15 epochs without improvement
        min_delta=0.001,            # Minimum IoU change to be considered improvement
        cooldown=5,                 
        verbose=1, 
        mode='max',                 # Because higher IoU is better
        min_lr=1e-6
    )

    # Early stopping to prevent overfitting and restore best weights
    early_stop = EarlyStopping(
        monitor='val_iou_coef',     # Stop when validation IoU stops improving
        mode='max',                 # Higher is better
        patience=40,
        min_delta=0.001,
        restore_best_weights=True, 
        verbose=1
    )
    
    # -------------------- Human-readable log ----------------- #
    logs_dir = os.path.join(model_path, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"{model_name}_training_log.txt")

    # Hyperparameters to record at the top of the log
    hyperparams = {
        "Model name": model_name,
        "Start time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "Architecture": "UNet++ (DeepAxon++)",
        "Input size (H×W×C)": f"{IMG_HEIGHT}×{IMG_WIDTH}×{IMG_CHANNELS}",
        "Classes": 3,
        "Patch size": patch_size,
        "Batch size": batch_size,
        "Epoch limit": epochs,
        "Test fraction": test_fraction,
        "Augmentations": "H/V flip, small rotation, brightness/contrast (img only), gamma (img only), Gaussian noise (img only)",
        "ReduceLROnPlateau": "factor=0.5, patience=15, min_delta=0.001, min_lr=1e-6, monitor=val_loss",
        "EarlyStopping": "patience=40, min_delta=0.001, restore_best_weights=True, monitor=val_loss",
        "Device": "GPU" if use_gpu in ['y','yes'] else "CPU-only",
        "Train samples": X_train.shape[0],
        "Test samples": X_test.shape[0],
    }

    merged_logger = TrainingLogger(
        log_file_path=log_file,
        hyperparams=hyperparams,
        aug_counts_list=augmented_counts_per_batch,
        steps_per_epoch=steps_per_epoch
    )
    
    # ----T/T Split & Aug Status------------------------------#
    total_patches = len(X)
    num_train = len(X_train)
    num_val = len(X_test)

    num_classes = len(np.unique(Y))
    aug_status = "ON" if aug_prob > 0.0 else "OFF"

    summary_line = (
        f"[bold green]✅ {total_patches} patches[/bold green] → "
        f"[cyan]{num_train} train[/cyan] / [magenta]{num_val} val[/magenta] | "
        f"[yellow]{num_classes} classes[/yellow] | "
        f"[bold blue]aug={aug_status}[/bold blue]"
    )

    console.rule("DATASET SUMMARY", style="green")
    console.print(summary_line)
    console.rule(style="green")

    # -------------------- Train ------------------------------ #
    console.rule("[bold magenta]TRAINING INITIALIZATION[/bold magenta]")
    
    if aug_prob > 0.0:
        console.print("[bold cyan]Augmentation ON:[/bold cyan] random on-the-fly transformations enabled.")
    else:
        console.print("[bold yellow]Augmentation OFF:[/bold yellow] training on raw patches only.")
    
    #console.print("X_train:", X_train.shape, X_train.dtype, X_train.min(), X_train.max())
    #console.print("y_train unique labels (sample):", np.unique(y_train)[:10])
    #console.print("y_train dtype/shape:", y_train.dtype, y_train.shape)
    # after one to_categorical check shape:
    #console.print("y_test_cat shape:", y_test_cat.shape)

    # --- Write hyperparameters and header to log before training ---
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("DEEPAXON++ TRAINING LOG\n")
        f.write("="*70 + "\n\n")
        for k, v in hyperparams.items():
            f.write(f"{k:22}: {v}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("EPOCH METRICS\n")
        f.write("="*70 + "\n")
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_test, y_test_cat),
        epochs=epochs,
        verbose=1,
        shuffle=False,
        callbacks=[lr_scheduler, early_stop, merged_logger]
    )

    # -------------------- Save model ------------------------- #
    os.makedirs(model_path, exist_ok=True)
    full_model_path = os.path.join(model_path, model_name + ".keras")
    model.save(full_model_path)

    # -------------------- Compute metrics + duration ---------- #
    console.rule("[bold green]TRAINING COMPLETE[/bold green]")

    # Safely extract key metrics
    final_train_loss = history.history.get('loss', [float('nan')])[-1]
    final_val_loss   = history.history.get('val_loss', [float('nan')])[-1]
    final_train_dice = history.history.get('dice_coef', [float('nan')])[-1]
    final_val_dice   = history.history.get('val_dice_coef', [float('nan')])[-1]
    final_train_iou  = history.history.get('iou_coef', [float('nan')])[-1]
    final_val_iou    = history.history.get('val_iou_coef', [float('nan')])[-1]

    # Compute elapsed time (in minutes + seconds)
    end_time = datetime.now()
    start_time_str = hyperparams.get("Start time")
    try:
        start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
        elapsed = end_time - start_time
        elapsed_minutes = int(elapsed.total_seconds() // 60)
        elapsed_seconds = int(elapsed.total_seconds() % 60)
        duration_str = f"{elapsed_minutes}m {elapsed_seconds}s"
    except Exception:
        duration_str = "N/A"
        
    # -------------------- Patch statistics -------------------- #
    total_patches_processed = int(np.sum(patches_per_batch))
    total_patches_augmented = int(np.sum(augmented_counts_per_batch))
        
    # ---- Console Summary Panel ---- #
    summary_panel = Panel.fit(
        f"[bold white]Model saved at:[/bold white] [green]{full_model_path}[/green]\n"
        f"[bold cyan]Final Training Loss:[/bold cyan] {final_train_loss:.4f}\n"
        f"[bold cyan]Final Validation Loss:[/bold cyan] {final_val_loss:.4f}\n\n"
        f"[bold green]Final Training Dice:[/bold green] {final_train_dice:.4f}\n"
        f"[bold green]Final Validation Dice:[/bold green] {final_val_dice:.4f}\n"
        f"[bold magenta]Final Training IoU:[/bold magenta] {final_train_iou:.4f}\n"
        f"[bold magenta]Final Validation IoU:[/bold magenta] {final_val_iou:.4f}\n\n"
        f"[bold yellow]Total Training Time:[/bold yellow] {duration_str}\n"
        f"[bold blue]Total Patches Processed:[/bold blue] {total_patches_processed:,}\n"
        f"[bold blue]Total Patches Augmented:[/bold blue] {total_patches_augmented:,}",
        title="[bold green]DeepAxon++ Summary[/bold green]",
        border_style="green",
        padding=(1, 2)
    )
    console.print(summary_panel)
    
    # ---- Append summary to log file ---- #
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("FINAL SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Final training loss:       {final_train_loss:.4f}\n")
        f.write(f"Final validation loss:     {final_val_loss:.4f}\n")
        f.write(f"Final training Dice:       {final_train_dice:.4f}\n")
        f.write(f"Final validation Dice:     {final_val_dice:.4f}\n")
        f.write(f"Final training IoU:        {final_train_iou:.4f}\n")
        f.write(f"Final validation IoU:      {final_val_iou:.4f}\n")
        f.write(f"Total training time:       {duration_str}\n")
        f.write(f"Total patches processed:   {total_patches_processed:,}\n")
        f.write(f"Total patches augmented:   {total_patches_augmented:,}\n")
        f.write(f"Model saved at:            {full_model_path}\n")
        f.write("=" * 80 + "\n")

    return model