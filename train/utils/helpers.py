# train/utils/helpers.py
"""
Helper functions for DeepAxon++
"""

import os
from train.utils.console_utils import info, warn, error, success


# ---------------------- Directory/File Helpers ---------------------- #
def ensure_dir(path):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)
    return path

def list_files(dir_path, extensions=None):
    """Return sorted list of files optionally filtered by extensions; skip hidden/system files."""
    files = []
    for f in os.listdir(dir_path):
        if f.startswith('.') or f.lower().endswith('db'):  # skip hidden/system
            continue
        full_path = os.path.join(dir_path, f)
        if os.path.isfile(full_path):
            if extensions is None or os.path.splitext(f)[1].lower() in extensions:
                files.append(full_path)
    return sorted(files)


# ---------------------- User Input Helpers ------------------------- #
def get_valid_path(prompt):
    """Ask user for a folder path until a valid one is provided."""
    while True:
        path = input(prompt).strip()
        if os.path.isdir(path):
            return path
        error(f"Path not found: {path}. Please enter a valid folder path.")

def get_int_input(prompt, default, min_value=1):
    """Ask for a positive integer input, with default if empty."""
    while True:
        user_input = input(f"{prompt} (press Enter for default={default}): ").strip()
        if not user_input:
            return default
        try:
            value = int(user_input)
            if value < min_value:
                warn(f"Please enter an integer >= {min_value}.")
                continue
            return value
        except ValueError:
            error("Invalid input. Please enter an integer.")

def get_float_input(prompt, default, min_value=0.0, max_value=1.0):
    """Ask the user for a float input. Return default if empty."""
    while True:
        user_input = input(f"{prompt} (press Enter for default={default}): ").strip()
        if not user_input:
            return default
        try:
            value = float(user_input)
            if not (min_value <= value <= max_value):
                warn(f"Please enter a decimal number between {min_value} and {max_value}.")
                continue
            return value
        except ValueError:
            error("Invalid input. Please enter a decimal number.")

def get_training_dir():
    """Ask for the training folder and ensure it has 'images' and 'masks' subfolders."""
    while True:
        training_dir = get_valid_path("Input the path to the training folder that holds the images and masks: ")
        images_dir = os.path.join(training_dir, "images")
        masks_dir = os.path.join(training_dir, "masks")
        if not os.path.isdir(images_dir):
            warn(f"'images' subfolder not found in {training_dir}.")
            continue
        if not os.path.isdir(masks_dir):
            warn(f"'masks' subfolder not found in {training_dir}.")
            continue
        return training_dir

def get_model_dir():
    """Ask for a model save folder and create it if it doesn't exist."""
    while True:
        model_dir = input("Input the path of the folder where the model will be saved: ").strip()
        if not model_dir:
            model_dir = os.getcwd()
            info(f"No path entered. Using current directory: {model_dir}")
        try:
            os.makedirs(model_dir, exist_ok=True)
            return model_dir
        except Exception as e:
            error(f"Failed to create/access folder: {e}")
            
            
# ------------------------------ Data Augmentation Logic --------------------------- #
def compute_aug_prob(use_aug: bool, default_prob=0.25):
    """
    Convert user choice into augmentation probability and summary text.
    Returns:
        aug_prob (float), summary_lines (list[str])
    """
    lines = []
    if use_aug:
        aug_prob = default_prob
        lines.append("Data augmentation ENABLED")
        lines.append(f"  • Global augmentation probability: {aug_prob}")
        lines.append("  • Random flips, small rotations, brightness/gamma jitter, light noise")
    else:
        aug_prob = 0.0
        lines.append("Data augmentation DISABLED")
        lines.append("  • Model will train on original, unmodified patches only")
    return aug_prob, lines


# ------------------------------ Batch Size Logic --------------------------- #
def compute_batch_size(num_patches: int, desired_batch=None):
    """
    Determine recommended batch size, efficiency, and leftover patches.
    Returns:
        batch_size, remainder, efficiency
    """
    possible_sizes = [4, 8, 16, 32, 64]
    perfect_divisors = [s for s in possible_sizes if num_patches % s == 0]

    if desired_batch:
        batch_size = desired_batch
        remainder = num_patches % batch_size
    elif perfect_divisors:
        batch_size = min(perfect_divisors)
        remainder = 0
    else:
        batch_size = possible_sizes[0]
        remainder = num_patches % batch_size
        for s in possible_sizes:
            r = num_patches % s
            if r < remainder:
                remainder = r
                batch_size = s

    efficiency = (1 - remainder / num_patches) * 100
    return batch_size, remainder, efficiency

def count_patches(patch_dir):
    return len([
        f for f in os.listdir(patch_dir)
        if os.path.isfile(os.path.join(patch_dir, f))
        and not f.startswith(".")
    ])
