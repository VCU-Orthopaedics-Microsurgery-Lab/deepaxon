"""
train/dataset/data_loader.py

Load preprocessed patches from disk into numpy arrays for training.

Supports two dataset split modes:
    Phenotype mode — explicit train/regen/, train/control/, val/regen/, val/control/
                     folders. User controls split. val_fraction ignored.
                     Warns if regen/control patch counts are imbalanced.
    Flat mode      — single train/ folder.
                     If val_-tagged images exist, uses them as val set.
                     Otherwise uses val_fraction to auto-split.

Mode is detected by presence of regen/ and control/ subfolders.
"""

from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split

from utils.helpers import list_files


# ─── Mask encoding ────────────────────────────────────────────────────────────

def base_label(arr: np.ndarray) -> np.ndarray:
    """
    Remap mask pixel values to class indices.
    Pixel value → class index:
        0         → 0 (background)
        127, 128  → 1 (myelin / grey)
        255       → 2 (axon / white)
    """
    pixel_to_class = {0: 0, 127: 1, 128: 1, 255: 2}
    return np.vectorize(lambda x: pixel_to_class.get(int(x), 0))(arr).astype(np.float32)


# ─── Patch loading ────────────────────────────────────────────────────────────

def load_patches(patches_dir: str, is_mask: bool = False) -> np.ndarray:
    """
    Load all patches from a directory into a numpy array.
    Images: L2 normalization axis=1. Masks: base_label() remapping.
    Returns (N, H, W, 1) float32.
    """
    files = list_files(patches_dir, extensions=('.png', '.tif', '.tiff'))
    if not files:
        raise FileNotFoundError(f"No patch files found in {patches_dir}")

    patches = []
    for f in sorted(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Could not load patch, skipping: {f}")
            continue
        patches.append(img)

    arr = np.array(patches, dtype=np.float32)
    arr = np.expand_dims(arr, axis=-1)

    if is_mask:
        arr = base_label(arr)
    else:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        arr   = arr / norms

    return arr


# ── NEW — load patches filtered by source image stem ─────────────────────────
def _load_patches_for_stems(
    patches_dir: Path,
    stems: list[str],
    is_mask: bool = False
) -> np.ndarray:
    """
    Load only patches whose filename starts with one of the given stems.
    Used to separate val_-tagged patches from train patches.

    Args:
        patches_dir: directory containing patch files
        stems:       list of source image stems to include
        is_mask:     True for mask patches

    Returns:
        (N, H, W, 1) float32 array
    """
    all_files = list_files(str(patches_dir), extensions=('.png', '.tif', '.tiff'))
    matched   = [f for f in all_files if any(f.stem.startswith(s) for s in stems)]

    if not matched:
        raise FileNotFoundError(
            f"No patches found for stems {stems} in {patches_dir}"
        )

    patches = []
    for f in sorted(matched):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Could not load patch, skipping: {f}")
            continue
        patches.append(img)

    arr = np.array(patches, dtype=np.float32)
    arr = np.expand_dims(arr, axis=-1)

    if is_mask:
        arr = base_label(arr)
    else:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        arr   = arr / norms

    return arr
# ─────────────────────────────────────────────────────────────────────────────


# ── NEW — recover source image stems from patch filenames ─────────────────────
def _patches_to_image_stems(files: list) -> list[str]:
    """
    Extract unique source image stems from patch filenames.
    Patch naming: {stem}_{row:02d}{col:02d}.png
    e.g. 1501_RMNDist_100X_002_0002.png → 1501_RMNDist_100X_002
    """
    seen  = set()
    stems = []
    for f in files:
        stem = '_'.join(Path(f).stem.split('_')[:-1])  # drop last _RRCC segment
        if stem not in seen:
            seen.add(stem)
            stems.append(stem)
    return sorted(stems)
# ─────────────────────────────────────────────────────────────────────────────


def _load_phenotype_patches(
    base_dir: Path,
    is_mask: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load patches from regen/ and control/ subfolders separately.
    Returns (regen_patches, control_patches).
    """
    regen_dir   = base_dir / 'regen'   / 'cropped' / 'patches'
    control_dir = base_dir / 'control' / 'cropped' / 'patches'
    return (
        load_patches(str(regen_dir),   is_mask=is_mask),
        load_patches(str(control_dir), is_mask=is_mask),
    )


def _check_phenotype_balance(
    n_regen: int,
    n_control: int,
    split_name: str,
    warn_threshold: float = 0.2
) -> str | None:
    if n_regen == 0 or n_control == 0:
        return f"{split_name}: one phenotype has 0 patches — check folder structure."
    total    = n_regen + n_control
    pct_diff = abs(n_regen - n_control) / total
    if pct_diff > warn_threshold:
        majority  = 'regen'   if n_regen > n_control else 'control'
        pct_regen = int(n_regen   / total * 100)
        pct_ctrl  = int(n_control / total * 100)
        return (
            f"{split_name}: imbalanced — {n_regen} regen ({pct_regen}%) vs "
            f"{n_control} control ({pct_ctrl}%). "
            f"Model may skew toward {majority}."
        )
    return None


def detect_split_mode(images_train_dir: Path) -> str:
    """
    Detect dataset split mode based on folder structure.
    Returns 'phenotype' if regen/ and control/ both exist, else 'flat'.
    """
    if (images_train_dir / 'regen').exists() and (images_train_dir / 'control').exists():
        return 'phenotype'
    return 'flat'


def load_all_patches(
    img_train_dir: str,
    mask_train_dir: str,
    img_val_dir: str = None,
    mask_val_dir: str = None,
    val_fraction: float = 0.2,
    split_mode: str = 'flat',
    val_tagged_stems: list[str] = None,  # ← NEW parameter
    log=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train and validation patches.

    Phenotype mode:
        train/regen/cropped/patches/ + train/control/cropped/patches/
        val/regen/ + val/control/ — val_fraction ignored.
        Warns if regen/control counts imbalanced.

    Flat mode + val_ tags:                                       ← NEW
        Patches from val_-tagged images → val set.               ← NEW
        All other patches → train set.                           ← NEW
        val_fraction ignored.                                     ← NEW

    Flat mode + no tags:
        Auto-split using val_fraction (default 0.2).

    Returns:
        (X_train, Y_train, X_val, Y_val) — all (N, H, W, 1) float32
    """
    img_train_dir    = Path(img_train_dir)
    mask_train_dir   = Path(mask_train_dir)
    val_tagged_stems = val_tagged_stems or []                    # ← NEW

    # ── Phenotype mode ────────────────────────────────────────────────────────
    if split_mode == 'phenotype':
        X_regen, X_ctrl = _load_phenotype_patches(img_train_dir,  is_mask=False)
        Y_regen, Y_ctrl = _load_phenotype_patches(mask_train_dir, is_mask=True)

        warn = _check_phenotype_balance(len(X_regen), len(X_ctrl), 'Train')
        if warn:
            if log: log.warn(warn)
            else:   print(f"[WARN] {warn}")

        X_train = np.concatenate([X_regen, X_ctrl], axis=0)
        Y_train = np.concatenate([Y_regen, Y_ctrl], axis=0)
        idx     = np.random.permutation(len(X_train))
        X_train, Y_train = X_train[idx], Y_train[idx]

        img_val_path  = Path(img_val_dir)  if img_val_dir  else None
        mask_val_path = Path(mask_val_dir) if mask_val_dir else None

        if not img_val_path or not mask_val_path:
            raise FileNotFoundError(
                "Phenotype mode requires val/regen/ and val/control/ folders."
            )

        X_val_regen, X_val_ctrl = _load_phenotype_patches(img_val_path,  is_mask=False)
        Y_val_regen, Y_val_ctrl = _load_phenotype_patches(mask_val_path, is_mask=True)

        warn_val = _check_phenotype_balance(len(X_val_regen), len(X_val_ctrl), 'Val')
        if warn_val:
            if log: log.warn(warn_val)
            else:   print(f"[WARN] {warn_val}")

        X_val = np.concatenate([X_val_regen, X_val_ctrl], axis=0)
        Y_val = np.concatenate([Y_val_regen, Y_val_ctrl], axis=0)
        idx_v = np.random.permutation(len(X_val))
        X_val, Y_val = X_val[idx_v], Y_val[idx_v]

        # Log val set composition                                # ← NEW
        if log:                                                  # ← NEW
            val_regen_stems   = _patches_to_image_stems(        # ← NEW
                list_files(str(img_val_path / 'regen' / 'cropped' / 'patches'),  # ← NEW
                           extensions=('.png','.tif','.tiff'))  # ← NEW
            )                                                    # ← NEW
            val_control_stems = _patches_to_image_stems(        # ← NEW
                list_files(str(img_val_path / 'control' / 'cropped' / 'patches'),  # ← NEW
                           extensions=('.png','.tif','.tiff'))  # ← NEW
            )                                                    # ← NEW
            n_val_images = len(val_regen_stems) + len(val_control_stems)  # ← NEW
            n_total      = n_val_images + len(                  # ← NEW
                _patches_to_image_stems(                        # ← NEW
                    list_files(str(img_train_dir / 'regen' / 'cropped' / 'patches'),  # ← NEW
                               extensions=('.png','.tif','.tiff'))  # ← NEW
                )                                               # ← NEW
            ) + len(                                            # ← NEW
                _patches_to_image_stems(                        # ← NEW
                    list_files(str(img_train_dir / 'control' / 'cropped' / 'patches'),  # ← NEW
                               extensions=('.png','.tif','.tiff'))  # ← NEW
                )                                               # ← NEW
            )                                                   # ← NEW
            eff_pct = int(n_val_images / n_total * 100)         # ← NEW
            log.info(                                           # ← NEW
                f"Train images: {n_total - n_val_images}  |  "  # ← NEW
                f"Val images: {n_val_images}  |  "             # ← NEW
                f"Total: {n_total}  |  "                       # ← NEW
                f"Effective split: {eff_pct}% val"             # ← NEW
            )                                                   # ← NEW
            log.info(f"Val set — {n_val_images} image(s):")    # ← NEW
            for stem in val_regen_stems:                        # ← NEW
                log.info(f"  [regen]   {stem}")                # ← NEW
            for stem in val_control_stems:                      # ← NEW
                log.info(f"  [control] {stem}")                # ← NEW

        return X_train, Y_train, X_val, Y_val

    # ── Flat mode + val_ tags ─────────────────────────────────────────────────
    if val_tagged_stems:                                         # ← NEW block
        patches_img  = img_train_dir  / 'cropped' / 'patches'
        patches_mask = mask_train_dir / 'cropped' / 'patches'

        # Strip val_ prefix to get the actual image stem for patch matching
        val_stems_clean = [s[4:] if s.lower().startswith('val_') else s  # ← NEW
                           for s in val_tagged_stems]           # ← NEW

        all_img_files  = list_files(str(patches_img),  extensions=('.png','.tif','.tiff'))
        all_mask_files = list_files(str(patches_mask), extensions=('.png','.tif','.tiff'))

        # Split patch files into train and val by stem           # ← NEW
        train_img_files = [f for f in all_img_files             # ← NEW
                           if not any(f.stem.startswith(s) for s in val_stems_clean)]  # ← NEW
        val_img_files   = [f for f in all_img_files             # ← NEW
                           if any(f.stem.startswith(s) for s in val_stems_clean)]  # ← NEW
        train_mask_files= [f for f in all_mask_files            # ← NEW
                           if not any(f.stem.startswith(s) for s in val_stems_clean)]  # ← NEW
        val_mask_files  = [f for f in all_mask_files            # ← NEW
                           if any(f.stem.startswith(s) for s in val_stems_clean)]  # ← NEW

        def _load_file_list(files, is_mask):                    # ← NEW
            patches = []                                        # ← NEW
            for f in sorted(files):                             # ← NEW
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)  # ← NEW
                if img is None:                                 # ← NEW
                    print(f"[WARN] Could not load patch, skipping: {f}")  # ← NEW
                    continue                                    # ← NEW
                patches.append(img)                            # ← NEW
            arr = np.expand_dims(                               # ← NEW
                np.array(patches, dtype=np.float32), axis=-1   # ← NEW
            )                                                   # ← NEW
            if is_mask:                                        # ← NEW
                return base_label(arr)                         # ← NEW
            norms = np.linalg.norm(arr, axis=1, keepdims=True) # ← NEW
            norms = np.where(norms == 0, 1, norms)             # ← NEW
            return arr / norms                                  # ← NEW

        X_train = _load_file_list(train_img_files,  is_mask=False)  # ← NEW
        Y_train = _load_file_list(train_mask_files, is_mask=True)   # ← NEW
        X_val   = _load_file_list(val_img_files,    is_mask=False)  # ← NEW
        Y_val   = _load_file_list(val_mask_files,   is_mask=True)   # ← NEW

        # Log val set                                           # ← NEW
        if log:                                                 # ← NEW
            train_stems  = _patches_to_image_stems(train_img_files)  # ← NEW
            val_stems    = _patches_to_image_stems(val_img_files)    # ← NEW
            n_total      = len(train_stems) + len(val_stems)   # ← NEW
            eff_pct      = round(len(val_stems) / n_total * 100, 1)  # ← NEW
            log.info(                                          # ← NEW
                f"Train images: {len(train_stems)}  |  "      # ← NEW
                f"Val images: {len(val_stems)}  |  "          # ← NEW
                f"Total: {n_total}  |  "                      # ← NEW
                f"Effective split: {eff_pct}% val"            # ← NEW
            )                                                  # ← NEW
            log.info(f"Val set — {len(val_stems)} image(s):")  # ← NEW
            for stem in val_stems:                             # ← NEW
                log.info(f"  {stem}")                         # ← NEW

        return X_train, Y_train, X_val, Y_val                  # ← NEW

    # ── Flat mode + no tags — random split ────────────────────────────────────
    patches_img  = img_train_dir  / 'cropped' / 'patches'
    patches_mask = mask_train_dir / 'cropped' / 'patches'

    X = load_patches(str(patches_img),  is_mask=False)
    Y = load_patches(str(patches_mask), is_mask=True)

    if len(X) != len(Y):
        raise ValueError(f"Patch count mismatch: {len(X)} images vs {len(Y)} masks")

    all_img_files = list_files(str(patches_img), extensions=('.png','.tif','.tiff'))
    indices       = list(range(len(all_img_files)))

    train_idx, val_idx = train_test_split(                      # ← CHANGED: split indices not arrays
        indices, test_size=val_fraction, random_state=42
    )

    X_train, Y_train = X[train_idx], Y[train_idx]              # ← CHANGED
    X_val,   Y_val   = X[val_idx],   Y[val_idx]                # ← CHANGED

    # Log val set                                               # ← NEW
    if log:                                                     # ← NEW
        val_files   = [all_img_files[i] for i in val_idx]      # ← NEW
        train_files = [all_img_files[i] for i in train_idx]    # ← NEW
        val_stems   = _patches_to_image_stems(val_files)        # ← NEW
        train_stems = _patches_to_image_stems(train_files)      # ← NEW
        n_total     = len(val_stems) + len(train_stems)         # ← NEW
        eff_pct     = round(len(val_stems) / n_total * 100, 1) # ← NEW
        log.info(                                               # ← NEW
            f"Train images: {len(train_stems)}  |  "           # ← NEW
            f"Val images: {len(val_stems)}  |  "               # ← NEW
            f"Total: {n_total}  |  "                           # ← NEW
            f"Effective split: {eff_pct}% val  |  "            # ← NEW
            f"Requested: {val_fraction*100:.0f}%"              # ← NEW
        )                                                       # ← NEW
        log.info(f"Val set — {len(val_stems)} image(s):")      # ← NEW
        for stem in val_stems:                                  # ← NEW
            log.info(f"  {stem}")                              # ← NEW

    return X_train, Y_train, X_val, Y_val