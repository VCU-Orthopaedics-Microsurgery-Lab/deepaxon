"""
train/dataset/split.py

Stratified train/val split for analysis runs (v5_analysis branch).
Replaces val_ prefix detection for all Wave 1/2 jobs.

Guarantees:
    - Equal ctrl/regen balance in both train and val sets
    - Reproducible via fixed seed
    - Dataset size subsampling (10, 20, 30 images) draws equally from each phenotype
    - Val fraction maps to brief ratios: 70/30 → 0.30, 80/20 → 0.20, 90/10 → 0.10

On v5_analysis, val_ prefix detection has been removed from data_loader.py.
This module is the sole split mechanism for all analysis runs.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional
import math


# ─── Phenotype detection ──────────────────────────────────────────────────────

def get_phenotype(stem: str, ctrl_prefix: str, regen_prefix: str) -> str:
    """
    Return 'ctrl' or 'regen' from a filename stem.
    Raises ValueError if stem matches neither prefix.
    """
    if stem.startswith(ctrl_prefix):
        return 'ctrl'
    if stem.startswith(regen_prefix):
        return 'regen'
    raise ValueError(
        f"Cannot determine phenotype for '{stem}' — "
        f"expected prefix '{ctrl_prefix}' or '{regen_prefix}'"
    )


def build_image_manifest(
    images_dir: str,
    ctrl_prefix: str  = 'ctrl_',
    regen_prefix: str = 'regen_',
    extensions: tuple = ('.tif', '.tiff', '.png'),
) -> list[dict]:
    """
    Scan images_dir for source images and return a manifest list.

    Each entry: {'stem': str, 'path': Path, 'phenotype': 'ctrl'|'regen'}

    Raises:
        FileNotFoundError  — images_dir does not exist
        ValueError         — any image stem cannot be phenotyped
        ValueError         — phenotype counts are not equal (1:1 ratio required)
    """
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")

    files = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in extensions
    ])
    if not files:
        raise FileNotFoundError(f"No image files found in {images_dir}")

    manifest = []
    for p in files:
        phenotype = get_phenotype(p.stem, ctrl_prefix, regen_prefix)
        manifest.append({'stem': p.stem, 'path': p, 'phenotype': phenotype})

    ctrl_n  = sum(1 for m in manifest if m['phenotype'] == 'ctrl')
    regen_n = sum(1 for m in manifest if m['phenotype'] == 'regen')

    if ctrl_n != regen_n:
        raise ValueError(
            f"Phenotype imbalance: {ctrl_n} ctrl vs {regen_n} regen — "
            f"1:1 ratio required"
        )

    return manifest


# ─── Stratified split ─────────────────────────────────────────────────────────

def stratified_split(
    manifest:     list[dict],
    n_images:     int,
    val_fraction: float,
    seed:         int,
    ctrl_prefix:  str = 'ctrl_',
    regen_prefix: str = 'regen_',
) -> tuple[list[dict], list[dict]]:
    """
    Draw n_images from manifest (equal ctrl/regen) then split into
    train and val sets, maintaining phenotype balance in both.

    Args:
        manifest:     output of build_image_manifest()
        n_images:     total images to use — must be even, <= len(manifest)
        val_fraction: fraction assigned to val (e.g. 0.30 for 70/30)
        seed:         random seed for reproducibility (1–5 per brief)
        ctrl_prefix:  phenotype prefix (passed through for validation)
        regen_prefix: phenotype prefix (passed through for validation)

    Returns:
        (train_manifest, val_manifest) — each a list of manifest dicts

    Raises:
        ValueError — n_images odd, exceeds dataset, or val set < 1 per phenotype
    """
    if n_images % 2 != 0:
        raise ValueError(f"n_images must be even (equal ctrl/regen), got {n_images}")

    ctrl_images  = [m for m in manifest if m['phenotype'] == 'ctrl']
    regen_images = [m for m in manifest if m['phenotype'] == 'regen']
    n_per_class  = n_images // 2

    if n_per_class > len(ctrl_images):
        raise ValueError(
            f"Requested {n_per_class} ctrl images but only {len(ctrl_images)} available"
        )
    if n_per_class > len(regen_images):
        raise ValueError(
            f"Requested {n_per_class} regen images but only {len(regen_images)} available"
        )

    rng = np.random.default_rng(seed)

    # Subsample equally from each phenotype
    ctrl_sample  = rng.choice(len(ctrl_images),  size=n_per_class,  replace=False).tolist()
    regen_sample = rng.choice(len(regen_images), size=n_per_class, replace=False).tolist()
    ctrl_sel     = [ctrl_images[i]  for i in sorted(ctrl_sample)]
    regen_sel    = [regen_images[i] for i in sorted(regen_sample)]

    # Split each phenotype into train/val
    n_val_per_class = max(1, math.floor(n_per_class * val_fraction + 0.5))
    if n_val_per_class < 1:
        raise ValueError(
            f"val set too small: {n_val_per_class} per phenotype at "
            f"n_images={n_images}, val_fraction={val_fraction}"
        )

    # Shuffle within each phenotype before splitting (seed advances naturally)
    ctrl_order  = rng.permutation(len(ctrl_sel)).tolist()
    regen_order = rng.permutation(len(regen_sel)).tolist()

    ctrl_val   = [ctrl_sel[i]  for i in ctrl_order[:n_val_per_class]]
    ctrl_train = [ctrl_sel[i]  for i in ctrl_order[n_val_per_class:]]
    regen_val  = [regen_sel[i] for i in regen_order[:n_val_per_class]]
    regen_train= [regen_sel[i] for i in regen_order[n_val_per_class:]]

    train_manifest = ctrl_train + regen_train
    val_manifest   = ctrl_val   + regen_val

    return train_manifest, val_manifest


# ─── Patch-level index builder ────────────────────────────────────────────────

def get_patch_indices_for_split(
    all_patch_files: list[Path],
    split_manifest:  list[dict],
) -> list[int]:
    """
    Given a list of all patch file paths and a manifest subset (train or val),
    return the indices into all_patch_files whose source image stem is in
    the manifest subset.

    Patch filename convention:  {source_stem}_{RRCC}.png
    Source stem recovered by dropping the last _XXXX segment.

    Args:
        all_patch_files: sorted list of all patch Paths
        split_manifest:  train or val manifest from stratified_split()

    Returns:
        Sorted list of integer indices into all_patch_files
    """
    target_stems = {m['stem'] for m in split_manifest}

    indices = []
    for i, p in enumerate(all_patch_files):
        source_stem = '_'.join(p.stem.split('_')[:-1])
        if source_stem in target_stems:
            indices.append(i)

    return sorted(indices)


# ─── Summary helper ───────────────────────────────────────────────────────────

def split_summary(
    train_manifest: list[dict],
    val_manifest:   list[dict],
    n_train_patches: Optional[int] = None,
    n_val_patches:   Optional[int] = None,
) -> dict:
    """
    Return a loggable summary dict for a split.
    """
    train_ctrl  = sum(1 for m in train_manifest if m['phenotype'] == 'ctrl')
    train_regen = sum(1 for m in train_manifest if m['phenotype'] == 'regen')
    val_ctrl    = sum(1 for m in val_manifest   if m['phenotype'] == 'ctrl')
    val_regen   = sum(1 for m in val_manifest   if m['phenotype'] == 'regen')

    n_total = len(train_manifest) + len(val_manifest)
    eff_val_pct = round(len(val_manifest) / n_total * 100, 1)

    summary = {
        'train_images':       len(train_manifest),
        'train_ctrl':         train_ctrl,
        'train_regen':        train_regen,
        'val_images':         len(val_manifest),
        'val_ctrl':           val_ctrl,
        'val_regen':          val_regen,
        'effective_val_pct':  eff_val_pct,
        'train_stems':        [m['stem'] for m in train_manifest],
        'val_stems':          [m['stem'] for m in val_manifest],
    }
    if n_train_patches is not None:
        summary['train_patches'] = n_train_patches
    if n_val_patches is not None:
        summary['val_patches'] = n_val_patches

    return summary
