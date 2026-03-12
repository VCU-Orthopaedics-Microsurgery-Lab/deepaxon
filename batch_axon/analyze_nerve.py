"""
batch_axon/analyze_nerve.py

Compiles per-image morphometrics into nerve-level and image-level summaries.
Handles CSA extraction from FIJI overlays (40X) or full image area (100X).
Reads conversion factors from config.json — no hardcoded values.
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
from pathlib import Path

from utils.helpers import get_pixel_size, load_config
from utils.console import DeepAxonLogger
from batch_axon.overlay.process_overlay import get_overlay_area


def make_aggregate_df(morph_dir: Path, log: DeepAxonLogger) -> pd.DataFrame:
    """Concatenate all per-image morphometrics .xlsx files into one DataFrame."""
    dfs = []
    for f in sorted(morph_dir.glob("*.xlsx")):
        try:
            df = pd.read_excel(str(f))
            dfs.append(df)
        except Exception as e:
            log.warn(f"Could not read {f.name}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def get_image_csa(
    img_name: str,
    csa_dir: Path,
    mag: str,
    img_width: int,
    log: DeepAxonLogger
) -> float | None:
    """
    Get the cross-sectional area (in µm²) for a single image.

    For 40X: reads from CSA overlay file, converts px² → µm².
    For 100X: uses full image area (width × height) × px_size².
    For unknown mag: returns None.
    """
    config = load_config()
    csa_suffix = config.get("csa_suffix", "_CSA.tif")
    px_size = get_pixel_size(mag, img_width)

    if mag == "40X":
        csa_file = csa_dir / f"{img_name}{csa_suffix}"
        if not csa_file.exists():
            log.warn(f"CSA file not found: {csa_file.name}")
            return None

        px_count = get_overlay_area(str(csa_file))
        if px_count is None:
            log.warn(f"Could not extract overlay area for {img_name}")
            return None

        if px_size is not None:
            return px_count * (px_size ** 2)
        else:
            log.warn(f"No pixel size for {mag} — returning pixel CSA")
            return float(px_count)

    elif mag == "100X":
        # Full image area
        # img_width already known; need height — assume standard 1024 after resize
        img_height = 1024
        px_count = img_width * img_height

        if px_size is not None:
            return px_count * (px_size ** 2)
        else:
            log.warn(f"100X pixel size not configured — returning pixel CSA")
            return float(px_count)

    else:
        log.warn(f"Unknown magnification '{mag}' — cannot compute CSA")
        return None


def get_nerve_fourx_csa(
    nerve_name: str,
    csa_dir: Path,
    log: DeepAxonLogger
) -> float | None:
    """
    Get the whole-nerve CSA from the 4X overlay.
    File expected: {nerve_name}_4X_CSA.tif
    """
    config = load_config()
    csa_suffix = config.get("csa_suffix", "_CSA.tif")
    px_size = get_pixel_size("4X", 2880)  # 4X images are always full res

    # Find all 4X CSA files — handles single or multiple (large nerve spanning 2 images)
    candidates = list(dict.fromkeys(
        list(csa_dir.glob(f"{nerve_name}_4X{csa_suffix}")) +
        list(csa_dir.glob(f"{nerve_name}_4X_*{csa_suffix}")) +
        list(csa_dir.glob(f"{nerve_name}_4x{csa_suffix}")) +
        list(csa_dir.glob(f"{nerve_name}_4x_*{csa_suffix}"))
    ))

    if not candidates:
        log.warn(f"No 4X CSA file(s) found for nerve: {nerve_name}")
        return None

    if len(candidates) > 1:
        log.info(f"Found {len(candidates)} 4X CSA files for {nerve_name} — summing areas")

    total_px = 0
    for csa_file in sorted(candidates):
        px_count = get_overlay_area(str(csa_file))
        if px_count is None:
            log.warn(f"Could not extract area from {csa_file.name} — skipping")
            continue
        log.info(f"  {csa_file.name}: {px_count:,} px")
        total_px += px_count

    if total_px == 0:
        return None

    if px_size is not None:
        return total_px * (px_size ** 2)
    return float(total_px)


def get_nerve_data(
    nerve_dir: Path,
    mag: str,
    log: DeepAxonLogger
) -> tuple[list[dict], dict]:
    """
    Compile per-image and aggregate data for a single nerve.

    Returns:
        (image_rows, aggregate_dict)
    """
    config = load_config()
    morph_dir = nerve_dir / config.get("morphometrics_folder", "Morphometrics")
    csa_dir = nerve_dir / config.get("csa_folder", "CSA")

    if not morph_dir.exists():
        log.warn(f"No morphometrics folder found: {morph_dir}")
        return [], {}

    # Aggregate all morphometrics
    agg_df = make_aggregate_df(morph_dir, log)
    if agg_df.empty:
        log.warn(f"No morphometrics data found in {morph_dir}")
        return [], {}

    # 4X whole-nerve CSA
    fourx_csa = None
    if csa_dir.exists():
        fourx_csa = get_nerve_fourx_csa(nerve_dir.name, csa_dir, log)

    # Per-image rows
    image_rows = []
    img_count = 0

    for morph_file in sorted(morph_dir.glob("*.xlsx")):
        img_name = morph_file.stem.replace('_morphometrics', '')

        try:
            img_df = pd.read_excel(str(morph_file))
        except Exception as e:
            log.warn(f"Could not read {morph_file.name}: {e}")
            continue

        # Get image width from the data if available
        img_width = 1440  # Default post-resize width
        if 'resolution' in img_df.columns and len(img_df) > 0:
            try:
                res_str = str(img_df['resolution'].iloc[0])
                img_width = int(res_str.split('x')[0])
            except Exception:
                pass

        # Get CSA for this image
        img_csa = None
        if csa_dir.exists():
            img_csa = get_image_csa(img_name, csa_dir, mag, img_width, log)

        px_size = get_pixel_size(mag, img_width)

        row = {
            'name': img_name,
            'resolution': f"{img_width}px",
            'csa_um2': img_csa,
            'total_axons': len(img_df),
            'gratio': img_df['gratio'].mean() if 'gratio' in img_df.columns else None,
            'axon_diam_um': img_df['axon_diam_um'].mean() if 'axon_diam_um' in img_df.columns else (
                img_df['axon_diam_px'].mean() * px_size if px_size and 'axon_diam_px' in img_df.columns else None
            ),
        }

        # Axon density
        if img_csa and img_csa > 0:
            row['axon_density_per_um2'] = len(img_df) / img_csa
        else:
            row['axon_density_per_um2'] = None

        image_rows.append(row)
        img_count += 1

    # Aggregate summary
    aggregate = {
        'fourx_csa_um2': fourx_csa,
        'total_images': img_count,
        'total_axons': len(agg_df),
        'mean_gratio': agg_df['gratio'].mean() if 'gratio' in agg_df.columns else None,
    }

    # Axon diameter (prefer um if available)
    if 'axon_diam_um' in agg_df.columns:
        aggregate['mean_axon_diam_um'] = agg_df['axon_diam_um'].mean()
    elif 'axon_diam_px' in agg_df.columns:
        px_size = get_pixel_size(mag, 1440)
        aggregate['mean_axon_diam_um'] = agg_df['axon_diam_px'].mean() * px_size if px_size else None

    # Total sample CSA (sum of per-image CSAs)
    valid_csas = [r['csa_um2'] for r in image_rows if r.get('csa_um2')]
    aggregate['total_sample_csa_um2'] = sum(valid_csas) if valid_csas else None

    # Estimated full nerve axon count (extrapolated using 4X CSA)
    if fourx_csa and aggregate.get('total_sample_csa_um2') and aggregate['total_sample_csa_um2'] > 0:
        aggregate['estimated_total_axons'] = int(
            aggregate['total_axons'] / aggregate['total_sample_csa_um2'] * fourx_csa
        )
    else:
        aggregate['estimated_total_axons'] = None

    return image_rows, aggregate
