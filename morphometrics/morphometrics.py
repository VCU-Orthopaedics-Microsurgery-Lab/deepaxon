"""
morphometrics/morphometrics.py

Per-image morphometric analysis of segmented nerve cross-sections.
Algorithm matches V3 exactly:
  - Threshold-based watershed seeds (distance > 0.1 * max) with disk(5) background
  - Two watersheds: axon mask + fibre (axon+myelin) mask
  - Full V3 column set including equiv diameters, deformation, minor axes
  - Myelin-first matching loop with g_ratio < 1 filter
  - Duplicate assignments removed

BGW colour scheme: black=background, grey=axon, white=fibre (axon+myelin)
"""

from __future__ import annotations

import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops_table
from skimage.morphology import dilation, disk
from skimage.segmentation import watershed

from utils.console import DeepAxonLogger
from utils.helpers import get_pixel_size


# ─── Watershed labelling ──────────────────────────────────────────────────────

def get_labels(img: np.ndarray) -> np.ndarray:
    """
    Watershed segmentation using V3 threshold-based seeding.
    Foreground: distance > 0.1 * distance.max()
    Background: dilation of mask with disk(5)
    """
    distance = ndi.distance_transform_edt(img)
    sure_fg_mask = distance > 0.1 * distance.max()
    markers = label(sure_fg_mask)
    sure_bg_mask = dilation(img, disk(5))
    markers[sure_bg_mask == 0] = markers.max() + 1
    segmented = watershed(-distance, markers, mask=img)
    del distance, sure_fg_mask, sure_bg_mask, markers
    return segmented


# ─── Matching helpers ─────────────────────────────────────────────────────────

def get_axon_row(axon_df: pd.DataFrame, left, right, top, bottom) -> pd.DataFrame:
    """Find the largest axon whose centroid falls within the fibre bounding box."""
    left, right, top, bottom = int(left), int(right), int(top), int(bottom)
    axons_in_box = axon_df[
        (axon_df['centroid-0'] >= left) & (axon_df['centroid-0'] <= right) &
        (axon_df['centroid-1'] >= top)  & (axon_df['centroid-1'] <= bottom)
    ]
    if axons_in_box.empty:
        return axons_in_box
    return axons_in_box[axons_in_box['area'] == axons_in_box['area'].max()]


# ─── Main analysis ────────────────────────────────────────────────────────────

def get_morphometrics(
    seg_path: str,
    mag: str,
    log: DeepAxonLogger
) -> pd.DataFrame | None:
    """
    Extract morphometric measurements from a single segmented BGW image.

    Args:
        seg_path: path to segmented .tif (BGW: black=bg, grey=axon, white=fibre)
        mag:      magnification string e.g. '40X'
        log:      logger instance

    Returns:
        DataFrame of per-axon measurements, or None on failure.
    """
    img = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        log.error(f"Could not read: {seg_path}")
        return None

    h, w = img.shape
    px_size = get_pixel_size(mag, w)
    if px_size is None:
        log.warn(f"No pixel size for {mag} at width {w}px — outputting pixel units only")

    # Extract masks from BGW image (matches V3 inRange values)
    axon_mask  = cv2.inRange(img, 200, 255)  # white = axon
    fibre_mask = cv2.inRange(img, 1, 255)    # everything non-black = axon + myelin

    if not np.any(axon_mask):
        log.warn(f"No axons detected in {Path(seg_path).name}")
        return pd.DataFrame()

    # Watershed both masks using V3 get_labels
    axon_label  = get_labels(axon_mask.astype(bool))
    fibre_label = get_labels(fibre_mask.astype(bool))

    # regionprops_table on full label arrays — memory efficient
    axon_props = regionprops_table(axon_label, properties=(
        'label', 'centroid', 'area',
        'axis_minor_length', 'axis_major_length',
        'eccentricity', 'orientation', 'perimeter', 'solidity'
    ))
    fibre_props = regionprops_table(fibre_label, properties=(
        'label', 'bbox', 'area',
        'axis_minor_length', 'axis_major_length',
        'eccentricity', 'orientation', 'perimeter'
    ))

    axon_df  = pd.DataFrame.from_dict(axon_props)
    fibre_df = pd.DataFrame.from_dict(fibre_props)

    del axon_label, fibre_label, axon_mask, fibre_mask, img
    gc.collect()

    if axon_df.empty or fibre_df.empty:
        log.warn(f"No regions found in {Path(seg_path).name}")
        return pd.DataFrame()

    # ── Myelin-first matching loop (V3 active loop) ───────────────────────────
    rows = []

    for _, row in fibre_df.iterrows():
        left   = row['bbox-0']
        right  = row['bbox-2']
        top    = row['bbox-1']
        bottom = row['bbox-3']

        axon_row = get_axon_row(axon_df, left, right, top, bottom)
        if axon_row.empty:
            continue

        # ── Axon morphology ───────────────────────────────────────────────
        axon_area        = float(axon_row['area'].iloc[0])
        axon_perimeter   = float(axon_row['perimeter'].iloc[0])
        axon_major       = float(axon_row['axis_major_length'].iloc[0])
        axon_minor       = float(axon_row['axis_minor_length'].iloc[0])
        axon_eccentricity = float(axon_row['eccentricity'].iloc[0])
        axon_orientation = float(axon_row['orientation'].iloc[0])
        axon_solidity    = float(axon_row['solidity'].iloc[0])
        x                = float(axon_row['centroid-0'].iloc[0])
        y                = float(axon_row['centroid-1'].iloc[0])

        # Area-based equivalent circle diameter (V3)
        axon_equiv_diam   = np.sqrt(4 * axon_area / np.pi)
        axon_deformation  = axon_major / axon_equiv_diam if axon_equiv_diam > 0 else np.nan

        # ── Fibre / Myelin morphology ─────────────────────────────────────
        fiber_area        = float(row['area'])
        fiber_major       = float(row['axis_major_length'])
        fiber_minor       = float(row['axis_minor_length'])
        fiber_perimeter   = float(row['perimeter'])
        fiber_eccentricity = float(row['eccentricity'])
        fiber_orientation = float(row['orientation'])

        fiber_equiv_diam  = np.sqrt(fiber_area / np.pi) * 2
        fiber_deformation = fiber_major / fiber_minor if fiber_minor > 0 else np.nan

        myelin_area       = fiber_area - axon_area
        myelin_thickness  = (fiber_equiv_diam - axon_major) / 2  # V3 formula

        # ── Derived: three g-ratio measures ──────────────────────────────
        # Option 2: equivalent circle diameters (area-based)
        gratio_area = axon_equiv_diam / fiber_equiv_diam if fiber_equiv_diam > 0 else np.nan

        # Option 4: mean of major+minor axes
        axon_mean_axis  = (axon_major + axon_minor) / 2
        fiber_mean_axis = (fiber_major + fiber_minor) / 2
        gratio_axes = axon_mean_axis / fiber_mean_axis if fiber_mean_axis > 0 else np.nan

        # Average of both
        if np.isnan(gratio_area) or np.isnan(gratio_axes):
            continue
        gratio = (gratio_area + gratio_axes) / 2

        # Filter: discard if ANY gratio is outside (0, 1)
        if not (0 < gratio_area < 1 and 0 < gratio_axes < 1 and 0 < gratio < 1):
            continue

        new_row = {
            # Identification
            'label':              int(row['label']),
            'x':                  round(x, 1),
            'y':                  round(y, 1),

            # Axon morphometrics
            'axon_area':          round(axon_area, 2),
            'axon_perimeter':     round(axon_perimeter, 3),
            'axon_diam':          round(axon_equiv_diam, 3),
            'axon_major':         round(axon_major, 3),
            'axon_minor':         round(axon_minor, 3),
            'axon_solidity':      round(axon_solidity, 4),
            'axon_deformation':   round(axon_deformation, 4) if not np.isnan(axon_deformation) else np.nan,
            'axon_eccentricity':  round(axon_eccentricity, 4),
            'axon_orientation':   round(axon_orientation, 4),

            # Myelin / Fibre morphometrics
            'myelin_area':        round(myelin_area, 2),
            'myelin_thickness':   round(myelin_thickness, 3),
            'myelin_perimeter':   round(fiber_perimeter, 3),
            'fiber_area':         round(fiber_area, 2),
            'fiber_equiv_diam':   round(fiber_equiv_diam, 3),
            'fiber_major':        round(fiber_major, 3),
            'fiber_minor':        round(fiber_minor, 3),
            'fiber_deformation':  round(fiber_deformation, 4) if not np.isnan(fiber_deformation) else np.nan,
            'fiber_eccentricity': round(fiber_eccentricity, 4),
            'fiber_orientation':  round(fiber_orientation, 4),

            # Derived g-ratios
            'gratio_area':        round(gratio_area, 4),   # equiv circle diam / equiv circle diam
            'gratio_axes':        round(gratio_axes, 4),   # mean(axon axes) / mean(fiber axes)
            'gratio':             round(gratio, 4),        # average of gratio_area and gratio_axes
        }

        # Physical units if pixel size available
        if px_size is not None:
            new_row['axon_area_um2']       = round(axon_area   * px_size ** 2, 3)
            new_row['myelin_area_um2']     = round(myelin_area * px_size ** 2, 3)
            new_row['fiber_area_um2']      = round(fiber_area  * px_size ** 2, 3)
            new_row['axon_diam_um']        = round(axon_equiv_diam  * px_size, 3)
            new_row['axon_major_um']       = round(axon_major       * px_size, 3)
            new_row['axon_minor_um']       = round(axon_minor       * px_size, 3)
            new_row['fiber_equiv_diam_um'] = round(fiber_equiv_diam * px_size, 3)
            new_row['fiber_major_um']      = round(fiber_major      * px_size, 3)
            new_row['fiber_minor_um']      = round(fiber_minor      * px_size, 3)
            new_row['myelin_thickness_um'] = round(myelin_thickness * px_size, 3)

        rows.append(new_row)

    del axon_df, fibre_df
    gc.collect()

    if not rows:
        log.warn(f"No valid axon-fibre pairs in {Path(seg_path).name}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.insert(0, 'image', Path(seg_path).stem)
    df.insert(1, 'resolution', f"{w}x{h}")
    df.insert(2, 'magnification', mag)

    return df


# ─── Save output ──────────────────────────────────────────────────────────────

def save_morphometrics(df: pd.DataFrame, output_dir: str, stem: str) -> str:
    """Save morphometrics DataFrame to Excel. Returns output path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}_morphometrics.xlsx"
    df.to_excel(str(out_path), index=False)
    return str(out_path)
