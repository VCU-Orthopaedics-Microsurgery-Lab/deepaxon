"""
morphometrics/morphometrics.py

Per-image morphometric analysis of segmented nerve cross-sections.
Algorithm matches V3 exactly:
  - Threshold-based watershed seeds (distance > 0.1 * max) with disk(5) background
  - Two watersheds: axon mask + fiber (axon+myelin) mask
  - Full V3 column set including equiv diameters, deformation, minor axes
  - Myelin-first matching loop with g_ratio < 1 filter
  - Duplicate assignments removed
  - Quality filters applied from config.json quality_filters block

BGW color scheme:
  - Black (0)   = background
  - Grey  (128) = myelin
  - White (255) = axon
  morphometrics.py depends on this mapping — update inRange() thresholds
  if segment.py recolor() is ever changed.
"""

from __future__ import annotations

import gc
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import label, regionprops_table, regionprops
from skimage.morphology import dilation, disk
from skimage.segmentation import watershed

from utils.logger import DeepAxonLogger
from utils.helpers import get_pixel_size, load_config


# ─── Watershed labelling ──────────────────────────────────────────────────────

def get_labels_axons(img: np.ndarray) -> np.ndarray:
    """
    Watershed segmentation using V3 threshold-based seeding.

    Foreground seeds: distance > threshold * distance.max()
      Low threshold intentional — maximizes seed count to separate
      closely packed axons. V3 validated value.
      Configurable via config.json: watershed.distance_threshold

    Background marker: dilation of mask with disk(radius)
      Configurable via config.json: watershed.dilation_disk
    """
    config    = load_config()
    wsh_cfg   = config.get("watershed", {})
    threshold = wsh_cfg.get("distance_threshold", 0.17)
    disk_r    = wsh_cfg.get("dilation_disk", 5)

    distance     = ndi.distance_transform_edt(img)
    sure_fg_mask = distance > threshold * distance.max()
    markers      = label(sure_fg_mask)
    sure_bg_mask = dilation(img, disk(disk_r))
    markers[sure_bg_mask == 0] = markers.max() + 1
    segmented = watershed(-distance, markers, mask=img)
    del distance, sure_fg_mask, sure_bg_mask, markers
    return segmented


def get_labels_fiber(
    fiber_mask: np.ndarray,
    axon_label: np.ndarray,
) -> np.ndarray:
    """
    Watershed segmentation of fiber_mask seeded from axon labels.

    Instead of deriving markers from fiber_mask's own distance transform
    (which fails when myelin sheaths touch), we transfer one seed per
    axon region from the already-labelled axon_mask.

    Foreground seeds: distance-transform peak of each axon region,
      guaranteed to land interior to the corresponding fiber region
      since axon ⊂ fiber geometrically.

    Background marker: dilation of fiber_mask with disk(5) — same
      logic as original get_labels().
    """
    config  = load_config()
    wsh_cfg = config.get("watershed", {})
    disk_r  = wsh_cfg.get("dilation_disk", 5)

    # ── 1. Extract one seed per axon at its distance-transform peak ──────────
    axon_distance = ndi.distance_transform_edt(axon_label > 0)
    markers       = np.zeros_like(fiber_mask, dtype=int)

    for region in regionprops(axon_label):
        region_dist = np.where(axon_label == region.label, axon_distance, 0)
        peak        = np.unravel_index(np.argmax(region_dist), region_dist.shape)
        markers[peak] = region.label

    # ── 2. Snap any seeds that fall outside fiber_mask to nearest fg pixel ───
    lost = (markers > 0) & (fiber_mask == 0)
    if np.any(lost):
        fg_coords   = np.array(np.where(fiber_mask > 0)).T
        lost_coords = np.array(np.where(lost)).T
        tree        = cKDTree(fg_coords)
        for coord in lost_coords:
            _, idx  = tree.query(coord)
            nearest = tuple(fg_coords[idx])
            markers[nearest]      = markers[tuple(coord)]
            markers[tuple(coord)] = 0

    # ── 3. Background marker ──────────────────────────────────────────────────
    sure_bg               = dilation(fiber_mask, disk(disk_r))
    markers[sure_bg == 0] = markers.max() + 1

    # ── 4. Watershed on fiber_mask ────────────────────────────────────────────
    fiber_distance = ndi.distance_transform_edt(fiber_mask)
    segmented      = watershed(-fiber_distance, markers, mask=fiber_mask)

    del axon_distance, fiber_distance, sure_bg, markers
    return segmented


# ─── Matching helpers ─────────────────────────────────────────────────────────

def get_axon_row(axon_df: pd.DataFrame, left, right, top, bottom) -> pd.DataFrame:
    """Find the largest axon whose centroid falls within the fiber bounding box."""
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
        seg_path: path to segmented .tif (BGW: black=background, grey=myelin, white=axon)
        mag:      magnification string e.g. '40X'
        log:      logger instance

    Returns:
        DataFrame of per-axon measurements, or None on failure.

    Quality filters applied from config.json quality_filters block:
        min/max axon area, solidity, eccentricity, myelin thickness,
        g-ratio range, min/max fiber area.
        All filters require physical units — skipped if px_size uncalibrated.
        G-ratio range filter always applied (dimensionless).
    """
    img = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        log.error(f"Could not read: {seg_path}")
        return None

    h, w    = img.shape
    px_size = get_pixel_size(mag, w)
    config  = load_config()

    if px_size is None:
        log.warn(f"No pixel size for {mag} at width {w}px — outputting pixel units only")

    # ── Quality filter thresholds ─────────────────────────────────────────────
    # All configurable via config.json quality_filters block.
    # Physical unit filters only applied when px_size is available.
    # G-ratio filters always applied (dimensionless).
    qf = config.get("quality_filters", {})

    min_axon_area_um2      = qf.get("min_axon_area_um2",      1.0)
    max_axon_area_um2      = qf.get("max_axon_area_um2",      500.0)
    min_axon_solidity      = qf.get("min_axon_solidity",      0.7)
    max_axon_eccentricity  = qf.get("max_axon_eccentricity",  0.95)
    min_myelin_thick_um    = qf.get("min_myelin_thickness_um", 0.3)
    gratio_min             = qf.get("gratio_min",             0.2)
    gratio_max             = qf.get("gratio_max",             0.9)
    min_fiber_area_um2     = qf.get("min_fiber_area_um2",     2.0)
    max_fiber_area_um2     = qf.get("max_fiber_area_um2",     800.0)

    # ── Extract masks from BGW image ─────────────────────────────────────────
    # BGW contract: white(255)=axon, grey(128)=myelin
    # See segment/segment.py recolor() — this mapping must stay in sync.
    axon_mask  = cv2.inRange(img, 200, 255)  # white = axon
    fiber_mask = cv2.inRange(img, 1, 255)    # all non-black = axon + myelin

    if not np.any(axon_mask):
        log.warn(f"No axons detected in {Path(seg_path).name}")
        return pd.DataFrame()

    # Watershed both masks
    axon_label  = get_labels_axons(axon_mask.astype(bool))
    fiber_label = get_labels_fiber(fiber_mask.astype(bool), axon_label)

    axon_props = regionprops_table(axon_label, properties=(
        'label', 'centroid', 'area',
        'axis_minor_length', 'axis_major_length',
        'eccentricity', 'orientation', 'perimeter', 'solidity'
    ))
    fiber_props = regionprops_table(fiber_label, properties=(
        'label', 'bbox', 'area',
        'axis_minor_length', 'axis_major_length',
        'eccentricity', 'orientation', 'perimeter'
    ))

    axon_df  = pd.DataFrame.from_dict(axon_props)
    fiber_df = pd.DataFrame.from_dict(fiber_props)

    del axon_label, fiber_label, axon_mask, fiber_mask, img
    gc.collect()

    if axon_df.empty or fiber_df.empty:
        log.warn(f"No regions found in {Path(seg_path).name}")
        return pd.DataFrame()

    # ── Myelin-first matching loop ────────────────────────────────────────────
    rows                 = []
    assigned_axon_labels = set()
    n_filtered           = 0  # track how many pairs were filtered

    for _, row in fiber_df.iterrows():
        left   = row['bbox-0']
        right  = row['bbox-2']
        top    = row['bbox-1']
        bottom = row['bbox-3']

        axon_row = get_axon_row(axon_df, left, right, top, bottom)
        if axon_row.empty:
            continue

        axon_label_id = int(axon_row['label'].iloc[0])
        if axon_label_id in assigned_axon_labels:
            continue
        assigned_axon_labels.add(axon_label_id)

        # ── Axon morphology ───────────────────────────────────────────────────
        axon_area         = float(axon_row['area'].iloc[0])
        axon_perimeter    = float(axon_row['perimeter'].iloc[0])
        axon_major        = float(axon_row['axis_major_length'].iloc[0])
        axon_minor        = float(axon_row['axis_minor_length'].iloc[0])
        axon_eccentricity = float(axon_row['eccentricity'].iloc[0])
        axon_orientation  = float(axon_row['orientation'].iloc[0])
        axon_solidity     = float(axon_row['solidity'].iloc[0])
        x                 = float(axon_row['centroid-0'].iloc[0])
        y                 = float(axon_row['centroid-1'].iloc[0])

        axon_equiv_diam  = np.sqrt(4 * axon_area / np.pi)
        axon_deformation = axon_major / axon_equiv_diam if axon_equiv_diam > 0 else np.nan

        # ── Fiber / Myelin morphology ─────────────────────────────────────────
        fiber_area         = float(row['area'])
        fiber_major        = float(row['axis_major_length'])
        fiber_minor        = float(row['axis_minor_length'])
        fiber_perimeter    = float(row['perimeter'])
        fiber_eccentricity = float(row['eccentricity'])
        fiber_orientation  = float(row['orientation'])

        fiber_equiv_diam  = np.sqrt(fiber_area / np.pi) * 2
        fiber_deformation = fiber_major / fiber_minor if fiber_minor > 0 else np.nan
        myelin_area       = fiber_area - axon_area
        myelin_thickness  = (fiber_equiv_diam - axon_major) / 2

        # ── G-ratios ──────────────────────────────────────────────────────────
        gratio_area = axon_equiv_diam / fiber_equiv_diam if fiber_equiv_diam > 0 else np.nan
        axon_mean_axis  = (axon_major + axon_minor) / 2
        fiber_mean_axis = (fiber_major + fiber_minor) / 2
        gratio_axes     = axon_mean_axis / fiber_mean_axis if fiber_mean_axis > 0 else np.nan

        if np.isnan(gratio_area) or np.isnan(gratio_axes):
            continue
        gratio = (gratio_area + gratio_axes) / 2

        # ── Quality filters ───────────────────────────────────────────────────
        # Applied before appending — all configurable via config.json.
        # Biologically justified: excludes measurements inconsistent with
        # myelinated axon morphology. Applied consistently to all images.

        # G-ratio range — always applied (dimensionless, no px_size needed)
        if not (0 < gratio_area < 1 and 0 < gratio_axes < 1 and 0 < gratio < 1):
            n_filtered += 1
            continue
        if not (gratio_min < gratio < gratio_max):
            n_filtered += 1
            continue

        # Shape filters — always applied (dimensionless)
        if axon_solidity < min_axon_solidity:
            n_filtered += 1
            continue
        if axon_eccentricity > max_axon_eccentricity:
            n_filtered += 1
            continue

        # Physical unit filters — only applied when px_size is calibrated
        if px_size is not None:
            axon_area_um2   = axon_area   * px_size ** 2
            fiber_area_um2  = fiber_area  * px_size ** 2
            myelin_thick_um = myelin_thickness * px_size

            if not (min_axon_area_um2 <= axon_area_um2 <= max_axon_area_um2):
                n_filtered += 1
                continue
            if not (min_fiber_area_um2 <= fiber_area_um2 <= max_fiber_area_um2):
                n_filtered += 1
                continue
            if myelin_thick_um < min_myelin_thick_um:
                n_filtered += 1
                continue

        # ── Build output row ──────────────────────────────────────────────────
        new_row = {
            # Identification
            'label': int(row['label']),
            'x':     round(x, 1),
            'y':     round(y, 1),

            # Axon morphometrics (pixel units)
            'axon_area':         round(axon_area, 2),
            'axon_perimeter':    round(axon_perimeter, 3),
            'axon_diam':         round(axon_equiv_diam, 3),
            'axon_major':        round(axon_major, 3),
            'axon_minor':        round(axon_minor, 3),
            'axon_solidity':     round(axon_solidity, 4),
            'axon_deformation':  round(axon_deformation, 4) if not np.isnan(axon_deformation) else np.nan,
            'axon_eccentricity': round(axon_eccentricity, 4),
            'axon_orientation':  round(axon_orientation, 4),

            # Myelin / Fiber morphometrics (pixel units)
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

            # G-ratios
            'gratio_area': round(gratio_area, 4),
            'gratio_axes': round(gratio_axes, 4),
            'gratio':      round(gratio, 4),
        }

        # Physical units — only when px_size calibrated
        if px_size is not None:
            new_row['axon_area_um2']       = round(axon_area        * px_size ** 2, 3)
            new_row['myelin_area_um2']     = round(myelin_area      * px_size ** 2, 3)
            new_row['fiber_area_um2']      = round(fiber_area       * px_size ** 2, 3)
            new_row['axon_diam_um']        = round(axon_equiv_diam  * px_size, 3)
            new_row['axon_major_um']       = round(axon_major       * px_size, 3)
            new_row['axon_minor_um']       = round(axon_minor       * px_size, 3)
            new_row['fiber_equiv_diam_um'] = round(fiber_equiv_diam * px_size, 3)
            new_row['fiber_major_um']      = round(fiber_major      * px_size, 3)
            new_row['fiber_minor_um']      = round(fiber_minor      * px_size, 3)
            new_row['myelin_thickness_um'] = round(myelin_thickness * px_size, 3)

        rows.append(new_row)

    del axon_df, fiber_df
    gc.collect()

    if not rows:
        log.warn(f"No valid axon-fiber pairs in {Path(seg_path).name}")
        return pd.DataFrame()

    if n_filtered > 0:
        log.info(f"  Quality filters removed {n_filtered} axon-fiber pair(s)")

    df = pd.DataFrame(rows)
    df.insert(0, 'image', Path(seg_path).stem)
    df.insert(1, 'resolution', f"{w}x{h}")
    df.insert(2, 'magnification', mag)

    return df


# ─── Save output ──────────────────────────────────────────────────────────────

def save_morphometrics(df: pd.DataFrame, output_dir: str, stem: str) -> str:
    """Save morphometrics DataFrame to Excel. Returns output path."""
    config = load_config()
    suffix = config.get("morphometrics_suffix", "_morphometrics")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}{suffix}.xlsx"
    df.to_excel(str(out_path), index=False)
    return str(out_path)