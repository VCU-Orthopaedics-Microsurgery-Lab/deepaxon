"""
morphometrics/morphometrics.py

Per-image morphometric analysis of segmented nerve cross-sections.
Algorithm matches V3 exactly:
  - Threshold-based watershed seeds (distance > 0.17 * max) with disk(5) background
  - Two watersheds: axon mask + fiber (axon+myelin) mask
  - Full V3 column set including equiv diameters, deformation, minor axes
  - Myelin-first matching loop with g_ratio < 1 filter
  - Duplicate assignments removed
  - Quality filters applied from config.json quality_filters block
    (toggle via quality_filters.enabled in config.json)

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
import matplotlib.colors as mcolors

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

    axon_distance = ndi.distance_transform_edt(axon_label > 0)
    markers       = np.zeros_like(fiber_mask, dtype=int)

    for region in regionprops(axon_label):
        region_dist = np.where(axon_label == region.label, axon_distance, 0)
        peak        = np.unravel_index(np.argmax(region_dist), region_dist.shape)
        markers[peak] = region.label

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

    sure_bg               = dilation(fiber_mask, disk(disk_r))
    markers[sure_bg == 0] = markers.max() + 1

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


# ─── QC colormap helper ───────────────────────────────────────────────────────

def make_random_cmap(n_labels: int, seed: int = 42) -> mcolors.ListedColormap:
    """Randomized colormap. Label 0 (background) always black."""
    rng    = np.random.default_rng(seed)
    colors = rng.random((max(n_labels + 1, 2), 3))
    colors[0] = [0, 0, 0]
    return mcolors.ListedColormap(colors)


# ─── Main analysis ────────────────────────────────────────────────────────────

def get_morphometrics(
    seg_path: str,
    mag: str,
    log: DeepAxonLogger,
    return_labels: bool = False,
) -> pd.DataFrame | tuple | None:
    """
    Extract morphometric measurements from a single segmented BGW image.

    Args:
        seg_path:      path to segmented .tif (BGW: black=bg, grey=myelin, white=axon)
        mag:           magnification string e.g. '40X'
        log:           logger instance
        return_labels: if True, returns (df, axon_label, fiber_label, filtered_centroids)
                       instead of just df. Used by save_watershed_qc().

    Returns:
        DataFrame of per-axon measurements, or None on failure.
        If return_labels=True: (df, axon_label, fiber_label, filtered_centroids)
        where filtered_centroids is list of (x, y) tuples for filtered pairs.

    Quality filters:
        Toggled via config.json quality_filters.enabled (default: true).
        When disabled all matched pairs are kept regardless of morphology.
        Per-filter counters always logged when filters are active.
    """
    img = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        log.error(f"Could not read: {seg_path}")
        return (None, None, None, []) if return_labels else None

    h, w    = img.shape
    px_size = get_pixel_size(mag, w)
    config  = load_config()

    if px_size is None:
        log.warn(f"No pixel size for {mag} at width {w}px — outputting pixel units only")

    # ── Quality filter config ─────────────────────────────────────────────────
    qf         = config.get("quality_filters", {})
    qf_enabled = qf.get("enabled", True)

    min_axon_area_um2     = qf.get("min_axon_area_um2",       0.5)
    max_axon_area_um2     = qf.get("max_axon_area_um2",       500.0)
    min_axon_solidity     = qf.get("min_axon_solidity",       0.3)
    max_axon_eccentricity = qf.get("max_axon_eccentricity",   0.95)
    min_myelin_thick_um   = qf.get("min_myelin_thickness_um", 0.1)
    gratio_min            = qf.get("gratio_min",              0.2)
    gratio_max            = qf.get("gratio_max",              0.9)
    min_fiber_area_um2    = qf.get("min_fiber_area_um2",      1.0)
    max_fiber_area_um2    = qf.get("max_fiber_area_um2",      800.0)

    # ── Extract masks from BGW image ─────────────────────────────────────────
    # BGW contract: white(255)=axon, grey(128)=myelin
    # See segment/segment.py recolor() — this mapping must stay in sync.
    axon_mask  = cv2.inRange(img, 200, 255)
    fiber_mask = cv2.inRange(img, 1, 255)

    if not np.any(axon_mask):
        log.warn(f"No axons detected in {Path(seg_path).name}")
        return (pd.DataFrame(), None, None, []) if return_labels else pd.DataFrame()

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

    # Keep label arrays alive if return_labels requested
    axon_label_out  = axon_label  if return_labels else None
    fiber_label_out = fiber_label if return_labels else None

    if not return_labels:
        del axon_label, fiber_label
    del axon_mask, fiber_mask, img
    gc.collect()

    if axon_df.empty or fiber_df.empty:
        log.warn(f"No regions found in {Path(seg_path).name}")
        return (pd.DataFrame(), axon_label_out, fiber_label_out, []) if return_labels else pd.DataFrame()

    rows                 = []
    filtered_centroids   = []  # (x, y) of filtered pairs — used by QC sheet col 4
    assigned_axon_labels = set()

    # ── Per-filter counters (always tracked, logged when qf_enabled) ──────────
    n_filtered                = 0
    n_filtered_gratio_invalid = 0
    n_filtered_gratio_range   = 0
    n_filtered_solidity       = 0
    n_filtered_eccentricity   = 0
    n_filtered_area           = 0
    n_filtered_myelin         = 0

    def _filtered():
        filtered_centroids.append((x, y))
            
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
        myelin_thickness = (fiber_equiv_diam - axon_equiv_diam) / 2

        # ── G-ratios ──────────────────────────────────────────────────────────
        gratio_area = axon_equiv_diam / fiber_equiv_diam if fiber_equiv_diam > 0 else np.nan
        axon_mean_axis  = (axon_major + axon_minor) / 2
        fiber_mean_axis = (fiber_major + fiber_minor) / 2
        gratio_axes     = axon_mean_axis / fiber_mean_axis if fiber_mean_axis > 0 else np.nan

        if np.isnan(gratio_area) or np.isnan(gratio_axes):
            continue
        gratio = (gratio_area + gratio_axes) / 2

        # ── Quality filters ───────────────────────────────────────────────────
        # Toggled via config.json quality_filters.enabled.
        # filtered_centroids always populated for QC sheet col 4.
        # Per-filter counters always tracked for diagnostic logging.
        if qf_enabled:
            if not (0 < gratio_area < 1 and 0 < gratio_axes < 1 and 0 < gratio < 1):
                n_filtered += 1; n_filtered_gratio_invalid += 1; _filtered(); continue
            if not (gratio_min < gratio < gratio_max):
                n_filtered += 1; n_filtered_gratio_range += 1; _filtered(); continue
            if axon_solidity < min_axon_solidity:
                n_filtered += 1; n_filtered_solidity += 1; _filtered(); continue
            if axon_eccentricity > max_axon_eccentricity:
                n_filtered += 1; n_filtered_eccentricity += 1; _filtered(); continue

            if px_size is not None:
                axon_area_um2   = axon_area   * px_size ** 2
                fiber_area_um2  = fiber_area  * px_size ** 2
                myelin_thick_um = myelin_thickness * px_size

                if not (min_axon_area_um2 <= axon_area_um2 <= max_axon_area_um2):
                    n_filtered += 1; n_filtered_area += 1; _filtered(); continue
                if not (min_fiber_area_um2 <= fiber_area_um2 <= max_fiber_area_um2):
                    n_filtered += 1; n_filtered_area += 1; _filtered(); continue
                if myelin_thick_um < min_myelin_thick_um:
                    n_filtered += 1; n_filtered_myelin += 1; _filtered(); continue

        # ── Build output row ──────────────────────────────────────────────────
        new_row = {
            'label': int(row['label']),
            'x':     round(x, 1),
            'y':     round(y, 1),
            'axon_area':         round(axon_area, 2),
            'axon_perimeter':    round(axon_perimeter, 3),
            'axon_diam':         round(axon_equiv_diam, 3),
            'axon_major':        round(axon_major, 3),
            'axon_minor':        round(axon_minor, 3),
            'axon_solidity':     round(axon_solidity, 4),
            'axon_deformation':  round(axon_deformation, 4) if not np.isnan(axon_deformation) else np.nan,
            'axon_eccentricity': round(axon_eccentricity, 4),
            'axon_orientation':  round(axon_orientation, 4),
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
            'gratio_area': round(gratio_area, 4),
            'gratio_axes': round(gratio_axes, 4),
            'gratio':      round(gratio, 4),
        }

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
        return (pd.DataFrame(), axon_label_out, fiber_label_out, filtered_centroids) if return_labels else pd.DataFrame()

    log_breakdown = qf.get("log_filter_breakdown", True)
    
    if qf_enabled and n_filtered > 0:
        log.info(f"  Quality filters removed {n_filtered} pair(s)")
        if log_breakdown:
            log.info(f"    g-ratio invalid: {n_filtered_gratio_invalid} | "
                    f"g-ratio range: {n_filtered_gratio_range} | "
                    f"solidity: {n_filtered_solidity} | "
                    f"eccentricity: {n_filtered_eccentricity} | "
                    f"area: {n_filtered_area} | "
                    f"myelin: {n_filtered_myelin}")

    df = pd.DataFrame(rows)
    df.insert(0, 'image', Path(seg_path).stem)
    df.insert(1, 'resolution', f"{w}x{h}")
    df.insert(2, 'magnification', mag)

    if return_labels:
        return df, axon_label_out, fiber_label_out, filtered_centroids
    return df


# ─── Watershed QC sheet ───────────────────────────────────────────────────────

def save_watershed_qc(
    seg_images: list,
    all_data: list,
    morph_dir: Path,
    nerve_name: str,
    suffix: str,
    mag: str,
    log: DeepAxonLogger = None,
):
    """
    Generate a watershed QC sheet for all images in a nerve.
    Saved as {nerve_name}_watershed_qc{suffix}.png in the QC/ folder.

    Layout: one row per image
        Always:  Col 1: Axon watershed (randomized colormap)
                 Col 2: Fiber watershed (randomized colormap)
                 Col 3: Fiber colors + red axon outlines + red matched centroids (image label above)  # ← CHANGED
        qf ON:   Col 4: Filter map — green=kept, red=filtered, black=background
        qf OFF:  Col 4 omitted — 3 columns only

    Args:
        seg_images: list of Path — segmented image files in order
        all_data:   list of (df, axon_label, fiber_label, filtered_centroids)
                    parallel to seg_images
        morph_dir:  Path to Morphometrics folder (QC saved to sibling QC/ folder)
        nerve_name: str — used in filename and title
        suffix:     str — e.g. '_M1', '' for default
        mag:        str — magnification string
        log:        DeepAxonLogger
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        if log:
            log.warn("matplotlib not available — skipping watershed QC sheet")
        return

    config    = load_config()
    qc_folder = config.get("qc_folder", "QC")
    qc_dir    = morph_dir.parent / qc_folder
    qc_dir.mkdir(parents=True, exist_ok=True)

    n_images = len(seg_images)
    if n_images == 0:
        return

    qf_on  = config.get("quality_filters", {}).get("enabled", True)
    n_cols = 4 if qf_on else 3
    fig_w  = 24
    row_h  = fig_w / n_cols

    fig, axes = plt.subplots(
        n_images, n_cols,
        figsize=(fig_w, row_h * n_images),
        squeeze=False
    )

    col_titles = [
        "Axon watershed",
        "Fiber watershed",
        "Matched axons (grey overlay + red dots)",
    ]
    if qf_on:
        col_titles.append("Filter map (green=kept  red=filtered)")

    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=8, fontweight='bold', pad=6)

    wsh_cfg   = config.get("watershed", {})
    threshold = wsh_cfg.get("distance_threshold", 0.17)
    disk_r    = wsh_cfg.get("dilation_disk", 5)

    for row_idx, (img_path, data) in enumerate(zip(seg_images, all_data)):
        df, axon_lbl, fiber_lbl, filtered_xy = data

        ax_axon  = axes[row_idx][0]
        ax_fiber = axes[row_idx][1]
        ax_over  = axes[row_idx][2]
        ax_filt  = axes[row_idx][3] if qf_on else None

        n_kept     = len(df) if df is not None and not df.empty else 0
        n_filtered = len(filtered_xy)
        row_label  = (
            f"{img_path.name}  —  {n_kept} kept  |  {n_filtered} filtered"
            if qf_on else
            f"{img_path.name}  —  {n_kept} axons"
        )
        ax_over.set_title(row_label, fontsize=7, fontweight='normal', pad=4)   # ← CHANGED

        if axon_lbl is None or fiber_lbl is None:
            no_data_axes = [ax_axon, ax_fiber, ax_over]
            if ax_filt is not None:
                no_data_axes.append(ax_filt)
            for ax in no_data_axes:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10)
                ax.axis('off')
            continue

        # ── Col 1: Axon watershed ─────────────────────────────────────────────
        n_axon = axon_lbl.max()
        ax_axon.imshow(axon_lbl, cmap=make_random_cmap(n_axon),
                       interpolation='nearest')
        ax_axon.set_title(f"Axon watershed  ({n_axon} regions)", fontsize=7)
        ax_axon.axis('off')

        # ── Col 2: Fiber watershed ────────────────────────────────────────────
        n_fiber = fiber_lbl.max()
        ax_fiber.imshow(fiber_lbl, cmap=make_random_cmap(n_fiber, seed=99),
                        interpolation='nearest')
        ax_fiber.set_title(f"Fiber watershed  ({n_fiber} regions)", fontsize=7)
        ax_fiber.axis('off')

        # ── Col 3: Fiber colors + red axon outlines + red matched centroids (image label above) ──────────
        ax_over.imshow(fiber_lbl, cmap=make_random_cmap(n_fiber, seed=99),
                       interpolation='nearest', alpha=0.85)

        axon_boundary = np.zeros((*axon_lbl.shape, 4), dtype=np.float32)           # ← CHANGED
        binary_axon   = (axon_lbl > 0).astype(np.uint8)                            # ← NEW
        contours, _   = cv2.findContours(                                           # ← NEW
            binary_axon, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE                # ← NEW
        )                                                                           # ← NEW
        contour_mask  = np.zeros(axon_lbl.shape, dtype=np.uint8)                   # ← NEW
        cv2.drawContours(contour_mask, contours, -1, 1, thickness=1)               # ← NEW
        axon_boundary[contour_mask == 1] = [0.9, 0.15, 0.15, 0.9]                 # ← NEW — red, 90% opacity
        ax_over.imshow(axon_boundary, interpolation='nearest')   

        if df is not None and not df.empty and 'x' in df.columns and 'y' in df.columns:
            ax_over.scatter(df['y'].values, df['x'].values,
                            s=6, c='red', marker='.', linewidths=0)
        ax_over.axis('off')

        # ── Col 4: Filter map (qf ON only) ────────────────────────────────────
        if qf_on and ax_filt is not None:
            filter_rgb = np.zeros((*axon_lbl.shape, 3), dtype=np.float32)

            if df is not None and not df.empty and 'x' in df.columns and 'y' in df.columns:
                kept_mask = np.zeros(axon_lbl.shape, dtype=bool)
                for _, r in df.iterrows():
                    xi, yi = int(round(r['x'])), int(round(r['y']))
                    if 0 <= xi < axon_lbl.shape[0] and 0 <= yi < axon_lbl.shape[1]:
                        lbl_id = axon_lbl[xi, yi]
                        if lbl_id > 0:
                            kept_mask |= (axon_lbl == lbl_id)
                filter_rgb[kept_mask] = [0.2, 0.8, 0.2]

            if filtered_xy:
                for (fx, fy) in filtered_xy:
                    xi, yi = int(round(fx)), int(round(fy))
                    if 0 <= xi < axon_lbl.shape[0] and 0 <= yi < axon_lbl.shape[1]:
                        lbl_id = axon_lbl[xi, yi]
                        if lbl_id > 0:
                            filter_rgb[axon_lbl == lbl_id] = [0.9, 0.15, 0.15]

            ax_filt.imshow(filter_rgb, interpolation='nearest')
            ax_filt.axis('off')

    qf_label = "filters ON" if qf_on else "filters OFF"
    fig.suptitle(
        f"Watershed QC — {nerve_name}{suffix}  |  {mag}  |  "
        f"threshold={threshold}  disk={disk_r}  |  {qf_label}",
        fontsize=10, y=1.001
    )
    plt.tight_layout()

    out_path = qc_dir / f"{nerve_name}_watershed_qc{suffix}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)

    if log:
        log.success(f"Watershed QC saved → QC/{nerve_name}_watershed_qc{suffix}.png")


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