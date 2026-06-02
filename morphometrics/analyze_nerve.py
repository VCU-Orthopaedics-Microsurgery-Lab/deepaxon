"""
batch_axon/analyze_nerve.py

Compiles per-image morphometrics into nerve-level and image-level summaries.
Handles CSA extraction from Fiji overlays for 40X and 4X magnifications.
For 100X images, only the 4X whole-nerve CSA is used — per-image CSA is not
required since the image field captures only axonal area.
Reads conversion factors from config.json — no hardcoded values.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from utils.helpers import get_pixel_size, load_config
from utils.logger import DeepAxonLogger
from utils.resize import TARGET_SIZE
from batch_axon.overlay.process_overlay import get_overlay_area
from morphometrics.distributions import bin_nerve_diameters


def make_aggregate_df(morph_dir: Path, log: DeepAxonLogger) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    dfs      = []
    per_file = {}
    for f in sorted(morph_dir.glob("*.xlsx")):
        if f.stem.endswith("_binned"):                                 
            continue
        try:
            df = pd.read_excel(str(f), sheet_name='Axon')             
            dfs.append(df)
            per_file[f.stem] = df
        except Exception as e:
            log.warn(f"Could not read {f.name}: {e}")

    if not dfs:
        return pd.DataFrame(), {}

    return pd.concat(dfs, ignore_index=True), per_file


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
    For 100X: per-image CSA not applicable — returns None.
              Use fourx_csa from get_nerve_fourx_csa() instead.
    For unknown mag: returns None.
    """
    config     = load_config()
    csa_suffix = config.get("csa_suffix", "_CSA.tif")
    px_size    = get_pixel_size(mag, img_width)

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
        # Per-image CSA not required for 100X — image field = axonal area only.
        # Axon density for 100X is calculated using the 4X whole-nerve CSA.
        return None

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
    Handles single or multiple CSA files (large nerve spanning multiple images).
    File pattern: {nerve_name}_4X_CSA.tif or {nerve_name}_4X_###_CSA.tif
    """
    config     = load_config()
    csa_suffix = config.get("csa_suffix", "_CSA.tif")
    px_size    = get_pixel_size("4X", 2880)  # 4X images are always full res (2880px wide)

    # Search for all 4X CSA files — case insensitive on the X
    candidates = list(dict.fromkeys(
        list(csa_dir.glob(f"{nerve_name}_4X{csa_suffix}"))   +
        list(csa_dir.glob(f"{nerve_name}_4X_*{csa_suffix}")) +
        list(csa_dir.glob(f"{nerve_name}_4x{csa_suffix}"))   +
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
    config        = load_config()
    morph_folder  = config.get("morphometrics_folder", "Morphometrics")
    morph_suffix  = config.get("morphometrics_suffix", "_morphometrics")
    morph_dir     = nerve_dir / morph_folder
    csa_dir       = nerve_dir / config.get("csa_folder", "CSA")

    if not morph_dir.exists():
        log.warn(f"No morphometrics folder found: {morph_dir}")
        return [], {}

    # Load all morphometrics — aggregate df and per-file cache
    agg_df, per_file = make_aggregate_df(morph_dir, log)
    bins_df = bin_nerve_diameters(morph_dir, nerve_dir.name, mag, log)
    if agg_df.empty:
        log.warn(f"No morphometrics data found in {morph_dir}")
        return [], {}

    # 4X whole-nerve CSA — used for all magnifications
    fourx_csa = None
    if csa_dir.exists():
        fourx_csa = get_nerve_fourx_csa(nerve_dir.name, csa_dir, log)

    # Per-image rows — use cached per_file dict to avoid double disk reads
    image_rows = []
    img_count  = 0

    for morph_file in sorted(morph_dir.glob("*.xlsx")):
        img_name = morph_file.stem.replace(morph_suffix, '')
        img_df   = per_file.get(morph_file.stem)
        if img_df is None:
            continue

        # Get image width from resolution column if available
        img_width = TARGET_SIZE[0]  # Default: 1440px post-resize width
        if 'resolution' in img_df.columns and len(img_df) > 0:
            try:
                res_str   = str(img_df['resolution'].iloc[0])
                img_width = int(res_str.split('x')[0])
            except Exception:
                pass

        # Per-image CSA — 40X only, None for 100X
        img_csa = None
        if csa_dir.exists():
            img_csa = get_image_csa(img_name, csa_dir, mag, img_width, log)

        px_size = get_pixel_size(mag, img_width)

        _gratio_method = config.get('primary_gratio_method', 'equiv_diam')         
        _gratio_col    = (                                                           
            'gratio_equiv_diam' if _gratio_method == 'equiv_diam'                  
            else 'gratio_mean_axes'                                                 
        )                                                                            
        row = {
            'name':        img_name,
            'resolution':  f"{img_width}px",
            'csa_um2':     img_csa,
            'total_axons': len(img_df),
            'gratio':      (                                                         
                img_df[_gratio_col].mean() if _gratio_col in img_df.columns        
                else img_df['gratio'].mean() if 'gratio' in img_df.columns         
                else None                                                           
            ),                                                                      
            'axon_diam_um': (
                img_df['axon_diam_um'].mean() if 'axon_diam_um' in img_df.columns else
                img_df['axon_diam_px'].mean() * px_size if px_size and 'axon_diam_px' in img_df.columns else  
                None
            ),
        }

        # Axon density — use per-image CSA for 40X, fourx_csa for 100X
        csa_for_density = img_csa if mag == "40X" else fourx_csa
        row['axon_density_per_um2'] = (
            len(img_df) / csa_for_density if csa_for_density and csa_for_density > 0 else None
        )

        image_rows.append(row)
        img_count += 1

    # Aggregate summary
    aggregate = {
        'fourx_csa_um2': fourx_csa,
        'total_images':  img_count,
        'total_axons':   len(agg_df),
        'mean_gratio':   (                                                           
            agg_df[_gratio_col].mean() if _gratio_col in agg_df.columns            
            else agg_df['gratio'].mean() if 'gratio' in agg_df.columns             
            else None                                                               
        ),                                                                          
    }

    # Axon diameter — prefer um column, fall back to pixel column × px_size
    if 'axon_diam_um' in agg_df.columns:
        aggregate['mean_axon_diam_um'] = agg_df['axon_diam_um'].mean()
    elif 'axon_diam_px' in agg_df.columns:                                          
        px_size = get_pixel_size(mag, TARGET_SIZE[0])
        aggregate['mean_axon_diam_um'] = agg_df['axon_diam_px'].mean() * px_size if px_size else None  
    else:
        aggregate['mean_axon_diam_um'] = None

    # Total sample CSA (sum of per-image CSAs) — 40X only
    valid_csas = [r['csa_um2'] for r in image_rows if r.get('csa_um2')]
    aggregate['total_sample_csa_um2'] = sum(valid_csas) if valid_csas else None

    # Estimated full nerve axon count — extrapolated using 4X CSA / sample CSA ratio
    # For 100X: total_sample_csa_um2 will be None so falls back to fourx_csa directly
    sample_csa = aggregate.get('total_sample_csa_um2')
    if fourx_csa and sample_csa and sample_csa > 0:
        aggregate['estimated_total_axons'] = int(
            aggregate['total_axons'] / sample_csa * fourx_csa
        )
    else:
        aggregate['estimated_total_axons'] = None

    return image_rows, aggregate, bins_df