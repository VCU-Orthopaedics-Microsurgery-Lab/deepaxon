"""
morphometrics/distributions.py

Axon fiber diameter distribution analysis for DeepAxon.
Reads existing per-image morphometrics .xlsx files for a nerve,
pools all axons, and produces:
  - Global summary (total axons, mean fiber diameter, mean g-ratio)
  - Per-image summary (axon count, mean g-ratio per image)
  - Diameter distribution (binned fiber equiv diameter)

Requires fiber_equiv_diam_um and gratio columns — pixel size must be
calibrated in config.json for physical unit output.

Can be called standalone via: python -m morphometrics.distributions
Or from batch_axon/morphometrics pipelines.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from utils.helpers import load_config
from utils.logger import DeepAxonLogger


def bin_nerve_diameters(
    morph_dir: Path,
    nerve_name: str,
    mag: str,
    log: DeepAxonLogger
) -> dict | None:
    """
    Pool axon data across all per-image morphometrics files for a nerve.
    Computes global summary, per-image summary, and diameter distribution.

    Reads bin edges from config.json morphometrics_bins.
    Requires fiber_equiv_diam_um column — returns None if uncalibrated.

    Args:
        morph_dir:   Path to nerve Morphometrics/ folder
        nerve_name:  Name of the nerve (for metadata)
        mag:         Magnification string e.g. '40X'
        log:         Logger instance

    Returns:
        dict with keys: 'global_df', 'per_image_df', 'bins_df'
        Or None if no data found or pixel size uncalibrated.
    """
    config   = load_config()
    bins_cfg = config.get("morphometrics_bins", {})

    morph_dir = Path(morph_dir)
    if not morph_dir.exists():
        log.warn(f"Morphometrics folder not found: {morph_dir}")
        return None

    # ── Load and pool all morphometrics files ─────────────────────────────────
    morph_suffix = config.get("morphometrics_suffix", "_morphometrics")
    dfs          = {}
    for f in sorted(morph_dir.glob("*.xlsx")):
        # Skip previously generated output files
        if f.stem.endswith("_binned") or f.stem == "compiled_summary":
            continue
        try:
            df = pd.read_excel(str(f))
            dfs[f.stem.replace(morph_suffix, '')] = df
        except Exception as e:
            log.warn(f"Could not read {f.name}: {e}")

    if not dfs:
        log.warn(f"No morphometrics files found in {morph_dir}")
        return None

    agg_df = pd.concat(dfs.values(), ignore_index=True)
    log.info(f"  Pooled {len(agg_df)} axons from {len(dfs)} image(s)")

    # ── Require physical unit columns ─────────────────────────────────────────
    if 'fiber_equiv_diam_um' not in agg_df.columns or not agg_df['fiber_equiv_diam_um'].notna().any():
        log.warn(
            f"  fiber_equiv_diam_um not found — pixel size may not be calibrated for {mag}. "
            f"Distributions require physical unit calibration in config.json."
        )
        return None

    # ── Global summary ────────────────────────────────────────────────────────
    total_axons      = len(agg_df)
    mean_fiber_diam  = round(agg_df['fiber_equiv_diam_um'].mean(), 3)  if 'fiber_equiv_diam_um' in agg_df.columns else None
    mean_gratio      = round(agg_df['gratio'].mean(), 6)               if 'gratio'               in agg_df.columns else None
    mean_axon_diam   = round(agg_df['axon_diam_um'].mean(), 3)         if 'axon_diam_um'         in agg_df.columns else None

    global_df = pd.DataFrame([
        ['Nerve',                  nerve_name],
        ['Magnification',          mag],
        ['Total axons',            total_axons],
        ['Mean fiber diameter (µm)', mean_fiber_diam if mean_fiber_diam is not None else 'N/A'],
        ['Mean axon diameter (µm)',  mean_axon_diam  if mean_axon_diam  is not None else 'N/A'],
        ['Mean g-ratio',             mean_gratio     if mean_gratio     is not None else 'N/A'],
    ], columns=['Metric', 'Value'])

    # ── Per-image summary ─────────────────────────────────────────────────────
    per_image_rows = []
    for img_name, img_df in dfs.items():
        img_df['gratio'] = pd.to_numeric(img_df.get('gratio', pd.Series()), errors='coerce')
        per_image_rows.append({
            'Image':          img_name,
            'Axon Count':     len(img_df),
            'Mean G-ratio':   round(img_df['gratio'].dropna().mean(), 6) if 'gratio' in img_df.columns else None,
            'Mean Fiber Diam (µm)': round(img_df['fiber_equiv_diam_um'].dropna().mean(), 3) if 'fiber_equiv_diam_um' in img_df.columns else None,
        })
    per_image_df = pd.DataFrame(per_image_rows)

    # ── Diameter distribution ─────────────────────────────────────────────────
    diameters   = agg_df['fiber_equiv_diam_um'].dropna()
    total_axons = len(diameters)

    if total_axons == 0:
        log.warn(f"  No valid fiber diameters found")
        return None

    bin_edges = bins_cfg.get("axon_diameter_um", [0.5, 5, 10, 15, 20, 25, 30, 999])

    # right=False: left inclusive (≥), right exclusive (<). Last bin open-ended (≥).
    bins   = pd.cut(diameters, bins=bin_edges, right=False)
    counts = bins.value_counts(sort=False)

    percent    = (counts.values / total_axons * 100).round(2)
    cumulative = np.cumsum(percent).round(2)

    bin_labels = []
    for idx, b in enumerate(counts.index):
        if idx == len(counts.index) - 1:
            bin_labels.append(f"≥{b.left} µm")
        else:
            bin_labels.append(f"≥{b.left} and <{b.right} µm")

    bins_df = pd.DataFrame({
        'Diameter Range':     bin_labels,
        'Count':              counts.values.astype(int),
        'Percent (%)':        percent,
        'Cumulative (%)':     cumulative,
    })

    log.success(
        f"  Distribution complete: {len(bins_df)} bins | "
        f"{total_axons} axons | mean g-ratio={mean_gratio}"
    )

    return {
        'global_df':    global_df,
        'per_image_df': per_image_df,
        'bins_df':      bins_df,
    }


def save_distributions(
    data: dict,
    output_dir: str,
    nerve_name: str
) -> str:
    """
    Save distribution data to Excel with three sections:
      1. Global summary (nerve-level metadata + averages)
      2. Per-image summary (axon count + mean g-ratio per image)
      3. Diameter distribution (binned fiber equiv diameter)

    File named: {nerve_name}_binned.xlsx
    Returns output path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{nerve_name}_binned.xlsx"

    global_df    = data['global_df']
    per_image_df = data['per_image_df']
    bins_df      = data['bins_df']

    with pd.ExcelWriter(str(out_path), engine='openpyxl') as writer:
        current_row = 0

        # ── Global summary ────────────────────────────────────────────────────
        header_df = pd.DataFrame([['── Global Summary ──']], columns=[''])
        header_df.to_excel(writer, index=False, header=False, startrow=current_row)
        current_row += 1
        global_df.to_excel(writer, index=False, header=False, startrow=current_row)
        current_row += len(global_df) + 2

        # ── Per-image summary ─────────────────────────────────────────────────
        header_df = pd.DataFrame([['── Per-Image Summary ──']], columns=[''])
        header_df.to_excel(writer, index=False, header=False, startrow=current_row)
        current_row += 1
        per_image_df.to_excel(writer, index=False, startrow=current_row)
        current_row += len(per_image_df) + 3

        # ── Diameter distribution ─────────────────────────────────────────────
        header_df = pd.DataFrame([['── Diameter Distribution ──']], columns=[''])
        header_df.to_excel(writer, index=False, header=False, startrow=current_row)
        current_row += 1
        bins_df.to_excel(writer, index=False, startrow=current_row)

    return str(out_path)