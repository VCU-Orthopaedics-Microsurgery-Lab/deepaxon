"""
morphometrics/distributions.py

Axon fiber diameter distribution analysis for DeepAxon.
Reads existing per-image morphometrics .xlsx files for a nerve,
pools all axons, and produces:
  - Global summary (total axons, mean fiber diameter, mean g-ratio)
  - Per-image summary (axon count, mean g-ratio per image)
  - Three-tier diameter distribution with g-ratio mean + SD per bin:
      Granular : 0.5µm bins, 0–8µm physiological range + ≥8µm catch
      Mid      : 2µm bins,   0–16µm broad physiological range + ≥16µm catch
      Coarse   : 5µm bins,   full range global QC scan + ≥30µm catch

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
    log: DeepAxonLogger,
    model_name: str = '',
    clahe_cfg: dict = None, 
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
        dict with keys: 'global_df', 'per_image_df', 'bins_granular_df', 'bins_mid_df', 'bins_coarse_df'
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

    clahe_cfg  = clahe_cfg or {}
    clahe_on   = clahe_cfg.get('enabled', False)
    clahe_str  = f"ON (clip={clahe_cfg.get('clip_limit', 1.0)})" if clahe_on else 'OFF'  # ← NEW

    global_df = pd.DataFrame([
        ['Nerve',                  nerve_name],
        ['Magnification',          mag],
        ['Model',                  model_name or 'unknown'],                               # ← NEW
        ['CLAHE',                  clahe_str],                                             # ← NEW
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

    # ── Diameter distribution — three-tier ───────────────────────────────────
    diameters   = agg_df['fiber_equiv_diam_um'].dropna()
    gratios     = agg_df['gratio'].dropna() if 'gratio' in agg_df.columns else None
    total_axons = len(diameters)

    if total_axons == 0:
        log.warn(f"  No valid fiber diameters found")
        return None

    # Default bin edges if config keys missing
    default_granular = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 999]
    default_mid      = [0, 2, 4, 6, 8, 10, 12, 14, 16, 999]
    default_coarse   = [0, 5, 10, 15, 20, 25, 30, 999]

    granular_edges = bins_cfg.get("granular_um", default_granular)
    mid_edges      = bins_cfg.get("mid_um",      default_mid)
    coarse_edges   = bins_cfg.get("coarse_um",   default_coarse)

    def make_bins_df(edges: list) -> pd.DataFrame:
        """
        Bin diameters by edges, compute count/percent/cumulative and
        g-ratio mean+SD per bin. Last bin is open-ended catch (≥ lower bound).
        right=False: ≥ left, < right. 999 upper bound = open-ended catch box.
        """
        # Align gratio to the same index as diameters for per-bin lookup
        diam_series   = agg_df['fiber_equiv_diam_um'].dropna()
        gratio_series = agg_df.loc[diam_series.index, 'gratio'] if 'gratio' in agg_df.columns else None

        cut      = pd.cut(diam_series, bins=edges, right=False)
        groups   = diam_series.groupby(cut, observed=False)
        counts   = groups.count()
        percent    = (counts.values / total_axons * 100).round(2)
        cumulative = np.cumsum(percent).round(2)

        # G-ratio per bin — use mask-based lookup to avoid KeyError on empty intervals
        gratio_mean = []
        gratio_sd   = []
        if gratio_series is not None:
            for interval in counts.index:
                mask = (cut == interval)
                g    = gratio_series[mask].dropna()
                gratio_mean.append(round(g.mean(), 6) if len(g) > 0 else None)
                gratio_sd.append(round(g.std(),  6)   if len(g) > 1 else None)
        else:
            gratio_mean = [None] * len(counts)
            gratio_sd   = [None] * len(counts)

        # Build labels — last bin is catch box
        labels = []
        for idx, b in enumerate(counts.index):
            if idx == len(counts.index) - 1:
                labels.append(f"≥{b.left} µm")
            else:
                labels.append(f"≥{b.left} and <{b.right} µm")

        return pd.DataFrame({
            'Diameter Range':  labels,
            'Count':           counts.values.astype(int),
            'Percent (%)':     percent,
            'Cumulative (%)':  cumulative,
            'Mean G-ratio':    gratio_mean,
            'SD G-ratio':      gratio_sd,
        })

    bins_granular_df = make_bins_df(granular_edges)
    bins_mid_df      = make_bins_df(mid_edges)
    bins_coarse_df   = make_bins_df(coarse_edges)

    log.success(
        f"  Distribution complete: granular={len(bins_granular_df)} bins | "
        f"mid={len(bins_mid_df)} bins | coarse={len(bins_coarse_df)} bins | "
        f"{total_axons} axons | mean g-ratio={mean_gratio}"
    )

    return {
        'global_df':         global_df,
        'per_image_df':      per_image_df,
        'bins_granular_df':  bins_granular_df,
        'bins_mid_df':       bins_mid_df,
        'bins_coarse_df':    bins_coarse_df,
    }


def save_distributions(
    data: dict,
    output_dir: str,
    nerve_name: str
) -> str:
    """
    Save distribution data to Excel with five sections:
      1. Global summary (nerve-level metadata + averages)
      2. Per-image summary (axon count + mean g-ratio per image)
      3. Granular distribution (0.5µm bins, 0–8µm + catch)
      4. Mid distribution (2µm bins, 0–16µm + catch)
      5. Coarse distribution (5µm bins, full range + catch)

    File named: {nerve_name}_binned.xlsx
    Returns output path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{nerve_name}_binned.xlsx"

    global_df        = data['global_df']
    per_image_df     = data['per_image_df']
    bins_granular_df = data['bins_granular_df']
    bins_mid_df      = data['bins_mid_df']
    bins_coarse_df   = data['bins_coarse_df']

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

        # ── Granular distribution (0.5µm bins, 0–8µm + catch) ────────────────
        header_df = pd.DataFrame([['── Diameter Distribution — Granular (0.5µm bins, physiological range) ──']], columns=[''])
        header_df.to_excel(writer, index=False, header=False, startrow=current_row)
        current_row += 1
        bins_granular_df.to_excel(writer, index=False, startrow=current_row)
        current_row += len(bins_granular_df) + 3

        # ── Mid distribution (2µm bins, 0–16µm + catch) ───────────────────────
        header_df = pd.DataFrame([['── Diameter Distribution — Mid (2µm bins, broad physiological range) ──']], columns=[''])
        header_df.to_excel(writer, index=False, header=False, startrow=current_row)
        current_row += 1
        bins_mid_df.to_excel(writer, index=False, startrow=current_row)
        current_row += len(bins_mid_df) + 3

        # ── Coarse distribution (5µm bins, full range + catch) ────────────────
        header_df = pd.DataFrame([['── Diameter Distribution — Coarse (5µm bins, global QC scan) ──']], columns=[''])
        header_df.to_excel(writer, index=False, header=False, startrow=current_row)
        current_row += 1
        bins_coarse_df.to_excel(writer, index=False, startrow=current_row)

    return str(out_path)