"""
morphometrics/__main__.py

Entry point for: python -m morphometrics

Walks a study directory, finds all *_Segmented folders,
runs morphometric analysis per image, saves per-image .xlsx files.
Skips nerves that already have a *_Morphometrics folder.
"""

import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
from pathlib import Path
from datetime import datetime
import time

from rich.console import Console
from rich.panel import Panel
from rich.box import DOUBLE
from rich.align import Align

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import DeepAxonLogger
from utils.helpers import detect_study_mag, load_config, list_files, resolve_scan, print_panel
from utils.resize import get_image_resolution
from morphometrics.morphometrics import get_morphometrics, save_morphometrics, save_watershed_qc
from morphometrics.distributions import bin_nerve_diameters, save_distributions

console = Console()

def main():
    run_start = time.time()
    print_panel(console, Panel(
        Align.center("[bold white]Automated Axon-Myelin Histomorphometry of Brightfield Images[/bold white]"),
        title="[bold cyan]DEEPAXON — MORPHOMETRICS[/bold cyan]",
        border_style="bright_cyan",
        box=DOUBLE,
        expand=True,
        padding=(1, 4)
    ))

    # ── Study folder input ────────────────────────────────────────────────────
    while True:
        input_dir = input("\nInput the path to the study, animal, or nerve folder containing the segmented images: ").strip().strip('"')
        if not os.path.isdir(input_dir):
            console.print(f"[red]✗  Folder not found: {input_dir}[/red]")
            continue
        try:
            study, study_dir = resolve_scan(input_dir)
            break
        except ValueError as e:
            console.print(f"[red]✗  {e}[/red]")
            continue

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    config          = load_config()
    # ── Read QC config ────────────────────────────────────────────────────
    qc_cfg          = config.get("qc", {})
    watershed_qc_on = qc_cfg.get("watershed", True)

    logging_cfg     = config.get("logging", {})
    logging_on      = logging_cfg.get("morphometrics", False) if isinstance(logging_cfg, dict) else bool(logging_cfg)
    
    log_path        = str(Path(study_dir) / f"morphometrics_log_{timestamp}.txt") if logging_on else None
    log             = DeepAxonLogger(log_path=log_path, program="DeepAxon Morphometrics")

    log.info(f"Study: {study_dir}")

    # ── Scan study ────────────────────────────────────────────────────────────
    log.rule("SCANNING STUDY")

    mag          = detect_study_mag(study)
    morph_folder = config.get("morphometrics_folder", "Morphometrics")
    seg_suffix   = config.get("segmented_suffix", "_segmented")

    log.info(f"Detected magnification: [bold]{mag}[/bold]")

    no_seg   = []
    to_process = []
    to_skip    = []

    for animal, nerves in study.items():
        for nerve, info in nerves.items():
            if info['segmented_dir'] is None:
                no_seg.append(f"{animal}/{nerve}")
                continue
            elif info['morphometrics_dir'] is not None:
                to_skip.append(f"{animal}/{nerve}")
            else:
                to_process.append((animal, nerve, info))

    if no_seg:
        log.warn(f"No segmentation found — {len(no_seg)} nerve(s) skipped")
    
    if to_skip:
        log.warn(f"Skipping {len(to_skip)} already-processed nerve(s):")
        for s in to_skip:
            log.warn(f"  {s}")

    if not to_process:
        log.success("All nerves already have morphometrics. Nothing to do.")
        sys.exit(0)

    log.info(f"To process: {len(to_process)} nerve(s)")

    # ── Process each nerve ────────────────────────────────────────────────────
    total_images = 0
    total_failed = 0

    current_animal = None
    for animal, nerve, info in to_process:
        if animal != current_animal:
            console.rule(animal, characters="═", style="white")
            current_animal = animal

        seg_dirs = info['segmented_dir']  # now a list
        if not seg_dirs:
            continue

        for seg_dir in seg_dirs:
            nerve_path = seg_dir.parent
            suffix     = seg_dir.name.replace(config.get('segmented_folder', 'Segmented'), '')
            morph_dir  = nerve_path / f"{morph_folder}{suffix}"

            log.rule(f"{nerve}" if not suffix else f"{nerve} — {seg_dir.name}")

            seg_images = list_files(str(seg_dir), extensions=('.tif', '.tiff'))
            if not seg_images:
                log.warn(f"No segmented images found in {seg_dir}")
                continue

            log.info(f"Found {len(seg_images)} segmented image(s)")

            # Resolution check
            resolutions = {}
            for img_path in seg_images:
                try:
                    w, h = get_image_resolution(str(img_path))
                    resolutions[img_path.name] = (w, h)
                except Exception as e:
                    log.warn(f"Could not read resolution for {img_path.name}: {e}")

            unique_res = set(resolutions.values())
            if len(unique_res) > 1:
                log.warn("Resolution mismatch detected:")
                for name, res in resolutions.items():
                    log.warn(f"  {name}: {res[0]}×{res[1]}")
            else:
                res = list(unique_res)[0] if unique_res else ('?', '?')
                log.info(f"Image resolution: {res[0]}×{res[1]} px")

            nerve_success = 0
            all_data      = []  # collect (df, axon_lbl, fiber_lbl, filtered_xy) per image

            for img_path in seg_images:
                stem = img_path.stem.replace(seg_suffix, '')
                res  = resolutions.get(img_path.name, ('?', '?'))
                log.info(f"  Processing {img_path.name} [{res[0]}×{res[1]}]...")

                try:
                    if watershed_qc_on:
                        df, axon_lbl, fiber_lbl, filtered_xy = get_morphometrics(
                            str(img_path), mag, log, return_labels=True
                        )
                        all_data.append((df, axon_lbl, fiber_lbl, filtered_xy))
                    else:
                        df = get_morphometrics(str(img_path), mag, log)
                        all_data.append((df, None, None, []))

                    if df is not None and not df.empty:
                        out = save_morphometrics(df, str(morph_dir), stem)
                        log.success(f"  → {len(df)} axons → {Path(out).name}")
                        total_images += 1
                        nerve_success += 1
                    else:
                        log.warn(f"  → No data extracted for {img_path.name}")
                except Exception as e:
                    log.error(f"  → FAILED: {e}")
                    all_data.append((None, None, None, []))
                    total_failed += 1

            # ── Watershed QC sheet ────────────────────────────────────────────────
            if watershed_qc_on and nerve_success > 0:
                try:
                    save_watershed_qc(
                        seg_images  = seg_images,
                        all_data    = all_data,
                        morph_dir   = morph_dir,
                        nerve_name  = nerve_path.name,
                        suffix      = suffix,
                        mag         = mag,
                        log         = log,
                    )
                except Exception as e:
                    log.error(f"Watershed QC failed: {e}")

            if nerve_success > 0:
                log.rule(f"DISTRIBUTIONS — {nerve}{suffix}")
                try:
                    data = bin_nerve_diameters(morph_dir, nerve_path.name, mag, log)
                    if data is not None:
                        dist_out = save_distributions(data, str(morph_dir), nerve_path.name)
                        log.success(f"  → {Path(dist_out).name}")
                    else:
                        log.warn(f"  → No distribution data for {nerve}")
                except Exception as e:
                    log.error(f"  → Distribution FAILED: {e}")

    elapsed     = time.time() - run_start
    avg_per_img = elapsed / total_images if total_images > 0 else 0

    log.finalize(summary={
        "Study":                          study_dir,
        "Magnification":                  mag,
        "Nerves — no segmentation":       len(no_seg),
        "Nerves — already processed":     len(to_skip),
        "Nerves — processed this run":    len(to_process),
        "Images processed":               total_images,
        "Images failed":                  total_failed,
        "Elapsed time":                   f"{int(elapsed // 60)}m {int(elapsed % 60)}s",
        "Avg time per image":             f"{avg_per_img:.1f}s",
    })
    
    if log_path:
        console.print(f"\n[dim]Log saved to: {log_path}[/dim]")


if __name__ == "__main__":
    main()