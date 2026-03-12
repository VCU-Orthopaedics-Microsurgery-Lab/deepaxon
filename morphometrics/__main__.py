"""
morphometrics/__main__.py

Entry point for: python morphometrics

Walks a study directory, finds all *_Segmented folders,
runs morphometric analysis per image, saves per-image .xlsx files.
Skips nerves that already have a *_Morphometrics folder.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.console import DeepAxonLogger
from utils.helpers import scan_study, detect_study_mag, load_config, list_files
from morphometrics.morphometrics import get_morphometrics, save_morphometrics

console = Console()


def main():
    console.print(Panel(
        "[bold white]Per-Image Nerve Morphometrics[/bold white]",
        title="[bold cyan]DeepAxon — Morphometrics[/bold cyan]",
        border_style="cyan",
        expand=False
    ))

    # ── Study folder input ────────────────────────────────────────────────────
    study_dir = input("\nInput the path to the study folder: ").strip().strip('"')
    if not os.path.isdir(study_dir):
        console.print(f"[red]Study folder not found: {study_dir}[/red]")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = str(Path(study_dir) / f"morphometrics_log_{timestamp}.txt")
    log = DeepAxonLogger(log_path=log_path, program="DeepAxon Morphometrics")

    log.info(f"Study: {study_dir}")

    # ── Scan study ────────────────────────────────────────────────────────────
    log.rule("SCANNING STUDY")
    study = scan_study(study_dir)

    mag = detect_study_mag(study)
    log.info(f"Detected magnification: [bold]{mag}[/bold]")

    config = load_config()
    morph_folder = config.get("morphometrics_folder", "Morphometrics")

    to_process = []
    to_skip = []

    for animal, nerves in study.items():
        for nerve, info in nerves.items():
            if info['segmented_dir'] is None:
                log.warn(f"  {animal}/{nerve} — no Segmented folder, skipping")
                continue
            if info['morphometrics_dir'] is not None:
                to_skip.append(f"{animal}/{nerve}")
            else:
                to_process.append((animal, nerve, info))

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

    for animal, nerve, info in to_process:
        seg_dir = info['segmented_dir']
        nerve_path = seg_dir.parent
        morph_dir = nerve_path / morph_folder

        log.rule(f"{animal} / {nerve}")

        seg_images = list_files(str(seg_dir), extensions=('.tif', '.tiff'))
        if not seg_images:
            log.warn(f"No segmented images found in {seg_dir}")
            continue

        log.info(f"Found {len(seg_images)} segmented image(s)")

        # Resolution check
        from utils.resize import get_image_resolution
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

        for img_path in seg_images:
            stem = img_path.stem.replace('_segmented', '')
            res = resolutions.get(img_path.name, ('?', '?'))
            log.info(f"  Processing {img_path.name} [{res[0]}×{res[1]}]...")

            try:
                df = get_morphometrics(str(img_path), mag, log)
                if df is not None and not df.empty:
                    out = save_morphometrics(df, str(morph_dir), stem)
                    log.success(f"  → {len(df)} axons → {Path(out).name}")
                    total_images += 1
                else:
                    log.warn(f"  → No data extracted for {img_path.name}")
            except Exception as e:
                log.error(f"  → FAILED: {e}")
                total_failed += 1

    log.finalize(summary={
        "Study": study_dir,
        "Magnification": mag,
        "Images processed": total_images,
        "Images failed": total_failed,
        "Nerves processed": len(to_process),
        "Nerves skipped": len(to_skip),
    })
    console.print(f"\n[dim]Log saved to: {log_path}[/dim]")


if __name__ == "__main__":
    main()
