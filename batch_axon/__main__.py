"""
batch_axon/__main__.py

Entry point for: python -m batch_axon

Walks study → animals → nerves, compiles morphometrics into a single
Excel workbook with one worksheet per animal.

Output: {study_name}_Data.xlsx saved in the study folder.
"""

import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
from pathlib import Path
from datetime import datetime

import xlsxwriter
from rich.console import Console
from rich.panel import Panel
from rich.box import DOUBLE
from rich.align import Align

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import DeepAxonLogger
from utils.helpers import detect_study_mag, resolve_scan, load_config, print_panel
from morphometrics.analyze_nerve import get_nerve_data
from morphometrics.distributions import save_distributions

console = Console()


# ─── Excel formatting helpers ─────────────────────────────────────────────────

def add_formats(workbook):
    return {
        'header': workbook.add_format({'border': 2, 'bold': True, 'bg_color': '#D9E1F2'}),
        'bold':   workbook.add_format({'bold': True}),
        'grey':   workbook.add_format({'bg_color': '#8a8a8a', 'pattern': 1}),
        'normal': workbook.add_format({}),
        'italic': workbook.add_format({'italic': True, 'font_color': '#666666'}),
        'total':  workbook.add_format({'bold': True, 'top': 2}),
    }


HEADERS = [
    'Slide / Nerve', 'Sample', 'CSA (µm²)',
    'Total Axons', 'G-ratio', 'Axon Diameter (µm)',
    'Est. Full Axon Count', 'Axons / µm²'
]


def write_nerve_block(worksheet, row, nerve_name, animal_name, image_rows, aggregate, formats):
    """Write a nerve's data block to the worksheet. Returns next available row."""

    # Header row
    for col, h in enumerate(HEADERS):
        worksheet.write(row, col, h, formats['header'])
    row += 1

    # Nerve label row (4X CSA)
    label    = f"{animal_name} - {nerve_name}"
    fourx_csa = aggregate.get('fourx_csa_um2')
    worksheet.write(row, 0, label, formats['bold'])
    worksheet.write(row, 1, f"{animal_name}_4X_{nerve_name}")
    worksheet.write(row, 2, fourx_csa if fourx_csa else 'N/A', formats['bold'])
    for col in range(3, 8):
        worksheet.write(row, col, '', formats['grey'])
    row += 1

    # Per-image rows
    first_data_row = row
    for entry in image_rows:
        worksheet.write(row, 1, entry.get('name', ''))
        worksheet.write(row, 2, entry.get('csa_um2') or '')
        worksheet.write(row, 3, entry.get('total_axons') or '')
        worksheet.write(row, 4, round(entry['gratio'], 4) if entry.get('gratio') is not None else '')
        worksheet.write(row, 5, round(entry['axon_diam_um'], 3) if entry.get('axon_diam_um') is not None else '')
        worksheet.write(row, 6, '', formats['grey'])
        dens = entry.get('axon_density_per_um2')
        worksheet.write(row, 7, round(dens, 6) if dens is not None else '')
        row += 1

    last_data_row = row - 1

    # Conditional formatting
    if last_data_row >= first_data_row:
        worksheet.conditional_format(first_data_row, 3, last_data_row, 3, {
            'type': '2_color_scale',
            'min_color': '#ffffff', 'max_color': '#d53c3c'
        })
        worksheet.conditional_format(first_data_row, 4, last_data_row, 4, {
            'type': '2_color_scale',
            'min_color': '#ffffff', 'max_color': '#2DB133'
        })

    # Totals row
    n         = aggregate.get('total_images', 0)
    est       = aggregate.get('estimated_total_axons')
    total_csa = aggregate.get('total_sample_csa_um2')

    worksheet.write(row, 1, f"Totals (n={n})", formats['total'])
    worksheet.write(row, 2, round(total_csa, 2) if total_csa else '', formats['total'])
    worksheet.write(row, 3, aggregate.get('total_axons', ''), formats['total'])
    worksheet.write(row, 4, round(aggregate['mean_gratio'], 4) if aggregate.get('mean_gratio') is not None else '', formats['total'])
    worksheet.write(row, 5, round(aggregate['mean_axon_diam_um'], 3) if aggregate.get('mean_axon_diam_um') is not None else '', formats['total'])
    worksheet.write(row, 6, est if est is not None else '', formats['total'])
    if total_csa and total_csa > 0 and aggregate.get('total_axons'):
        worksheet.write(row, 7, round(aggregate['total_axons'] / total_csa, 6), formats['total'])

    row += 4  # Gap between nerves
    return row


def main():
    print_panel(console, Panel(
        Align.center("[bold white]Study-Level Morphometric\nCompilation[/bold white]"),
        title="[bold cyan]DEEPAXON — BATCH AXON[/bold cyan]",
        border_style="bright_cyan",
        box=DOUBLE,
        expand=True,
        padding=(1, 4)
    ))

    # ── Study folder input ────────────────────────────────────────────────────
    while True:
        input_dir = input("\nInput the path to the study, animal, or nerve folder: ").strip().strip('"')
        if not os.path.isdir(input_dir):
            console.print(f"[red]✗  Folder not found: {input_dir}[/red]")
            continue
        try:
            study, study_dir = resolve_scan(input_dir)
            break
        except ValueError as e:
            console.print(f"[red]✗  {e}[/red]")
            continue

    study_path = Path(study_dir)
    study_name = study_path.name
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    config     = load_config()
    logging_cfg = config.get("logging", {})
    logging_on  = logging_cfg.get("batch_axon", False) if isinstance(logging_cfg, dict) else bool(logging_cfg)
    log_path    = str(study_path / f"batch_axon_log_{timestamp}.txt") if logging_on else None
    log        = DeepAxonLogger(log_path=log_path, program="DeepAxon Batch Axon")

    log.info(f"Study: {study_dir}")

    # ── Scan study ────────────────────────────────────────────────────────────
    log.rule("SCANNING STUDY")
    mag          = detect_study_mag(study)
    total_nerves = sum(len(nerves) for nerves in study.values())
    log.info(f"Detected magnification: [bold]{mag}[/bold]")
    log.info(f"Animals: {len(study)} | Nerves: {total_nerves}")

    # ── Create workbook ───────────────────────────────────────────────────────
    workbook_path = study_path / f"{study_name}_Data.xlsx"
    workbook      = xlsxwriter.Workbook(str(workbook_path))
    formats       = add_formats(workbook)

    # ── Provenance sheet (first sheet) ────────────────────────────────────────
    from utils.version import __version__                                          
    prov_sheet  = workbook.add_worksheet('Provenance')                             
    prov_fmt    = workbook.add_format({'italic': True, 'font_color': '#666666'})   
    bold_fmt    = workbook.add_format({'bold': True})                              
    config_prov = load_config()                                                    
    wsh_cfg     = config_prov.get('watershed', {})                                 
    prov_rows   = [                                                                
        ('Study',                 str(study_path)),                                
        ('Generated',             datetime.now().strftime('%Y-%m-%d')),            
        ('DeepAxon version',      __version__),                                    
        ('Magnification',         mag),                                            
        ('Primary g-ratio method',config_prov.get('primary_gratio_method', 'equiv_diam')),  
        ('Watershed threshold',   wsh_cfg.get('distance_threshold', 0.17)),        
        ('Watershed disk',        wsh_cfg.get('dilation_disk', 5)),                
        ('CLAHE',                 'ON' if config_prov.get('clahe', {}).get('enabled') else 'OFF'),  
    ]                                                                              
    for r_idx, (label, value) in enumerate(prov_rows):                            
        prov_sheet.write(r_idx, 0, label, prov_fmt)                                
        prov_sheet.write(r_idx, 1, str(value))                                     
    prov_sheet.set_column(0, 0, 28)                                                
    prov_sheet.set_column(1, 1, 50)                                                

    nerves_processed = 0
    nerves_failed    = 0

    try:
        # ── Process animals ───────────────────────────────────────────────────
        for animal_name, nerves in study.items():
            if not nerves:
                log.warn(f"No nerves found for {animal_name}, skipping")
                continue

            log.rule(f"ANIMAL: {animal_name}")
            worksheet   = workbook.add_worksheet(animal_name[:31])  # Excel sheet name limit
            current_row = 0

            for nerve_name, info in nerves.items():
                log.info(f"  Processing nerve: {nerve_name}")
                nerve_path = info['tiff_dir'].parent

                if info['morphometrics_dir'] is None:
                    log.warn(f"  → No morphometrics folder found for {nerve_name}, skipping")
                    nerves_failed += 1
                    continue

                try:
                    image_rows, aggregate, bins_df = get_nerve_data(nerve_path, mag, log)

                    if not image_rows:
                        log.warn(f"  → No data returned for {nerve_name}")
                        nerves_failed += 1
                        continue

                    current_row = write_nerve_block(
                        worksheet, current_row,
                        nerve_name, animal_name,
                        image_rows, aggregate, formats
                    )
                    
                    # Save per-nerve binned distribution file
                    if bins_df is not None:
                        try:
                            morph_dir = nerve_path / config.get("morphometrics_folder", "Morphometrics")
                            dist_out  = save_distributions(bins_df, str(morph_dir), nerve_path.name)
                            log.success(f"  → Distribution saved: {Path(dist_out).name}")
                        except Exception as e:
                            log.error(f"  → Distribution save FAILED: {e}")
                    
                
                    log.success(
                        f"  → {nerve_name}: {aggregate.get('total_axons', 0)} axons, "
                        f"{aggregate.get('total_images', 0)} images"
                    )
                    nerves_processed += 1

                except Exception as e:
                    log.error(f"  → FAILED for {nerve_name}: {e}")
                    nerves_failed += 1

            # Column widths
            worksheet.set_column(0, 1, 28)
            worksheet.set_column(2, 7, 18)

    finally:
        workbook.close()

    log.success(f"Workbook saved: {workbook_path}")

    log.finalize(summary={
        'Study':                study_dir,
        'Magnification':        mag,
        'Workbook':             str(workbook_path),
        'Nerves processed':     nerves_processed,
        'Nerves failed/skipped':nerves_failed,
    })
    console.print(f"\n[dim]Log saved to: {log_path}[/dim]")


if __name__ == "__main__":
    main()