# DeepAxon — To-Do List
**Updated:** June 2026 | **Branch:** v5_analysis
**Previous sessions:** S1 Audit, S2 Bug fixing, S3 PyTorch migration, S4 Phenotype split, S5 Training overhaul, S6 Analysis pipeline, S7 Morphometrics refactor + pipeline hardening
**All changes through S7 committed and pushed to v5_analysis.**

---

## Priority legend
- 🔴 Critical — verification required before results can be trusted
- 🟠 High — significant issue or planned breaking change
- 🟡 Medium — improvement or cleanup
- 📄 Docs — documentation task
- ✅ Resolved

---

## Section 1 — Critical

**[CRIT-02] BGW Color Mapping** ✅
Segmentation visually confirmed correct on real images. G-ratio values physiologically
plausible in current morphometrics outputs. Practical verification complete.

**[CRIT-08] ROI Coordinate Space — Deferred**
Deferred to QuPath migration project. Not blocking current analysis.
Will be addressed when workflow transitions from Fiji ROI export to QuPath/GeoJSON.
No action until that project begins.

---

## Section 2 — Analysis Pipeline (v5_analysis branch)

**[ANAL-01] Fill real Athena paths into analysis_config.json** 🟠
Replace all `/path/to/` placeholders before Wave 1 launch.

**[ANAL-02] Install monai on Athena venv** 🟠
`pip install monai==1.5.2 --break-system-packages`

**[ANAL-03] Create output directories on Athena** 🟠
`~/deepaxon/analysis/{jobs,results,models,logs,aggregated}/`

**[ANAL-04] Preprocessing run on Athena** 🟠
Generate patches from 30 images before submitting Wave 1.

**[ANAL-05] Dry run test** 🟠
`python wave1_launcher.py --config analysis_config.json --dry-run`
Verify one job config generates correctly, sbatch script is valid.

**[ANAL-06] ADS and MedSAM external benchmark runs** 🟡
Run after Wave 1 completes. Required for Table 2 (ADS and MedSAM rows).
Separate inference pipelines — not part of the SLURM job array.

**[ANAL-07] aggregator.py Wave 2b — pull aug OFF from Wave 1 SW** 🟡
`build_table5()` currently matches by arch/encoder/weights/split/seed.
Verify matching logic after Wave 1 results are in and seeds are confirmed.

**[ANAL-08] winner_aug.json format** 📄
Document expected format after Wave 2a review:
`{ "optimized_params": { "hflip_prob": 0.75, ... } }`
Add example to aggregator.py --wave 2a terminal output.

**[ANAL-09] Production model retrain after Wave 3** 🟠
No .pt files saved during analysis pipeline (save_checkpoint=False throughout).
After Wave 3 completes and winner is selected, retrain once interactively:
`python -m train`
Names output `rb40x_v2.pt` — production model for Paper 1.

---

## Section 3 — Morphometrics

**[MORPH-01] File-level skip logic in morphometrics/__main__.py** 🟡
Currently skips whole nerve if Morphometrics/ folder exists.
Should skip individual images if {stem}_morphometrics.xlsx already exists.
Allows adding new images to a nerve without reprocessing everything.

**[MORPH-02] batch_axon/__main__.py — full run test** 🟡
Confirm Provenance sheet writes correctly, study workbook structure intact.
Test with a full study folder containing multiple animals and nerves.

**[MORPH-03] Delete batch_axon/analyze_nerve.py** 🟡
Moved to morphometrics/analyze_nerve.py. Old file still present — delete after
confirming batch_axon/__main__.py imports correctly from new location.

---

## Section 4 — Docs

**[DOC-01] README — v5_analysis update** 📄
Update training section to reflect:
- `ctrl_`/`regen_` prefixes instead of `val_` prefix
- sbatch mode (`python -m train --config`) instead of srun interactive
- v5_analysis branch
- New analysis pipeline entry points
- Fine-tuning entry point (python -m train.finetune)

**[DOC-03] Annotation protocol in README** 📄
Add Training section covering minimum mask counts, phenotype balance, naming convention.

---

## Section 5 — Post-Analysis / Paper 2 (do not implement mid-study)

These require full retraining or are Paper 2 scope. Listed here for reference only.

- STR-05: Extract select_model() to utils/helpers.py
- STR-06: Extract Excel helpers to batch_axon/excel_utils.py
- STR-09: Add evaluate/ module
- STR-10: Add models/model_registry.json
- STR-11: Archive rb_40x_v1_256.keras
- FEAT-21: visualize.py QC overlays
- FEAT-22: Per-phenotype val dice tracking
- FEAT-23: Boundary loss
- CRIT-08: QuPath/GeoJSON workflow
- GEN-01: 100X pixel size calibration
- GEN-04: Multi-GPU support
- FINETUNE-01: Fine-tuning sbatch support (interactive only currently)

---

## Section 6 — V2 Production Model (post-Paper 1)

Do not implement any of these mid-study — all require full retraining.

- V2-01: Z-score normalization per patch
- V2-02: Stain normalization (Macenko/Vahadane)
- V2-03: Instance normalization in encoder
- V2-04: Channel attention (CBAM or SE block)
- V2-05: Per-image statistics as conditioning input
- V2-06: Contrastive pretraining of encoder
- V2-07: Two-stage cascade architecture
- V2-08: Augmented rotation range (±15° → ±180°)
- V2-09: Increased noise sigma (0.02 → 0.05–0.08)
- V2-10: Boundary loss component

---

## Resolved — S6 (v5_analysis)

✅ Wave 1 SW launcher — LC removed, SW only, 5,760 jobs, 3 splits
✅ Wave 2 launcher — 2a OAT/matrix sweep, 2b aug ON only (5 jobs)
✅ Wave 3 launcher — LC on fully optimized model, 45 jobs
✅ aggregator.py — Wave 1/2a/2b/3 functions, candidates.json, winner.json
✅ split.py — math.floor rounding, 3 splits (67/33, 80/20, 93/7), docstring updated
✅ data_loader.py — manifest mode, val_ prefix removed
✅ augment.py — parametric aug_params mode, elastic/blur/CLAHE added
✅ train.py — aug_params passed through, deterministic seeding, save_checkpoint flag
✅ metrics.py — compute_epoch_metrics + compute_all_metrics with HD95
✅ analysis_config.json — deeplabv3+ replacing attention_unet, wave2 block, 3 splits
✅ utils/version.py — bumped to 5.1.0, codename v5_analysis
✅ FEAT-07: sbatch + --config non-interactive mode (completed S5)

---

## Resolved — S7 (morphometrics refactor + pipeline hardening)

✅ TIFF provenance metadata — segmented images embed model/version/CLAHE/watershed via tifffile
✅ morphometrics.py — gratio_area → gratio_equiv_diam, gratio_axes → gratio_mean_axes
✅ morphometrics.py — µm-primary output, raw pixel DataFrame separate (Sheet 5)
✅ save_morphometrics() — 5-sheet openpyxl output (Summary, Axon, Myelin+G-ratio, Fiber, Raw px)
✅ save_distributions() — multi-sheet output (Summary, Granular, Mid, Coarse)
✅ analyze_nerve.py — moved from batch_axon/ to morphometrics/, primary_gratio_method config
✅ batch_axon/__main__.py — Provenance sheet, updated import from morphometrics.analyze_nerve
✅ segment/segment.py — tifffile write, arch map in load_model, import cleanup, datetime fix
✅ segment/__main__.py — meta passed to segment_dir, stray typer import removed
✅ utils/helpers.py — resolve_scan PermissionError fix, nerve-level direct construction
✅ config.json — primary_gratio_method, three-tier morphometrics_bins, weight_decay confirmed
✅ requirements.txt — matplotlib==3.10.9 added
✅ save_checkpoint: False — all Wave 1/2a/2b/3 analysis jobs
✅ CRIT-02: Resolved — g-ratio physiologically plausible in morphometrics outputs
✅ DOC-02: Resolved — gratio_equiv_diam / gratio_mean_axes renamed
✅ FINETUNE-01: train/finetune.py built, 3 freeze strategies (full/encoder/none)

---

## Open item count
- Critical: 0
- Analysis pipeline: 9 (ANAL-01 through ANAL-09)
- Morphometrics: 3 (MORPH-01 through MORPH-03)
- Docs: 2 (DOC-01, DOC-03)
- Post-analysis/Paper 2: ~11
- V2: 10