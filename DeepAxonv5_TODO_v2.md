# DeepAxon — To-Do List
**Updated:** May 2026 | **Branch:** v5_analysis
**Previous sessions:** S1 Audit, S2 Bug fixing, S3 PyTorch migration, S4 Phenotype split, S5 Training overhaul, S6 Analysis pipeline
**All changes through S6 committed and pushed to v5_analysis.**

---

## Priority legend
- 🔴 Critical — verification required before results can be trusted
- 🟠 High — significant issue or planned breaking change
- 🟡 Medium — improvement or cleanup
- 📄 Docs — documentation task
- ✅ Resolved

---

## Section 1 — Critical

**[CRIT-02] BGW Color Mapping — Partial**
Practical verification complete (segmentation visually correct).
Remaining: run morphometrics on a known sample, confirm g-ratio ~0.6–0.7 for healthy control nerve.
Not blocking Wave 1 training. Blocking morphometric result trust.

**[CRIT-08] ROI Coordinate Space — Pending Workflow Change**
Workflow changing to QuPath + GeoJSON. Address before NT Validation Study 5 processing.
Not blocking analysis pipeline.

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

---

## Section 3 — Docs

**[DOC-01] README — v5_analysis update** 📄
Update training section to reflect:
- `ctrl_`/`regen_` prefixes instead of `val_` prefix
- sbatch mode (`python -m train --config`) instead of srun interactive
- v5_analysis branch
- New analysis pipeline entry points

**[DOC-02] G-ratio column names** 📄
Currently `gratio_area` and `gratio_axes`. Consider `gratio_equiv_diam` / `gratio_mean_axes` before publication. Low priority.

**[DOC-03] Annotation protocol in README** 📄
Add Training section covering minimum mask counts, phenotype balance, naming convention.

---

## Section 4 — Post-Analysis / Paper 2 (do not implement mid-study)

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

---

## Section 5 — V2 Production Model (post-Paper 1)

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
✅ train.py — aug_params passed through, log lines updated
✅ metrics.py — compute_epoch_metrics + compute_all_metrics with HD95
✅ analysis_config.json — deeplabv3+ replacing attention_unet, wave2 block, 3 splits
✅ utils/version.py — bumped to 5.1.0, codename v5_analysis
✅ FEAT-07: sbatch + --config non-interactive mode (completed S5)

---

## Open item count
- Critical: 2 (CRIT-02 partial, CRIT-08)
- Analysis pipeline: 8 (ANAL-01 through ANAL-08)
- Docs: 3 (DOC-01 through DOC-03)
- Post-analysis/Paper 2: ~10
- V2: 10
