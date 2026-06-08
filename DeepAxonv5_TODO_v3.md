# DeepAxon — To-Do List
**Updated:** June 8, 2026 | **Branch:** main (formerly v5_analysis)
**Previous sessions:** S1 Audit, S2 Bug fixing, S3 PyTorch migration, S4 Phenotype split, S5 Training overhaul, S6 Analysis pipeline, S7 Morphometrics refactor + pipeline hardening, S8 Wave 1 launch + AMP investigation + ADS installation
**All changes through S8 committed and pushed to main.**

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

## Section 2 — Analysis Pipeline

**[ANAL-01] Fill real Athena paths into analysis_config.json** ✅
All `/path/to/` placeholders replaced. Lustre paths confirmed on Athena.

**[ANAL-02] Install monai on Athena venv** ✅
Installed via pip as part of requirements.txt on Athena venv.

**[ANAL-03] Create output directories on Athena** ✅
`~/deepaxon/analysis/{jobs,results,models,logs,aggregated}/` created.

**[ANAL-04] Preprocessing run — patches generated** ✅
1,890 patches (63 per image × 30 images) generated locally and uploaded to Athena.
Patches at: `~/rb40x_val_hprc/images/cropped/patches/`

**[ANAL-05] Dry run test** ✅
Verified job configs generate correctly. Both fast and deeplab sbatch scripts validated.

**[ANAL-06] ADS external benchmark** 🟠
In progress. ADS v5.5.0 installed in separate `ads_env` conda environment locally.
Generalist model downloaded manually (SSL issue on VCU network).
Inference confirmed working on ctrl_img_1.png — output is 0/127/255 (myelin=127, needs remapping to 128).
Output dimensions match ground truth masks (1024×1440) ✅
Next: run generalist + dedicated-BF on all 30 images, write comparison script.
**MedSAM removed from scope per Preetam Ghosh.**

**[ANAL-07] aggregator.py Wave 2b — pull aug OFF from Wave 1 SW** 🟡
`build_table5()` matches by arch/encoder/weights/split/seed.
Verify matching logic after Wave 1 results are in.

**[ANAL-08] winner_aug.json format** 📄
Document expected format after Wave 2a review.
Add example to aggregator.py --wave 2a terminal output.

**[ANAL-09] Production model retrain after Wave 3** 🟠
No .pt files saved during analysis pipeline (save_checkpoint=False throughout).
After Wave 3 completes and winner is selected, retrain once interactively.
Names output `rb40x_v2.pt` — production model for Paper 1.

**[ANAL-10] Wave 1 currently running** 🟠
- Fast array (unet++, unet, manet): 4,320 jobs, 45 min wall time
- DeepLab array (deeplabv3+): 1,440 jobs, 6 hour wall time
- batch_size=128, no AMP, no concurrency cap in sbatch
- SLURM QOS limit: gres/gpu=9 concurrent (raised from 6 by Carlisle)
- Skip logic in train/__main__.py — already-completed jobs exit in seconds
- ~1,227 results already completed from previous run, being skipped on resubmission
- Estimated completion: ~28 hours from resubmission

---

## Section 3 — Known Findings (Wave 1)

**[FIND-01] unet++/efficientnet-b4 structural incompatibility** 🟠
unet++/efficientnet-b4 consistently collapses to background-only predictions
regardless of batch size, AMP, class weights, or split ratio.
Confirmed structural incompatibility — not a tuning issue.
Results flagged as degenerate by aggregator. **Reportable finding for paper.**

**[FIND-02] AMP (mixed precision) instability** ✅ Investigated and rejected
AMP (float16) tested at batch_size=128 and batch_size=256.
Caused training collapse on multiple arch/encoder combinations independent of batch size.
Root cause: gradient underflow in float16 for minority classes (myelin, axon).
Decision: AMP excluded from Wave 1. Not revisited without strong new evidence.
**Reportable in methods: AMP tested but excluded due to training instability.**

**[FIND-03] OOM at batch_size=256** ✅ Resolved
11 arch/encoder combinations exceeded H100 80GB at batch_size=256.
Fix: batch_size=128 for all jobs. Consistent sweep design maintained.
Failed combinations documented:
- unet++ with all encoders except resnet34
- deeplabv3+ with efficientnet-b3/b4, densenet121/169
- all architectures with efficientnet-b4

---

## Section 4 — Morphometrics

**[MORPH-01] File-level skip logic in morphometrics/__main__.py** 🟡
Currently skips whole nerve if Morphometrics/ folder exists.
Should skip individual images if {stem}_morphometrics.xlsx already exists.

**[MORPH-02] batch_axon/__main__.py — full run test** 🟡
Confirm Provenance sheet writes correctly, study workbook structure intact.

**[MORPH-03] Delete batch_axon/analyze_nerve.py** 🟡
Moved to morphometrics/analyze_nerve.py. Old file still present — delete after
confirming batch_axon/__main__.py imports correctly from new location.

---

## Section 5 — Infrastructure

**[INFRA-01] SLURM concurrency — no software cap** ✅
Removed max_concurrent from sbatch scripts. SLURM enforces its own QOS limits.
Current limit: gres/gpu=9 (normal QOS, VCU Athena).
wave1_launcher.py updated — no %N concurrency cap in array directive.

**[INFRA-02] Skip logic in train/__main__.py** ✅
Jobs check for existing result.json before training. Exits with code 0 if found.
Allows resubmission of full array without losing completed results.

**[INFRA-03] Wave 1 split into fast/deeplab sbatch scripts** ✅
wave1_launcher.py generates two separate sbatch scripts:
- wave1_sw_fast.sbatch — unet++, unet, manet (45 min wall time)
- wave1_sw_deeplab.sbatch — deeplabv3+ only (6 hour wall time)
DeepLab runs independently — results reviewed separately before inclusion.

---

## Section 6 — Docs

**[DOC-01] README — v5_analysis update** ✅
Updated: entry points, training section, analysis pipeline, development tree,
BGW convention, configuration reference.

**[DOC-02] README — remove Athena-specific references** 🟡
Replace Athena-specific cluster instructions with generic Linux/cluster instructions.
Add Windows instructions where applicable.

**[DOC-03] Annotation protocol in README** 📄
Add Training section covering minimum mask counts, phenotype balance, naming convention.

**[DOC-04] install.sh** ✅
Shell script for environment setup created. Covers pip install, PyTorch CUDA wheel,
patchify, and verification step. Cross-platform usage notes included.

---

## Section 7 — Post-Analysis / Paper 2 (do not implement mid-study)

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

## Section 8 — V2 Production Model (post-Paper 1)

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

## Resolved — S8 (Wave 1 launch + AMP investigation + ADS)

✅ ANAL-01 through ANAL-05 — all pre-launch checklist items complete
✅ Wave 1 launched — fast array (4,320) + deeplab array (1,440) running
✅ AMP investigated and rejected — training collapse confirmed, excluded from sweep
✅ unet++/efficientnet-b4 incompatibility confirmed — reportable finding
✅ batch_size reduced from 256 to 128 — consistent sweep design maintained
✅ max_concurrent removed from sbatch — SLURM enforces its own limits
✅ Skip logic added to train/__main__.py — safe resubmission of full array
✅ Wave 1 sbatch split into fast/deeplab with separate wall times
✅ ADS v5.5.0 installed locally — generalist model confirmed working
✅ ADS output format confirmed: 0/127/255 (myelin=127, remap needed)
✅ install.sh created
✅ README updated (DOC-01)
✅ analysis_config.json — all Lustre paths, batch_size=128, no concurrency cap
✅ wave1_launcher.py — no %N cap, fast/deeplab split, docstring updated
✅ HD95 fix in metrics.py — nanmean(dim=0) for MONAI 1.5.2 aggregate output
✅ hd95_myelin_axon added to result.json and RUN COMPLETE summary
✅ MedSAM removed from scope per Preetam Ghosh

---

## Open item count
- Critical: 0
- Analysis pipeline: 4 (ANAL-06 in progress, ANAL-07, ANAL-08, ANAL-09, ANAL-10 running)
- Known findings: 3 (documented, FIND-01 reportable)
- Morphometrics: 3 (MORPH-01 through MORPH-03)
- Infrastructure: 0
- Docs: 2 (DOC-02, DOC-03)
- Post-analysis/Paper 2: ~11
- V2: 10
