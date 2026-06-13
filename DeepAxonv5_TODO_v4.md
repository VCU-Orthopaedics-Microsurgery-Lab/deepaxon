# DeepAxon — To-Do List
**Updated:** June 13, 2026 | **Branch:** main (formerly v5_analysis)
**Previous sessions:** S1 Audit, S2 Bug fixing, S3 PyTorch migration, S4 Phenotype split, S5 Training overhaul, S6 Analysis pipeline, S7 Morphometrics refactor + pipeline hardening, S8 Wave 1 launch + AMP investigation + ADS installation, S9 Wave 1 analysis + architecture expansion
**All changes through S8 committed and pushed to main. S9 changes pending commit.**

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

**[ANAL-02] Install monai on Athena venv** ✅

**[ANAL-03] Create output directories on Athena** ✅
`~/deepaxon/analysis/{jobs,results,models,logs,aggregated}/` created.
Additional per-arch directories created: `analysis_unet3plus/`, `analysis_attention_unet/`

**[ANAL-04] Preprocessing run — patches generated** ✅
1,890 patches (63 per image × 30 images). Patches at: `~/rb40x_val_hprc/images/cropped/patches/`

**[ANAL-05] Dry run test** ✅

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
Post-paper seed sweep (10-20 seeds, regen_img_11 locked to train) to select best
production model initialization. Names output `rb40x_v2.pt`.

**[ANAL-10] Wave 1 sweep — in progress** 🟠
Current state (June 13, 2026):
- Main sweep (2614744): COMPLETED — unet, unet++, manet, deeplabv3+ (5,760 jobs)
- DeepLab rerun (2671364): RUNNING — 908/1440 done, densenet121/169 tail remaining
  - Resubmitted with 12hr wall time after original 6hr ceiling caused timeouts
  - Results in: `~/deepaxon/analysis/results/sw/`
  - sbatch files in: `~/deepaxon/analysis/`
- UNet3+ sweep (2682397): PENDING — dependency on 2671364
  - 1,440 jobs, 1:30:00 wall time
  - Results will land in: `~/deepaxon/analysis_unet3plus/results/sw/`
- Attention UNet sweep (2683XXX): PENDING — dependency on 2682397
  - 1,440 jobs, 1:30:00 wall time
  - Results will land in: `~/deepaxon/analysis_attention_unet/results/sw/`
- SLURM QOS: gres/gpu=16 concurrent (raised from 9 by Carlisle Jun 2026)
- ETA: DeepLab rerun Monday evening, UNet3+ Tuesday night, Attention UNet ~Thursday

**[ANAL-11] Post-Wave 1 results consolidation** 🟠
After all three sweeps complete, consolidate into single unified analysis directory.
Steps:
1. Download all results from Athena:
   - `~/deepaxon/analysis/results/sw/` (main sweep)
   - `~/deepaxon/analysis_unet3plus/results/sw/` (unet3+)
   - `~/deepaxon/analysis_attention_unet/results/sw/` (attention_unet)
2. Verify 100% result coverage before deleting anything on Athena
3. Merge into single flat results directory locally
4. Rebuild unified `analysis_config.json` with all 8 architectures
5. Strip per-arch configs of duplicated wave2/wave3 boilerplate
6. Clean Athena — upload unified config and merged results
7. Verify aggregator runs cleanly against unified directory
**Do not act until all three sweeps complete.**

**[ANAL-12] Seed variance — Preetam decision** 🟠
Decision received June 12, 2026:
- Main paper: seeds 2/4/5 filter (Option 1) + constrained splitter validation (Option 2)
- Supplementary: full 5-seed variance with contamination shown as limitation
- Scope of constrained splitter rerun: winner combo only (~50 jobs) — pending Preetam clarification
- regen_img_11 and regen_img_12 are intentional near-pure-background negatives — lock to train
- Post-paper: seed sweep (10-20 seeds, regen_img_11 locked) to optimize production model

---

## Section 3 — Known Findings (Wave 1)

**[FIND-01] EfficientNet encoder collapse — all architectures** 🟠
efficientnet-b3: ~42% collapse rate, efficientnet-b4: ~37% collapse rate.
Collapse is encoder-driven and class-weight-independent — all architectures affected equally.
Valid efficientnet runs do NOT outperform stable encoders on any metric (myelin delta=-0.007,
HD95 delta=-1.1px vs resnet/densenet). Excluded from competition on both stability AND
performance grounds. **Reportable finding for paper.**
Degenerate cutoffs (data-derived, bimodal gap): dice_myelin < 0.5, dice_axon < 0.6.

**[FIND-02] AMP (mixed precision) instability** ✅ Investigated and rejected
Root cause: float16 gradient underflow for minority classes (myelin, axon).
Decision: AMP excluded from Wave 1. **Reportable in methods.**

**[FIND-03] OOM at batch_size=256** ✅ Resolved
Fix: batch_size=128 for all jobs. Consistent sweep design maintained.

**[FIND-04] DeepLabV3+ ASPP compute cost** 🟠
237 sec/epoch observed for deeplabv3+/efficientnet-b4 (job 820, empirically confirmed).
deeplabv3+/densenet estimated 239-246 sec/epoch vs 5-10 sec/epoch for skip-connection archs.
~42× per-epoch overhead vs UNet family. Caused wall time failures at 6hr ceiling — rerun
required at 12hr. **Reportable in methods as computational cost finding.**
Decision: No further ASPP-family architectures (PSPNet excluded) due to compute cost.

**[FIND-05] Split comparison — interim finding** 🟠
67/33 wins on dice_myelin (0.751) and HD95 (14.6px) vs 80/20 (0.735, 15.4px).
93/7 eliminated — HD95 SD ~2× higher than 67/33 and 80/20, unstable at small n.
Confirmed with stable encoders only (EfficientNet excluded).

**[FIND-06] Seed variance / regen_img_11 contamination** 🟠
Seeds 1 and 3 show inflated val metrics due to regen_img_11 appearing in val set.
regen_img_11 and regen_img_12 are intentional near-pure-background negatives — when in val,
they artificially inflate background dice. Addressed by filtering to seeds 2/4/5.
See ANAL-12 for Preetam's decision.

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

**[INFRA-01] SLURM concurrency** ✅
Current limit: gres/gpu=16 (raised from 9 by Carlisle Jun 2026).
No software concurrency cap in sbatch scripts — SLURM enforces its own limits.

**[INFRA-02] Skip logic in train/__main__.py** ✅
Jobs check for existing result.json before training. Exits with code 0 if found.

**[INFRA-03] Wave 1 split into fast/deeplab sbatch scripts** ✅
wave1_launcher.py generates two separate sbatch scripts:
- wave1_sw_fast.sbatch — unet, unet++, manet (45 min wall time)
- wave1_sw_deeplab.sbatch — deeplabv3+ only (12hr wall time for rerun)

**[INFRA-04] UNet3+ implementation** ✅
`train/unet3plus.py` — full-scale skip connections, smp encoder registry compatible.
No deep supervision (single output head) — identical training conditions to other archs.
Deep supervision deferred to post-sweep ablation (FINETUNE-01 scope).
All 6 encoders tested and passing: resnet34/50, densenet121/169, efficientnet-b3/b4.
Separate sweep config: `analysis_config_unet3plus.json`
Separate results directory: `~/deepaxon/analysis_unet3plus/`

**[INFRA-05] Attention UNet implementation** ✅
`smp.Unet(decoder_attention_type='scse')` — SCSE attention in decoder blocks.
Handled in `train/train.py` `build_model()` via kwargs gate on arch == 'attention_unet'.
No new Python file required — parameterized variant of smp.Unet.
Separate sweep config: `analysis_config_attention_unet.json`
Separate results directory: `~/deepaxon/analysis_attention_unet/`

**[INFRA-06] SLURM dependency chaining** ✅
UNet3+ array depends on DeepLab rerun: `--dependency=afterok:2671364`
Attention UNet array depends on UNet3+: `--dependency=afterok:2682397`
Ensures sequential execution without manual intervention.

**[INFRA-07] wave1_launcher.py — empty DeepLab array guard** 🟡
When no DeepLab jobs exist (e.g. unet3+/attention_unet configs), launcher generates
an empty DeepLab sbatch and attempts submission, causing SLURM error.
Fix: add guard to skip sbatch submission when n_deeplab_jobs == 0.

**[INFRA-08] Per-phenotype val metric logging in train/train.py** 🟠
Current result.json stores only aggregate val metrics — no ctrl/regen breakdown.
val_stems field identifies which images were in val per run but per-image metrics
are not stored so phenotype decomposition is not possible post-hoc from sweep results.
Fix: split val images by ctrl_/regen_ prefix during val evaluation, compute metrics
separately, write to result.json as dice_myelin_ctrl, dice_myelin_regen etc.
**Must be implemented before Wave 2a launch.** Wave 1 sweep unaffected — already complete.
Constrained splitter rerun (~50 jobs) should also use updated train.py.
Per-phenotype breakdown is a Paper 1 primary result — model must be validated
separately on control and regenerating phenotype.

---

## Section 6 — Docs

**[DOC-01] README — v5_analysis update** ✅

**[DOC-02] README — remove Athena-specific references** 🟡
Replace Athena-specific cluster instructions with generic Linux/cluster instructions.
Add Windows instructions where applicable.

**[DOC-03] Annotation protocol in README** 📄
Add Training section covering minimum mask counts, phenotype balance, naming convention.

**[DOC-04] install.sh** ✅

**[DOC-05] Update Paper1 Brief and TODO to reflect S9 changes** 🟡
Architecture list, Athena directory structure, SLURM QOS, findings, local repo structure.
(This document is that update.)

---

## Section 7 — Post-Analysis / Paper 2 (do not implement mid-study)

- STR-05: Extract select_model() to utils/helpers.py
- STR-06: Extract Excel helpers to batch_axon/excel_utils.py
- STR-09: Add evaluate/ module
- STR-10: Add models/model_registry.json
- STR-11: Archive rb_40x_v1_256.keras
- FEAT-21: visualize.py QC overlays
- FEAT-23: Boundary loss
- CRIT-08: QuPath/GeoJSON workflow
- GEN-01: 100X pixel size calibration
- GEN-04: Multi-GPU support
- FINETUNE-01: Fine-tuning sbatch support + UNet3+ deep supervision ablation

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

## Resolved — S9 (Wave 1 analysis + architecture expansion)

✅ EfficientNet collapse confirmed as encoder-driven, class-weight-independent finding
✅ Degenerate cutoffs derived from bimodal distribution: dice_myelin<0.5, dice_axon<0.6
✅ 93/7 split eliminated — HD95 SD ~2× higher than other splits
✅ 67/33 wins on dice_myelin and HD95 vs 80/20
✅ DeepLab wall time finding: 237 sec/epoch (efficientnet-b4), ~42× overhead vs UNet family
✅ DeepLab rerun submitted with 12hr wall time (2671364)
✅ Missing index identification script written and executed
✅ aggregator.py updated: auto-derive expected count from config, detect_incomplete_archs takes cfg,
   EfficientNet section gated on eff_present, stable leaderboard (Finding 3) added
✅ analysis_config.json: time_deeplab updated to 12:00:00, concurrency note updated to 16,
   wave3 block corrected, step2b note corrected (67/33)
✅ wave2_launcher.py: Step 2b job count corrected to 5 in docstring
✅ UNet3+ implemented in train/unet3plus.py — all 6 encoders passing
✅ train/train.py: _ARCH_MAP updated (unet3+, attention_unet), build_model() updated
   with attention kwarg gate, docstring updated
✅ analysis_config_unet3plus.json created
✅ analysis_config_attention_unet.json created
✅ UNet3+ sweep queued (2682397) with dependency on DeepLab rerun
✅ Attention UNet sweep queued with dependency on UNet3+
✅ INFRA-07 identified: empty DeepLab sbatch submission bug in wave1_launcher.py
✅ Seed variance finding documented (FIND-06), Preetam decision received (ANAL-12)
✅ Post-paper seed sweep plan locked: 10-20 seeds, regen_img_11 locked to train
✅ Paper 2 supplementary table decision locked: use sweep means from Paper 1 (seeds 2/4/5)
✅ MIT license updated: dual copyright (Kush 2024, Marc/VCU 2026)
✅ SLURM QOS confirmed at gres/gpu=16

---

## Open item count
- Critical: 0
- Analysis pipeline: 5 (ANAL-06, ANAL-07, ANAL-08, ANAL-09, ANAL-10 running, ANAL-11, ANAL-12)
- Known findings: 4 active (FIND-01, FIND-04, FIND-05, FIND-06)
- Morphometrics: 3 (MORPH-01 through MORPH-03)
- Infrastructure: 1 (INFRA-07)
- Docs: 2 (DOC-02, DOC-03)
- Post-analysis/Paper 2: ~11
- V2: 10
