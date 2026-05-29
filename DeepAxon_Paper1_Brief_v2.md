# DeepAxon Paper 1 — Analysis Pipeline: Coding Brief
**Version:** 2.0 | **Branch:** v5_analysis | **Updated:** May 2026

---

## People
- **Marc Mazur** — MD, MSc, Research Fellow & PhD Student, VCU Orthopaedic Surgery. Primary developer, corresponding author.
- **Kush Savsani** — Co-programmer
- **Preetam Ghosh, PhD** — Technical supervisor (VCU CS)
- **Geetanjali Bendale** — Daily lab supervisor
- **J.H. Coert, MD PhD** — Utrecht Medical Center, Division of Plastic Surgery
- **Jonathan Isaacs, MD** — PI, last author

**Author order:** Marc Mazur, Kush Savsani, Preetam Ghosh, Geetanjali Bendale, J.H. Coert, Jonathan Isaacs

---

## Cluster & Environment
- Cluster: `mazurm@athena.hprc.vcu.edu`, H100 80GB (gpu-h100, athena531–534)
- venv: `~/deepaxon/venv` — **NEVER DELETE**
- Repo: `kushsavsani/deepaxon-legacy`, branch **`v5_analysis`** (cut from v5_pytorch)
- Data: `~/rb40x_v2_hprc` (flat `images/` + `masks/`, `ctrl_` / `regen_` prefixes)
- Analysis outputs: `~/deepaxon/analysis/` (jobs/, results/, models/, logs/, aggregated/)
- Training time: ~7 min per run on H100
- Always use `sbatch` for cluster runs — never run training interactively

---

## Dataset (current state)
- 30 annotated images, 22 animals, 3 independent studies
- 9 animals contributed 2 images each; remaining contributed 1
- 1:1 phenotype ratio: 15 control (`ctrl_` prefix), 15 regenerating (`regen_` prefix)
- 40X rabbit sciatic nerve, Toluidine Blue brightfield
- **Pipeline is dataset-size agnostic** — adding images requires only a config value change
- Phenotype label tracked per image via filename prefix throughout all analyses
- Target: 40 images (10 more masks needed); interim analysis at 30 images is valid

---

## Current Production Model
- Architecture: UNet++, Encoder: ResNet34 (segmentation-models-pytorch v0.5.0)
- 3-class segmentation: background, myelin sheath, axon interior
- Loss: Weighted Dice (0.5) + CrossEntropy (0.5), class weights [3.0, 1.0, 1.0] — [bg, myelin, axon]
- Checkpointing and early stopping monitor `val_loss`
- Optimizer: AdamW, lr=0.001, weight_decay=0.01
- Learning rate scheduling: ReduceLROnPlateau, patience=15, factor=0.5

### Augmentation (production defaults)
- 6 types: H-flip, V-flip, Rotation (geometric) + Brightness, Gamma, Noise (photometric)
- New types available for Wave 2 sweep: Gaussian blur, Elastic deformation, CLAHE
- Each aug type has its own independent probability
- Photometric augmentations applied to **images only** — never to masks
- All parameters configurable via `config.json` and overridable per-job via `aug_params` in job config

---

## Split Strategy
- **Stratified random splitting** — phenotype balance (1:1 ctrl/regen) maintained within each split
- 5 fixed seeds per condition (seeds 1–5) — reproducibility controls, **never optimized**
- Same seeds used across all architectures, encoders, augmentation conditions
- Splits are phenotype-balanced approximations — effective splits at n=30:

| Target | n_val_per_class | Val total | Train total | Effective split |
|---|---|---|---|---|
| 67/33 | 5 | 10 | 20 | 67/33 ✅ exact |
| 80/20 | 3 | 6 | 24 | 80/20 ✅ exact |
| 93/7 | 1 | 2 | 28 | 93/7 ✅ exact |

All three splits used in Wave 1. 93/7 is minimum viable phenotype-balanced val set (1 ctrl + 1 regen).

---

## Metrics — Computed Every Run
**Per-epoch (lightweight):** Dice macro + per-class, IoU macro + per-class
**Post-training (full suite):** Dice, IoU, Precision, Recall (macro + per-class), HD95 via MONAI

### Per-class reporting
- 3 classes: background, myelin, axon
- Background metrics **included** — false positive axon/myelin in connective tissue directly corrupts morphometric outputs
- **Background class inflates macro Dice** — report with note in paper; checkpoint metric is val_loss not val_dice to avoid circularity
- **Myelin class is hardest and most scientifically meaningful** — most likely to differentiate architectures
- **HD95** — boundary accuracy; myelin boundary precision directly affects downstream g-ratio

---

## Architecture List (Wave 1 SW)

**4 architectures × 6 encoders = 24 combinations (segmentation-models-pytorch v0.5.0):**

```python
# Architecture comparison (encoder fixed at ResNet34)
smp.Unet(encoder_name="resnet34")           # UNet — field standard baseline
smp.UnetPlusPlus(encoder_name="resnet34")   # UNet++ — current DeepAxon
smp.MAnet(encoder_name="resnet34")          # MAnet — multi-scale attention
smp.DeepLabV3Plus(encoder_name="resnet34")  # DeepLabV3+ — ASPP context

# Encoders (within each architecture)
resnet34, resnet50,
efficientnet-b3, efficientnet-b4,
densenet121, densenet169
```

**Note:** AttentionUnet is **not available** in smp v0.5.0 and has been replaced by DeepLabV3+.

### External benchmarks (separate inference pipelines, not in Wave 1 job array)
- **ADS** — run out-of-the-box on all images, metrics computed against same ground truth masks
- **MedSAM** — zero-shot inference, no fine-tuning (bowang-lab/MedSAM)

### Class weights sweep (16 configs)
`[bg, myelin, axon]` — full sweep. Row 3 = `[3.0, 1.0, 1.0]` = current production value.

---

## Three-Wave Analysis Structure

### Wave 1 — Architecture/Encoder/Weight Sweep (aug OFF)
```
24 arch/encoder × 16 class weights × 3 splits × 5 seeds = 5,760 jobs
~34 hours wall time at 60 concurrent on 20 H100s
```

Launcher: `python wave1_launcher.py --config analysis_config.json [--dry-run]`

**Tables from Wave 1:**
- **Table 2** — Architecture comparison (encoder=resnet34, optimal weights per arch)
- **Table 3** — Encoder comparison (winning arch, optimal weights per encoder)

**Manual review gate after Wave 1:**
```bash
python aggregator.py --config analysis_config.json
# Review candidates.json and table2/table3 CSVs with Preetam
python aggregator.py --config analysis_config.json --select \
    --arch unet++ --encoder resnet34 --weights 3,1,1 \
    [--split 67,33] --note "rationale"
# Writes winner.json — required before Wave 2
```

---

### Wave 2 — Augmentation Sweep (on winning model only)

**Step 2a — OAT + matrix sweep (2,065 jobs)**
All aug types OFF except the one being swept (pure OAT design).

| Aug type | Design | Jobs (×5 seeds) |
|---|---|---|
| H-flip prob | OAT, 4 prob levels | 20 |
| V-flip prob | OAT, 4 prob levels | 20 |
| Rotation prob | OAT, 4 prob levels | 20 |
| Rotation intensity | OAT, 5 deg levels | 25 |
| Brightness | 4 prob × 7 intensity | 140 |
| Gamma | 4 prob × 7 intensity | 140 |
| Noise | 4 prob × 6 sigma | 120 |
| Gaussian blur | 4 prob × 5 sigma | 100 |
| Elastic deformation | 4 prob × 5 alpha × 5 sigma | 500 |
| CLAHE | 4 prob × 7 clip × 7 tile | 980 |
| **Total 2a** | | **2,065** |

Launcher: `python wave2_launcher.py --config analysis_config.json --step 2a`

**Manual review gate after 2a:**
```bash
python aggregator.py --config analysis_config.json --wave 2a
# Review table4_aug_sweep.csv
# Manually write analysis/aggregated/winner_aug.json with optimized params
```

**Step 2b — Aug ON validation (5 jobs)**
Aug ON (optimized params from 2a) × 5 seeds × best_split from winner.json.
Aug OFF baseline pulled from Wave 1 SW results — no redundant re-runs.

Launcher: `python wave2_launcher.py --config analysis_config.json --step 2b`

**Tables from Wave 2:**
- **Table 4** — Aug sweep matrix (best params per aug type)
- **Table 5** — Aug ON vs OFF (delta per metric, ctrl vs regen)

---

### Wave 3 — Learning Curve (fully optimized model)
```
3 dataset sizes × 3 splits × 5 seeds = 45 jobs
Fully optimized model — best arch, encoder, weights, aug params
```

Launcher: `python wave3_launcher.py --config analysis_config.json`

**Table from Wave 3:**
- **Table 1** — Learning curve (Dice ± SD vs dataset size — plateau at n=30)

---

## Table Order (paper)

| Table | Wave | Content |
|---|---|---|
| Table 1 | Wave 3 | Learning curve — dataset sufficiency |
| Table 2 | Wave 1 SW | Architecture comparison |
| Table 3 | Wave 1 SW | Encoder comparison |
| Table 4 | Wave 2a | Aug parameter sweep |
| Table 5 | Wave 2b | Aug ON vs OFF by phenotype |

---

## Aggregator Usage

```bash
python aggregator.py --config analysis_config.json           # Wave 1 — Tables 2, 3
python aggregator.py --config analysis_config.json --wave 2a # Table 4
python aggregator.py --config analysis_config.json --wave 2b # Table 5
python aggregator.py --config analysis_config.json --wave 3  # Table 1
```

Outputs to `analysis/aggregated/`. Download via scp for review in Prism/Excel.

---

## Key Files (repo root, v5_analysis branch)

| File | Purpose |
|---|---|
| `analysis_config.json` | Master config — all wave parameters, SLURM settings, paths |
| `wave1_launcher.py` | Generate and submit Wave 1 SW job array |
| `wave2_launcher.py` | Generate and submit Wave 2a/2b job arrays |
| `wave3_launcher.py` | Generate and submit Wave 3 LC job array |
| `aggregator.py` | Aggregate results, produce tables, write winner.json |
| `train/train.py` | Training loop — parametric arch/encoder/aug_params |
| `train/__main__.py` | Entry point — interactive + sbatch modes |
| `train/dataset/split.py` | Stratified phenotype-balanced split |
| `train/dataset/data_loader.py` | Manifest mode loading (no val_ prefix) |
| `train/dataset/augment.py` | Parametric aug — config mode + aug_params mode |
| `utils/metrics.py` | compute_epoch_metrics + compute_all_metrics (HD95) |

---

## Checklist Before Wave 1 Launch

- [ ] 30 masks complete, `ctrl_`/`regen_` prefixes confirmed on all images
- [ ] Fill real Athena paths into `analysis_config.json`
- [ ] `git pull` v5_analysis on Athena
- [ ] `pip install monai==1.5.2 --break-system-packages`
- [ ] Create output directories: `~/deepaxon/analysis/{jobs,results,models,logs,aggregated}/`
- [ ] Run preprocessing: generate patches on Athena
- [ ] Dry run: `python wave1_launcher.py --config analysis_config.json --dry-run`
- [ ] Submit: `python wave1_launcher.py --config analysis_config.json`

---

## Circularity — Resolved
- Split and learning curve entangled → Wave 3 LC uses all 3 splits with optimized model
- Aug parameters require trained model → Wave 1 runs first (aug OFF), aug sweep on winner only
- Architecture unknown before aug sweep → Wave 1 establishes winner before Wave 2
- No circular dependencies anywhere in the plan

---

## Known Limitations (state in paper)
- Dataset limited to 40X rabbit, 30 images current (40 target)
- AttentionUnet excluded — not available in smp v0.5.0; DeepLabV3+ substituted
- SegFormer not included in primary analysis — flagged as future direction
- Aug parameters optimized on winning architecture only
- Control/regen ratio fixed at 1:1 — ratio analysis deferred to Paper 2
- 90/10 split deferred until dataset reaches 40 images

---

## Deferred to Paper 2 (Protocol Paper)
- Hold-out test set (4 images, 2 ctrl, 2 regen — never seen during development)
- Control/regen training ratio analysis
- Watershed sensitivity analysis
- Inter-rater variability on masks
- Bland-Altman morphometric comparisons (DeepAxon vs ADS vs manual)
- BatchAxon demonstration
- Multi-species and multi-magnification generalization
- Open-source release (GitHub, MIT license)

---

## Code Style
- Targeted patches — exact before/after diffs, never full file rewrites
- Mark changed lines `# ← CHANGED` / `# ← NEW`
- Flag out-of-scope issues — never fix silently
- All hyperparameters in JSON — never hardcoded
- Pipeline fully reproducible from config file alone
