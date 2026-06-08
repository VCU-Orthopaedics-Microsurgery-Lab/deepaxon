# DeepAxon Paper 1 — Analysis Pipeline: Coding Brief
**Version:** 3.0 | **Branch:** main (formerly v5_analysis) | **Updated:** June 8, 2026

---

## People
- **Marc Mazur** — MD, MSc, Research Fellow & PhD Student, VCU Orthopaedic Surgery. Primary developer, corresponding author.
- **Kush Savsani** — Co-programmer
- **Preetam Ghosh, PhD** — Technical supervisor (VCU CS)
- **Geetanjali Bendale** — Daily lab supervisor (VCU Ortho)
- **J.H. Coert, MD PhD** — PhD Supervisor, Utrecht Medical Center, Division of Plastic Surgery
- **Jonathan Isaacs, MD** — PI, last author (VCU Ortho)

**Author order:** Marc Mazur, Kush Savsani, Preetam Ghosh, Geetanjali Bendale, J.H. Coert, Jonathan Isaacs

---

## Cluster & Environment
- Cluster: `mazurm@athena.hprc.vcu.edu`, H100 80GB (gpu-h100, athena531–534)
- Home directory: `/lustre/home/mazurm/`
- venv: `~/deepaxon/venv` — **NEVER DELETE**
- Repo: `VCU-Orthopaedics-Microsurgery-Lab/deepaxon`, branch **`main`**
- Data: `~/rb40x_val_hprc/` (flat `images/` + `masks/`, `ctrl_` / `regen_` prefixes)
- Patches: `~/rb40x_val_hprc/images/cropped/patches/` (1,890 patches, pre-generated)
- Analysis outputs: `~/deepaxon/analysis/` (jobs/, results/, models/, logs/, aggregated/)
- Always use `sbatch` for cluster runs — never run training interactively on login nodes
- SLURM QOS limit: gres/gpu=9 concurrent (normal QOS, raised from 6 by Carlisle Jun 2026)
- No software concurrency cap in sbatch scripts — SLURM enforces its own limits

---

## Dataset (current state)
- 30 annotated images, 22 animals, 3 independent studies
- 9 animals contributed 2 images each; remaining contributed 1
- 1:1 phenotype ratio: 15 control (`ctrl_` prefix), 15 regenerating (`regen_` prefix)
- 40X rabbit sciatic nerve, Toluidine Blue brightfield, Olympus BX63
- Pixel size: 0.217 µm/px at 40X (1440px wide)
- **Pipeline is dataset-size agnostic** — adding images requires only a config value change
- Target: 40 images (10 more masks needed); interim analysis at 30 images is valid

---

## Current Production Model
- Name: `rb40x_v1`
- Architecture: UNet++, Encoder: ResNet34 (segmentation-models-pytorch v0.5.0)
- 3-class segmentation: background (0), myelin (128), axon (255)
- Loss: Weighted Dice (0.5) + CrossEntropy (0.5), class weights [3.0, 1.0, 1.0]
- Checkpointing and early stopping monitor `val_loss`
- Optimizer: AdamW, lr=0.001, weight_decay=0.01
- LR scheduling: ReduceLROnPlateau, patience=15, factor=0.5
- Early stopping: patience=40, min_delta=0.001

---

## Training Parameters (Wave 1)
- batch_size: 128
- epochs: 400 (early stopping fires well before limit — typically 50–100 epochs)
- augmentation: OFF (Wave 1 only)
- AMP (mixed precision): **NOT USED** — tested and rejected due to training collapse
  - Collapsed on multiple arch/encoder combinations at both bs=128 and bs=256
  - Root cause: float16 gradient underflow for minority classes (myelin, axon)
  - Do not revisit without strong new evidence
- save_checkpoint: False (all Wave 1/2/3 analysis jobs)

---

## Split Strategy
- **Stratified random splitting** — phenotype balance (1:1 ctrl/regen) maintained
- 5 fixed seeds per condition (seeds 1–5) — reproducibility controls, never optimized
- Same seeds used across all architectures, encoders, augmentation conditions

| Target | n_val_per_class | Val total | Train total |
|--------|-----------------|-----------|-------------|
| 67/33  | 5               | 10        | 20          |
| 80/20  | 3               | 6         | 24          |
| 93/7   | 1               | 2         | 28          |

---

## Metrics — Computed Every Run
**Per-epoch (lightweight):** Dice macro + per-class, IoU macro + per-class
**Post-training (full suite):** Dice, IoU, Precision, Recall, HD95 (macro + per-class)

### Primary reporting metric
- `hd95_myelin_axon` — mean of myelin and axon HD95, excludes background
- Background metrics included but noted as inflating macro averages
- Myelin class is hardest and most scientifically meaningful

### HD95 implementation note
MONAI 1.5.2 `aggregate(reduction="none")` returns (N, C) tensor.
Fix applied in `utils/metrics.py`: `nanmean(dim=0)` before per-class indexing.

---

## Architecture List (Wave 1 SW)

**4 architectures × 6 encoders = 24 combinations:**

```python
smp.Unet(encoder_name=...)           # UNet — field standard baseline
smp.UnetPlusPlus(encoder_name=...)   # UNet++ — current DeepAxon
smp.MAnet(encoder_name=...)          # MAnet — multi-scale attention
smp.DeepLabV3Plus(encoder_name=...)  # DeepLabV3+ — ASPP context (slow, separate sbatch)

# Encoders
resnet34, resnet50,
efficientnet-b3, efficientnet-b4,
densenet121, densenet169
```

### Known incompatibilities (reportable findings)
- **unet++/efficientnet-b4**: consistently collapses to background-only predictions
  regardless of batch size, AMP, class weights, or split. Structural incompatibility.
- **OOM at batch_size=256**: 11 arch/encoder combinations exceeded H100 80GB.
  Fix: batch_size=128 for all jobs.

### External benchmarks
- **ADS (AxonDeepSeg)** — run out-of-the-box, both generalist and dedicated-BF models
  Applied without fine-tuning, consistent with standard laboratory deployment.
  Output: 0/127/255 (myelin=127 — remap to 128 before metric computation)
  Dimensions match ground truth (1024×1440) ✅
- **MedSAM** — removed from scope per Preetam Ghosh

---

## Wave 1 — Current Status (June 8, 2026)

**Running on Athena:**
- Fast array ID: 2579129 (unet++, unet, manet — 4,320 jobs, 45 min wall time)
- DeepLab array ID: 2579130 (deeplabv3+ — 1,440 jobs, 6 hour wall time)
- ~1,227 results already completed from previous run, being skipped via skip logic
- Estimated completion: ~28 hours at 9 concurrent

**Key infrastructure changes made:**
- Skip logic in `train/__main__.py` — checks for existing result.json before training
- No %N concurrency cap in sbatch array directive
- wave1_launcher.py generates two separate sbatch scripts (fast + deeplab)

---

## Three-Wave Analysis Structure

### Checkpoint policy — all waves
No `.pt` files saved during analysis pipeline (`save_checkpoint=False`).
Results fully captured in `result.json`. Any model exactly reproducible from
`result.json` + `analysis_config.json` + seed (deterministic seeding confirmed).

After Wave 3 completes, retrain once interactively → production model `rb40x_v2.pt`.

---

### Wave 1 — Architecture/Encoder/Weight Sweep (aug OFF)
```
24 arch/encoder × 16 class weights × 3 splits × 5 seeds = 5,760 jobs
Fast (unet++, unet, manet): 4,320 jobs — 45 min wall time
DeepLab (deeplabv3+):       1,440 jobs — 6 hour wall time
batch_size: 128, no AMP
save_checkpoint: False
```

**Tables from Wave 1:**
- **Table 2** — Architecture comparison
- **Table 3** — Encoder comparison

**Manual review gate after Wave 1:**
```bash
python aggregator.py --config analysis_config.json
python aggregator.py --config analysis_config.json --select \
    --arch unet++ --encoder resnet34 --weights 3,1,1 --note "rationale"
# Writes winner.json — required before Wave 2
```

---

### Wave 2 — Augmentation Sweep (on winning model only)

**Step 2a — OAT + matrix sweep (2,265 jobs, save_checkpoint: False)**

| Aug type | Design | Jobs (×5 seeds) |
|---|---|---|
| H-flip prob | OAT, 4 levels | 20 |
| V-flip prob | OAT, 4 levels | 20 |
| Rotation prob | OAT, 4 levels | 20 |
| Rotation intensity | OAT, 5 levels | 25 |
| Brightness | 4×7 matrix | 140 |
| Gamma | 4×7 matrix | 140 |
| Noise | 4×6 matrix | 120 |
| Gaussian blur | 4×5 matrix | 100 |
| Elastic deformation | 4×5×5 matrix | 500 |
| CLAHE | 4×7×7 matrix | 980 |
| Contrast stretch | 4×5 matrix | 100 |
| Random erase | 4×5 matrix | 100 |
| **Total 2a** | | **2,265** |

**Step 2b — Aug ON validation (5 jobs)**
Aug OFF baseline pulled from Wave 1 SW — no redundant re-runs.

**Tables from Wave 2:**
- **Table 4** — Aug sweep matrix
- **Table 5** — Aug ON vs OFF by phenotype

---

### Wave 3 — Learning Curve (fully optimized model)
```
3 dataset sizes × 3 splits × 5 seeds = 45 jobs
```

**Table from Wave 3:**
- **Table 1** — Learning curve (dataset sufficiency)

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

## Key Files

| File | Purpose |
|---|---|
| `analysis_config.json` | Master config — all wave parameters, SLURM settings, Lustre paths |
| `wave1_launcher.py` | Generate and submit Wave 1 fast + deeplab sbatch arrays |
| `wave2_launcher.py` | Generate and submit Wave 2a/2b job arrays |
| `wave3_launcher.py` | Generate and submit Wave 3 LC job array |
| `aggregator.py` | Aggregate results, produce tables, write winner.json |
| `install.sh` | Environment setup script (pip, PyTorch CUDA, patchify, verify) |
| `train/train.py` | Training loop — parametric arch/encoder/aug_params, deterministic seeding |
| `train/__main__.py` | Entry point — interactive + sbatch modes, skip logic |
| `train/finetune.py` | Domain adaptation via fine-tuning |
| `train/dataset/split.py` | Stratified phenotype-balanced split |
| `train/dataset/data_loader.py` | Manifest mode loading (ctrl_/regen_ prefixes) |
| `train/dataset/augment.py` | Parametric aug — config mode + aug_params mode |
| `utils/metrics.py` | compute_epoch_metrics + compute_all_metrics (HD95, hd95_myelin_axon) |
| `utils/version.py` | Version string + full env fingerprint |
| `segment/segment.py` | Inference, Hann blending, BGW output, TIFF provenance |
| `morphometrics/morphometrics.py` | Watershed, matching, quality filters, 5-sheet xlsx |
| `morphometrics/distributions.py` | Three-tier binning, multi-sheet _binned.xlsx |

---

## Code Style (strict)
- Targeted patches — exact before/after diffs, never full file rewrites
- Flag out-of-scope issues — never fix silently
- Config file changes → diff only, never full rewrites
- Uncertain values → flag with `YOUR_VALUE_HERE`, never guess
- All hyperparameters in JSON — never hardcoded
- No software concurrency limits inserted into sbatch scripts
- Pipeline fully reproducible from config file alone

---

## Known Limitations (state in paper)
- Dataset: 40X rabbit, n=30 (target n=40)
- AMP excluded — training instability confirmed across multiple combinations
- unet++/efficientnet-b4 structural incompatibility — reportable finding
- batch_size=128 chosen for memory compatibility — consistent across all jobs
- DeepLabV3+ runs ~30× slower per epoch than other architectures
- ADS comparison: out-of-box inference only, no fine-tuning on our staining type
- AttentionUnet excluded — not in smp v0.5.0; DeepLabV3+ substituted
- Aug parameters optimized on winning architecture only
- 93/7 split is minimum viable phenotype-balanced val set

---

## Deferred to Paper 2
- Hold-out test set (4 images, 2 ctrl, 2 regen)
- Control/regen training ratio analysis
- Watershed sensitivity analysis
- Inter-rater variability on masks
- Bland-Altman morphometric comparisons
- Multi-species and multi-magnification generalization
- Fine-tuning sbatch support (FINETUNE-01)
- MedSAM comparison (removed from Paper 1 scope)