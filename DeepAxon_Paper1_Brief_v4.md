# DeepAxon Paper 1 — Analysis Pipeline: Coding Brief
**Version:** 4.0 | **Branch:** main (formerly v5_analysis) | **Updated:** June 13, 2026

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
- SLURM QOS limit: gres/gpu=16 concurrent (raised from 9 by Carlisle Jun 2026)
- No software concurrency cap in sbatch scripts — SLURM enforces its own limits

### Athena directory structure
```
~/deepaxon/
├── venv/                          # Python venv — NEVER DELETE
├── aggregator.py
├── wave1_launcher.py
├── wave2_launcher.py
├── wave3_launcher.py
├── analysis_config.json           # Main sweep config (unet, unet++, manet, deeplabv3+)
├── analysis_config_unet3plus.json # UNet3+ sweep config
├── analysis_config_attention_unet.json  # Attention UNet sweep config
├── config.json                    # Pipeline config (CLAHE, watershed, patch size, aug)
├── train_config.json              # Single interactive/sbatch run config — gitignored
├── train.sbatch                   # Single interactive training sbatch
├── requirements.txt
├── README.md
├── logs/                          # Root-level logs (non-analysis runs)
├── models/                        # Root-level models directory
├── train/
│   ├── train.py
│   ├── init.py
│   ├── main.py
│   ├── finetune.py
│   ├── unet3plus.py               # UNet3+ implementation
│   └── dataset/
│       ├── augment.py
│       ├── data_loader.py
│       ├── preprocess.py
│       ├── split.py
│       └── init.py
├── segment/
│   ├── segment.py
│   ├── init.py
│   └── main.py
├── morphometrics/
│   ├── morphometrics.py
│   ├── distributions.py
│   ├── analyze_nerve.py
│   ├── init.py
│   └── main.py
├── utils/
│   ├── gpu.py
│   ├── helpers.py
│   ├── logger.py
│   ├── metrics.py
│   ├── resize.py
│   ├── version.py
│   ├── class_balance.py
│   ├── init.py
│   └── main.py
├── analysis/                      # Main sweep outputs
│   ├── wave1_sw_fast.sbatch
│   ├── wave1_sw_deeplab.sbatch
│   ├── wave1_deeplab_rerun.sbatch
│   ├── jobs/sw/                   # Job configs (job_0000.json … job_5759.json)
│   ├── results/sw/                # result.json per run (sw__arch__encoder__cw__split__seed/)
│   ├── models/sw/                 # Empty — save_checkpoint=False
│   ├── logs/sw/                   # SLURM .out/.err per job
│   └── aggregated/                # candidates.json, winner.json, CSVs
├── analysis_unet3plus/            # UNet3+ sweep outputs (same structure as analysis/)
│   ├── wave1_sw_fast.sbatch
│   ├── jobs/sw/
│   ├── results/sw/
│   ├── logs/sw/
│   └── aggregated/
└── analysis_attention_unet/       # Attention UNet sweep outputs (same structure)
├── wave1_sw_fast.sbatch
├── jobs/sw/
├── results/sw/
├── logs/sw/
└── aggregated/
~/rb40x_val_hprc/                  # Dataset — images and masks
```

### Local repo structure (Windows, C:\Users\mazurm\deepaxon\)
```
deepaxon/
├── train/
│   ├── train.py
│   ├── __init__.py
│   ├── __main__.py
│   ├── finetune.py
│   ├── unet3plus.py
│   └── dataset/
│       ├── augment.py
│       ├── data_loader.py
│       ├── preprocess.py
│       ├── split.py
│       └── __init__.py
├── segment/
│   ├── segment.py
│   ├── __init__.py
│   └── __main__.py
├── morphometrics/
│   ├── morphometrics.py
│   ├── distributions.py
│   ├── analyze_nerve.py
│   ├── __init__.py
│   └── __main__.py
├── batch_axon/
│   ├── __init__.py
│   ├── __main__.py
│   └── overlay/
│       ├── __init__.py
│       └── process_overlay.py
├── utils/
│   ├── gpu.py
│   ├── helpers.py
│   ├── logger.py
│   ├── metrics.py
│   ├── resize.py
│   ├── version.py
│   ├── class_balance.py
│   ├── __init__.py
│   └── __main__.py
├── models/
│   └── inspect_model.py
├── aggregator.py
├── wave1_launcher.py
├── wave2_launcher.py
├── wave3_launcher.py
├── analysis_config.json
├── analysis_config_unet3plus.json
├── analysis_config_attention_unet.json
├── train_config.json              # gitignored
├── config.json
├── requirements.txt
├── install.sh
├── LICENSE.md
└── README.md
```

---

## Dataset (current state)
- 30 annotated images, 22 animals, 3 independent studies
- 9 animals contributed 2 images each; remaining contributed 1
- 1:1 phenotype ratio: 15 control (`ctrl_` prefix), 15 regenerating (`regen_` prefix)
- 40X rabbit sciatic nerve, Toluidine Blue brightfield, Olympus BX63
- Pixel size: 0.217 µm/px at 40X (1440px wide)
- **Pipeline is dataset-size agnostic** — adding images requires only a config value change
- Target: 40 images (10 more masks needed); interim analysis at 30 images is valid
- regen_img_11 and regen_img_12: intentional near-pure-background negatives — suppress
  false positive vessel/artifact segmentation. Lock to train in constrained splitter.

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
- epochs: 400 (early stopping fires well before limit — mean 57.1 ± 17.5, range 41–136)
- augmentation: OFF (Wave 1 only)
- AMP (mixed precision): **NOT USED** — tested and rejected due to training collapse
  - Root cause: float16 gradient underflow for minority classes (myelin, axon)
  - Do not revisit without strong new evidence
- save_checkpoint: False (all Wave 1/2/3 analysis jobs)

---

## Split Strategy
- **Stratified random splitting** — phenotype balance (1:1 ctrl/regen) maintained
- 5 fixed seeds per condition (seeds 1–5)
- Seed variance finding: seeds 1 and 3 show inflated metrics due to regen_img_11 in val
- Reporting: seeds 2/4/5 in main paper, full 5-seed variance in supplementary

| Target | n_val_per_class | Val total | Train total |
|--------|-----------------|-----------|-------------|
| 67/33  | 5               | 10        | 20          |
| 80/20  | 3               | 6         | 24          |
| 93/7   | 1               | 2         | 28          |

**Split winner (interim):** 67/33 — dice_myelin 0.751, HD95 14.6px vs 80/20 (0.735, 15.4px)
**93/7 eliminated** — HD95 SD ~2× higher, unstable at small n

---

## Metrics — Computed Every Run
**Per-epoch (lightweight):** Dice macro + per-class, IoU macro + per-class
**Post-training (full suite):** Dice, IoU, Precision, Recall, HD95 (macro + per-class)

### Primary reporting metric
- `hd95_myelin_axon` — mean of myelin and axon HD95, excludes background
- Composite score: (dice_myelin + dice_axon + hd95_norm) / 3, hd95_norm = max(0, 1 - hd95/50)

### HD95 implementation note
MONAI 1.5.2 `aggregate(reduction="none")` returns (N, C) tensor.
Fix applied in `utils/metrics.py`: `nanmean(dim=0)` before per-class indexing.

---

## Architecture List (Wave 1 SW)

**6 architectures × 6 encoders = 36 combinations across 3 separate sweeps:**

```python
# Main sweep (analysis_config.json) — 24 arch/encoder combos
smp.Unet(encoder_name=...)                                    # UNet
smp.UnetPlusPlus(encoder_name=...)                            # UNet++ — current DeepAxon
smp.MAnet(encoder_name=...)                                   # MANet — multi-scale attention
smp.DeepLabV3Plus(encoder_name=...)                           # DeepLabV3+ — ASPP (slow)

# Separate sweep (analysis_config_unet3plus.json) — 6 combos
UNet3Plus(encoder_name=...)                                   # UNet3+ — full-scale skip

# Separate sweep (analysis_config_attention_unet.json) — 6 combos
smp.Unet(encoder_name=..., decoder_attention_type='scse')     # Attention UNet

# Encoders (all sweeps)
resnet34, resnet50,
efficientnet-b3, efficientnet-b4,
densenet121, densenet169
```

### _ARCH_MAP order (train/train.py)
```python
_ARCH_MAP = {
    'unet':           smp.Unet,
    'attention_unet': smp.Unet,       # kwargs gate: decoder_attention_type='scse'
    'unet++':         smp.UnetPlusPlus,
    'unet3+':         UNet3Plus,
    'manet':          smp.MAnet,
    'deeplabv3+':     smp.DeepLabV3Plus,
}
```

### Known findings (reportable)
- **EfficientNet collapse:** B3 ~42%, B4 ~37% collapse rate, all architectures affected.
  Encoder-driven, class-weight-independent. Valid EfficientNet runs do NOT outperform
  stable encoders. Excluded from competition on stability AND performance grounds.
  Degenerate cutoffs (bimodal gap): dice_myelin < 0.5, dice_axon < 0.6.
- **DeepLabV3+ ASPP compute:** 237 sec/epoch (efficientnet-b4), ~246 sec/epoch (densenet169).
  ~42× overhead vs UNet family. Required 12hr wall time ceiling. No further ASPP architectures.
- **OOM at batch_size=256:** 11 combinations exceeded H100 80GB. Fix: batch_size=128.

### External benchmarks
- **ADS (AxonDeepSeg) v5.5.0** — generalist model, out-of-box inference only.
  Output: 0/127/255 (myelin=127 — remap to 128 before metric computation) ✅
- **MedSAM** — removed from scope per Preetam Ghosh

---

## Wave 1 — Current Status (June 13, 2026)

**Athena arrays:**
- 2614744 (main sweep): COMPLETED — 5,760 jobs, unet/unet++/manet/deeplabv3+
- 2671364 (DeepLab rerun): RUNNING — 908/1440 done, densenet tail remaining
  Results: `~/deepaxon/analysis/results/sw/`
- 2682397 (UNet3+): PENDING dependency on 2671364, 1,440 jobs, 1:30:00 wall time
  Results: `~/deepaxon/analysis_unet3plus/results/sw/`
- 2683XXX (Attention UNet): PENDING dependency on 2682397, 1,440 jobs, 1:30:00 wall time
  Results: `~/deepaxon/analysis_attention_unet/results/sw/`

**ETA:** DeepLab rerun Monday evening → UNet3+ Tuesday night → Attention UNet ~Thursday

---

## Three-Wave Analysis Structure

### Checkpoint policy — all waves
No `.pt` files saved during analysis pipeline (`save_checkpoint=False`).
After Wave 3 completes, retrain once interactively → `rb40x_v2.pt`.
Post-paper: additional seed sweep (10-20 seeds, regen_img_11 locked) to optimize
production model initialization.

---

### Wave 1 — Architecture/Encoder/Weight Sweep (aug OFF)
```
Main:          24 arch/encoder × 16 class weights × 3 splits × 5 seeds = 5,760 jobs
UNet3+:         6 arch/encoder × 16 class weights × 3 splits × 5 seeds = 1,440 jobs
Attention UNet: 6 arch/encoder × 16 class weights × 3 splits × 5 seeds = 1,440 jobs
Total: 8,640 jobs across 3 separate result directories
batch_size: 128, no AMP, save_checkpoint: False
```

**Post-Wave 1 aggregation (after ANAL-11 consolidation):**
```bash
python aggregator.py --config analysis_config_unified.json --wave 1
python aggregator.py --config analysis_config_unified.json --select \
    --arch <arch> --encoder <enc> --weights <w1,w2,w3> --note "rationale"
```

**Tables from Wave 1:**
- **Table 2** — Architecture comparison (stable leaderboard + full reference)
- **Table 3** — Encoder comparison

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
Aug OFF baseline pulled from Wave 1 SW by matching arch/encoder/weights/split/seed.
No redundant re-runs.

**Tables from Wave 2:**
- **Table 4** — Aug parameter sweep
- **Table 5** — Aug ON vs OFF

---

### Wave 3 — Learning Curve (fully optimized model)
```
67/33 wins → [6, 12, 18, 24, 30] × 1 split × 5 seeds = 25 jobs
Other split → [10, 20, 30] × 1 split × 5 seeds = 15 jobs
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
| Table 5 | Wave 2b | Aug ON vs OFF |

---

## Key Files

| File | Purpose |
|---|---|
| `analysis_config.json` | Main sweep config — unet/unet++/manet/deeplabv3+ |
| `analysis_config_unet3plus.json` | UNet3+ sweep config |
| `analysis_config_attention_unet.json` | Attention UNet sweep config |
| `wave1_launcher.py` | Generate and submit Wave 1 fast + deeplab sbatch arrays |
| `wave2_launcher.py` | Generate and submit Wave 2a/2b job arrays |
| `wave3_launcher.py` | Generate and submit Wave 3 LC job array |
| `aggregator.py` | Aggregate results, stable leaderboard, collapse diagnostics, tables, winner.json |
| `install.sh` | Environment setup script |
| `train/train.py` | Training loop — parametric arch/encoder/aug_params, attention kwarg gate |
| `train/__main__.py` | Entry point — interactive + sbatch modes, skip logic |
| `train/unet3plus.py` | UNet3+ — full-scale skip connections, smp encoder registry |
| `train/finetune.py` | Domain adaptation via fine-tuning (implemented, not validated) |
| `train/dataset/split.py` | Stratified phenotype-balanced split |
| `train/dataset/data_loader.py` | Manifest mode loading (ctrl_/regen_ prefixes) |
| `train/dataset/augment.py` | Parametric aug — config mode + aug_params mode |
| `utils/metrics.py` | compute_epoch_metrics + compute_all_metrics (HD95, hd95_myelin_axon) |
| `utils/version.py` | Version string + full env fingerprint |
| `utils/class_balance.py` | Pixel class balance reporting across segmentation directory |
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
- Multiple `__main__.py` files exist — always reference with parent folder (e.g. `train/__main__.py`)

---

## Known Limitations (state in paper)
- Dataset: 40X rabbit, n=30 (target n=40)
- AMP excluded — training instability confirmed across multiple combinations
- EfficientNet encoders excluded from competition — collapse and performance grounds
- DeepLabV3+ ~42× per-epoch overhead vs UNet family — no further ASPP architectures tested
- batch_size=128 chosen for memory compatibility — consistent across all jobs
- UNet3+ without deep supervision — single output head for sweep comparability;
  DS ablation deferred to FINETUNE-01
- ADS comparison: out-of-box inference only, no fine-tuning on our staining type
- Aug parameters optimized on winning architecture only
- 93/7 split eliminated — unstable val metrics at small n
- Seed variance: regen_img_11/12 contamination addressed by filtering to seeds 2/4/5
- Three separate result directories pre-consolidation — see ANAL-11

---

## Deferred to Paper 2
- Hold-out test set (4 images, 2 ctrl, 2 regen)
- Control/regen training ratio analysis
- Watershed sensitivity analysis
- Inter-rater variability on masks
- Bland-Altman morphometric comparisons
- Multi-species and multi-magnification generalization
- Fine-tuning sbatch support (FINETUNE-01)
- UNet3+ deep supervision ablation
- MedSAM comparison (removed from Paper 1 scope)
