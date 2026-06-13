# DeepAxon

Automated nerve cross-section segmentation and morphometric analysis using deep learning.

## Entry Points

```bash
# ── Core pipeline ──────────────────────────────────────────────────────────
python -m segment        # Segment a study folder of nerve images
python -m morphometrics  # Run per-image morphometric analysis
python -m batch_axon     # Compile study-level morphometric summary

# ── Training ───────────────────────────────────────────────────────────────
python -m train                   # Train a new segmentation model (interactive)
python -m train --config FILE     # Train non-interactively (sbatch mode)
python -m train.finetune          # Fine-tune an existing model on new images

# ── Analysis pipeline (main branch, Athena cluster) ────────────────────────
python wave1_launcher.py --config analysis_config.json [--dry-run]
python wave2_launcher.py --config analysis_config.json --step 2a [--dry-run]
python wave2_launcher.py --config analysis_config.json --step 2b [--dry-run]
python wave3_launcher.py --config analysis_config.json [--dry-run]
python aggregator.py     --config analysis_config.json [--wave 1/2a/2b/3]

# ── Utilities ──────────────────────────────────────────────────────────────
python -m utils          # Print DeepAxon version and full environment info
```

> **Note:** Use `python -m` to ensure the repo root is on the Python path.
> Fiji is only required for `python -m batch_axon`. All other entry points run without it.
> GPU is optional — all entry points will run on CPU if no CUDA-capable GPU is detected.

---

## Installation

**Python version:** 3.11.x required. Create and activate a Python 3.11 virtual environment before running.

```bash
chmod +x install.sh && ./install.sh   # Linux/macOS
bash install.sh                        # Windows
```

This installs all dependencies in the correct order: core requirements, PyTorch 2.5.1 (CUDA 12.1), and patchify (--no-deps).

**Manual steps (if not using install.sh):**
```bash
pip install -r requirements.txt
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install patchify==0.2.3 --no-deps
```

Core dependencies: PyTorch 2.5.1 (cu121), segmentation-models-pytorch 0.5.0, monai 1.5.2, patchify 0.2.3, opencv-python, numpy, scikit-learn, scipy, rich.

**Note:** patchify is installed with `--no-deps` to avoid numpy conflicts. For CPU-only or different CUDA versions, adjust the PyTorch wheel — see [pytorch.org](https://pytorch.org).

---

## File Naming Convention

### Folders
```
Study\
└── AnimalID\                          e.g. Rb41\
    └── AnimalID_Side_Segment\         e.g. Rb41_Left_E\
```

### Image Files
```
AnimalID_Side_Segment_Mag_###.tif

e.g.
Rb41_Left_E_40X_001.tif
Rb41_Left_E_40X_002.tif
```

### Training Images
```
ctrl_AnimalID_Side_Segment_###.tif    ← control phenotype
regen_AnimalID_Side_Segment_###.tif  ← regenerating phenotype
```

Phenotype prefix is required for analysis pipeline. All training images must have `ctrl_` or `regen_` prefix.

---

## Study Folder Structure

```
NT_Validation_Study4\
└── Rb41\
    └── Rb41_Left_E\
        ├── 4X_TIFF\
        ├── 40X_TIFF\
        ├── CSA\
        │   ├── Rb41_Left_E_4X_CSA.tif
        │   └── Rb41_Left_E_40X_001_CSA.tif
        ├── Cropped\          # Created by python -m segment
        ├── Segmented\        # Created by python -m segment
        └── Morphometrics\    # Created by python -m morphometrics
```

Study-level outputs created by `python -m batch_axon`:
```
NT_Validation_Study4\
    NT_Validation_Study4_Data.xlsx
    logs\
```

---

## BGW Segmentation Color Convention

| Pixel value | Color | Class |
|---|---|---|
| 0 | Black | Background |
| 128 | Grey | Myelin |
| 255 | White | Axon |

**This convention is the contract between `python -m segment` and `python -m morphometrics`.** Do not alter without updating `inRange` thresholds in `morphometrics/morphometrics.py`.

---

## Configuration

Edit `config.json` to set pixel calibration, CLAHE, watershed, patch size, training, and augmentation parameters. See comments in `config.json` for full documentation.

### Key Configuration Reference

| config.json key | Default | Effect |
|---|---|---|
| `clahe.enabled` | false | Enable CLAHE contrast enhancement before segmentation |
| `patch_size.40X` | 256 | Inference patch size for 40X images |
| `watershed.distance_threshold` | 0.17 | Watershed seed threshold |
| `training.class_weights` | [3.0, 1.0, 1.0] | Per-class loss weights [background, myelin, axon] |
| `training.checkpoint_metric` | val_loss | Metric monitored for checkpointing and early stopping |
| `training.early_stop_patience` | 40 | Epochs without val_loss improvement before stopping |
| `logging.train` | true | Write training log files |

---

## Models

Pre-trained `.pt` models are stored in `models/`. The `python -m segment` command presents an interactive model selection menu.

Current production model (rb40x_v1): UNet++ architecture, ResNet34 encoder, ImageNet pretrained weights, 256×256×1 input, 3-class output, PyTorch v5 pipeline. Architecture and encoder for rb40x_v2 to be determined from Wave 1 sweep results.

---

## Training a New Model

### Interactive (local development)

```bash
python -m train
```

Prompts for images folder, magnification, model name, epoch limit, augmentation, and batch size.

### Non-interactive (sbatch / HPRC cluster)

```bash
python -m train --config train_config.json
```

All settings read from `train_config.json`. No prompts. Use this for all cluster runs.

**Example `train_config.json`:**
```json
{
    "images_dir": "/path/to/dataset",
    "mag": "40X",
    "epochs": 400,
    "batch_size": 128,
    "augmentation": false,
    "model_name": "my_model"
}
```

> **Note:** batch_size=128 is the validated default for H100 80GB. AMP (mixed precision) is not supported — excluded after confirmed training collapse across multiple arch/encoder combinations.

### Supported Architectures

| arch string | Architecture | Notes |
|---|---|---|
| `unet` | UNet | Field standard baseline |
| `attention_unet` | Attention UNet | UNet + SCSE decoder attention |
| `unet++` | UNet++ | Current DeepAxon production |
| `unet3+` | UNet3+ | Full-scale skip connections |
| `manet` | MANet | Multi-scale attention |
| `deeplabv3+` | DeepLabV3+ | ASPP context (~42× slower per epoch) |

All architectures support encoders: `resnet34`, `resnet50`, `efficientnet-b3`, `efficientnet-b4`, `densenet121`, `densenet169`.

### On VCU Athena HPRC cluster

```bash
# Setup (once)
ssh mazurm@athena.hprc.vcu.edu
cd ~/deepaxon
source venv/bin/activate

# Submit training job
sbatch train.sbatch
```

Training pipeline steps:
1. Verify source masks for unexpected pixel values
2. Verify image/mask pairs
3. Preprocess into 256×256 patches at 50% overlap (or reuse existing)
4. Split into train/val using stratified phenotype-balanced split
5. Augment training patches in memory
6. Train with weighted Dice + CrossEntropy loss
7. Save best checkpoint based on `val_loss`
8. Write full training log and `result.json`

### Annotation Guidance

Minimum images for a valid training run (40X, bs=128):
- **Minimum:** 5 ctrl + 5 regen = 10 images
- **Good:** 10 ctrl + 10 regen = 20 images
- **Paper quality:** 15 ctrl + 15 regen = 30 images

Priority annotation order: regen images first (more morphological diversity), then different animals within each phenotype.

File naming: all training images must use `ctrl_` or `regen_` prefix (e.g. `ctrl_Rb41_Left_E.tif`).

---

## Analysis Pipeline (Paper 1)

The analysis pipeline runs on the `main` branch of the lab's private GitHub fork (`VCU-Orthopaedics-Microsurgery-Lab/deepaxon`). The public repo will be a clean open-source branch once the best model is selected — it will not include analysis pipeline internals or Athena-specific configs.

### Architecture Sweep Design

Wave 1 sweeps 6 architectures across 6 encoders in three separate SLURM arrays:

| Config | Architectures | Jobs | Array |
|---|---|---|---|
| `analysis_config.json` | unet, unet++, manet, deeplabv3+ | 5,760 | Fast + DeepLab |
| `analysis_config_unet3plus.json` | unet3+ | 1,440 | Fast only |
| `analysis_config_attention_unet.json` | attention_unet | 1,440 | Fast only |

Each sweep is isolated to its own results directory. Results are consolidated after all sweeps complete (see ANAL-11).

### Entry Points

```bash
# Main sweep
python wave1_launcher.py --config analysis_config.json [--dry-run]

# Per-architecture sweeps (queue with dependency on prior array)
python wave1_launcher.py --config analysis_config_unet3plus.json [--dry-run]
python wave1_launcher.py --config analysis_config_attention_unet.json [--dry-run]

# Aggregation (run against each config or unified config post-consolidation)
python aggregator.py --config analysis_config.json --wave 1
python aggregator.py --config analysis_config.json --select \
    --arch <arch> --encoder <enc> --weights <w1,w2,w3> --note "rationale"

# Wave 2 and 3 (after winner.json written)
python wave2_launcher.py --config analysis_config.json --step 2a [--dry-run]
python wave2_launcher.py --config analysis_config.json --step 2b [--dry-run]
python wave3_launcher.py --config analysis_config.json [--dry-run]
```

**Wave 1** — Architecture/encoder/class weight sweep (aug OFF), n=30, 3 splits, 5 seeds. 16 class weight configs. Results written to `analysis/results/sw/`, `analysis_unet3plus/results/sw/`, `analysis_attention_unet/results/sw/`.

**Wave 2** — Augmentation parameter sweep (2,265 jobs) then aug ON vs OFF validation (5 jobs) on winning model from Wave 1. Aug OFF baseline pulled from Wave 1 results — no redundant re-runs.

**Wave 3** — Learning curve on fully optimized model. Dataset sizes auto-selected from winning split: `[6,12,18,24,30]` if 67/33 wins, `[10,20,30]` otherwise. Single winning split only.

### Athena Directory Structure

```
~/deepaxon/
├── venv/                                # NEVER DELETE
├── aggregator.py
├── wave1_launcher.py
├── wave2_launcher.py
├── wave3_launcher.py
├── analysis_config.json
├── analysis_config_unet3plus.json
├── analysis_config_attention_unet.json
├── train/
│   ├── train.py
│   ├── __main__.py
│   ├── unet3plus.py
│   ├── finetune.py
│   └── dataset/
├── segment/
├── morphometrics/
├── utils/
├── analysis/                            # Main sweep (unet/unet++/manet/deeplabv3+)
│   ├── wave1_sw_fast.sbatch
│   ├── wave1_sw_deeplab.sbatch
│   ├── jobs/sw/                         # job_0000.json … job_5759.json
│   ├── results/sw/                      # result.json per run
│   ├── logs/sw/
│   └── aggregated/
├── analysis_unet3plus/                  # UNet3+ sweep
│   ├── wave1_sw_fast.sbatch
│   ├── jobs/sw/
│   ├── results/sw/
│   ├── logs/sw/
│   └── aggregated/
└── analysis_attention_unet/             # Attention UNet sweep
    ├── wave1_sw_fast.sbatch
    ├── jobs/sw/
    ├── results/sw/
    ├── logs/sw/
    └── aggregated/
```

---

## Prerequisites for batch_axon

1. ✅ `python -m segment` — Segmented images in `Segmented\` folder
2. ✅ `python -m morphometrics` — Per-image `.xlsx` in `Morphometrics\` folder
3. ✅ CSA overlays traced in Fiji, saved in `CSA\` folder
4. ✅ Fiji executable path set in `config.json`

---

## CSA Overlay Workflow

1. Open image in Fiji
2. Trace nerve/axonal area with **polygon tool**
3. Add to overlay: `Image > Overlay > Add Selection`
4. Save as `{image_name}_CSA.tif` in `CSA\` folder

> Use **polygon tool** only. Freehand ROIs are not reliably supported.

---

## Output

`python -m batch_axon` produces `{study_name}_Data.xlsx` with:
- One worksheet per animal
- Per-nerve blocks: CSA (µm²), axon count, g-ratio, axon diameter (µm), axon density
- Totals row with extrapolated full axon count
- Conditional formatting for QC

---

## Development

```
deepaxon/
├── config.json
├── analysis_config.json                 # Main sweep config
├── analysis_config_unet3plus.json       # UNet3+ sweep config
├── analysis_config_attention_unet.json  # Attention UNet sweep config
├── wave1_launcher.py
├── wave2_launcher.py
├── wave3_launcher.py
├── aggregator.py
├── requirements.txt
├── install.sh
├── train/
│   ├── train.py                         # Training loop, _ARCH_MAP, build_model()
│   ├── __init__.py
│   ├── __main__.py                      # Entry point, skip logic, --config flag
│   ├── unet3plus.py                     # UNet3+ — full-scale skip connections
│   ├── finetune.py                      # Fine-tuning (implemented, not validated)
│   └── dataset/
│       ├── split.py                     # Stratified phenotype-balanced split
│       ├── data_loader.py               # Manifest mode (ctrl_/regen_ prefixes)
│       ├── augment.py                   # Config mode + parametric aug_params mode
│       ├── preprocess.py
│       └── __init__.py
├── segment/
│   ├── segment.py                       # Inference, Hann blending, BGW output
│   ├── __init__.py
│   └── __main__.py
├── morphometrics/
│   ├── morphometrics.py                 # Watershed, matching, quality filters
│   ├── distributions.py                 # Three-tier diameter binning
│   ├── analyze_nerve.py
│   ├── __init__.py
│   └── __main__.py
└── utils/
    ├── metrics.py                       # Dice, IoU, HD95, hd95_myelin_axon
    ├── helpers.py                       # Shared utilities, get_path_input
    ├── class_balance.py                 # Pixel class balance reporting
    ├── version.py                       # v5.1.0 / v5_analysis
    ├── logger.py
    ├── resize.py
    ├── gpu.py
    ├── __init__.py
    └── __main__.py
```
