# DeepAxon

Automated nerve cross-section segmentation and morphometric analysis using deep learning.

## Entry Points

```bash
python -m segment        # Segment a study folder of nerve images
python -m morphometrics  # Run per-image morphometric analysis
python -m batch_axon     # Compile study-level morphometric summary
python -m train          # Train a new segmentation model
```

> **Note:** Use `python -m` to ensure the repo root is on the Python path.
> Fiji is only required for `python -m batch_axon`. All other entry points run without it.
> GPU is optional — all entry points will run on CPU if no CUDA-capable GPU is detected.

---

## Installation

```bash
pip install -r requirements.txt
pip install patchify --no-deps
```

**Python version:** 3.11.x required.

Core dependencies: PyTorch 2.5.1 (cu121), segmentation-models-pytorch 0.5.0, monai, patchify, opencv-python, numpy, scikit-learn, scipy, rich.

**Note:** patchify is installed with `--no-deps` to avoid numpy conflicts. Install PyTorch with the correct CUDA wheel for your system first. See [pytorch.org](https://pytorch.org) for the correct install command.

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

Phenotype prefix is required for analysis pipeline. All 30 training images must have `ctrl_` or `regen_` prefix.

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

All models: UNet++ architecture, ResNet34 encoder, ImageNet pretrained weights, 256×256×1 input, 3-class output, PyTorch v5 pipeline.

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
    "epochs": 200,
    "batch_size": 256,
    "augmentation": false,
    "model_name": "my_model"
}
```

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

Minimum images for a valid training run (40X, bs=256):
- **Minimum:** 5 ctrl + 5 regen = 10 images
- **Good:** 10 ctrl + 10 regen = 20 images
- **Paper quality:** 15 ctrl + 15 regen = 30 images

Priority annotation order: regen images first (more morphological diversity), then different animals within each phenotype.

File naming: all training images must use `ctrl_` or `regen_` prefix (e.g. `ctrl_Rb41_Left_E.tif`).

---

## Analysis Pipeline (Paper 1 — v5_analysis branch)

The `v5_analysis` branch contains a full three-wave SLURM job array pipeline for systematic architecture, encoder, class weight, and augmentation optimization.

### Entry Points

```bash
python wave1_launcher.py --config analysis_config.json [--dry-run]  # 5,760 jobs
python wave2_launcher.py --config analysis_config.json --step 2a    # 2,065 jobs
python wave2_launcher.py --config analysis_config.json --step 2b    # 5 jobs
python wave3_launcher.py --config analysis_config.json              # 45 jobs
python aggregator.py --config analysis_config.json [--wave 1/2a/2b/3]
```

See `analysis_config.json` for full configuration. Fill in real Athena paths before first run.

**Wave 1** — Architecture/encoder/class weight sweep (aug OFF), n=30, all 3 splits, 5 seeds.
**Wave 2** — Augmentation parameter sweep on winning model from Wave 1.
**Wave 3** — Learning curve on fully optimized model.

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
├── analysis_config.json        # Analysis pipeline master config (v5_analysis)
├── wave1_launcher.py           # Wave 1 SLURM launcher
├── wave2_launcher.py           # Wave 2 SLURM launcher
├── wave3_launcher.py           # Wave 3 SLURM launcher
├── aggregator.py               # Results aggregation and table generation
├── requirements.txt
├── utils/
│   ├── logger.py
│   ├── helpers.py
│   ├── metrics.py              # compute_epoch_metrics + compute_all_metrics
│   ├── resize.py
│   ├── gpu.py
│   └── version.py              # v5.1.0 / v5_analysis
├── train/
│   ├── train.py
│   └── dataset/
│       ├── split.py            # Stratified phenotype-balanced split
│       ├── data_loader.py      # Manifest mode (ctrl_/regen_ prefixes)
│       ├── augment.py          # Config mode + parametric aug_params mode
│       └── preprocess.py
├── segment/
├── morphometrics/
└── batch_axon/
```