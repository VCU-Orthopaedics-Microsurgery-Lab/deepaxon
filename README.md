# DeepAxon

Automated nerve cross-section segmentation and morphometric analysis using UNet++.

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
```

**Python version:** 3.10.x recommended.
TensorFlow 2.10.1 is pinned because it is the last native Windows GPU build before WSL2
was required. **Do not upgrade TensorFlow without testing GPU compatibility.**

---

## File Naming Convention

### Folders
```
Study\
└── AnimalID\                          e.g. Rb41\
    └── AnimalID_Side_Segment\         e.g. Rb41_Left_E\
```

- **Side**: `Left` or `Right` (or `L` / `R`)
- **Segment**: any identifier used in the study design — e.g. `E`, `F`, `Prox`, `Dist`

### Image Files
```
AnimalID_Side_Segment_Mag_###.tif

e.g.
Rb41_Left_E_40X_001.tif
Rb41_Left_E_40X_002.tif
Rb41_Right_Prox_100X_001.tif
```

The image number `###` is only required if more than one image exists per nerve.

### CSA Overlay Files
```
AnimalID_Side_Segment_4X_CSA.tif        <- whole-nerve (4X), one per nerve
AnimalID_Side_Segment_Mag_###_CSA.tif   <- per-image axonal area (40X)

e.g.
Rb41_Left_E_4X_CSA.tif
Rb41_Left_E_40X_001_CSA.tif
Rb41_Left_E_40X_002_CSA.tif
```

**100X images:** Per-image CSA tracing is not required at 100X because the image field
captures only axonal area. Only the 4X whole-nerve CSA is needed.

---

## Study Folder Structure

```
NT_Validation_Study4\
└── Rb41\
    └── Rb41_Left_E\
        ├── 4X_TIFF\               # Low-magnification reference images
        ├── 40X_TIFF\              # High-magnification images (or 100X_TIFF\)
        ├── CSA\                   # Fiji overlay files for area tracing
        │   ├── Rb41_Left_E_4X_CSA.tif
        │   ├── Rb41_Left_E_40X_001_CSA.tif
        │   └── Rb41_Left_E_40X_002_CSA.tif
        ├── Cropped\               # Created by python -m segment (cropped source images)
        ├── Segmented\             # Created by python -m segment
        └── Morphometrics\         # Created by python -m morphometrics
```

Study-level outputs created by `python -m batch_axon`:
```
NT_Validation_Study4\
    NT_Validation_Study4_Data.xlsx
    logs\
        batch_axon_log_YYYYMMDD_HHMMSS.txt
        segment_log_YYYYMMDD_HHMMSS.txt
        morphometrics_log_YYYYMMDD_HHMMSS.txt
```

All entry points accept study, animal, or nerve level input paths and resolve
the correct scope automatically.

---

## BGW Segmentation Color Convention

Segmented images use a fixed 3-class grayscale encoding:

| Pixel value | Color  | Class      |
|-------------|--------|------------|
| 0           | Black  | Background |
| 128         | Grey   | Axon       |
| 255         | White  | Myelin     |

**This convention is the contract between `python -m segment` and
`python -m morphometrics`.** Do not alter output colormap without updating
the `inRange` thresholds in `morphometrics/morphometrics.py`.

---

## Configuration

Edit `config.json` to set:
- **pixel_size_um**: Calibration values for your microscope (µm/pixel per magnification and image width)
- **fiji_executable**: Path to your Fiji executable (auto-prompted and saved on first run of `batch_axon` if blank)
- **clahe**: Contrast Limited Adaptive Histogram Equalization — disabled by default, configurable
- **watershed**: Distance threshold and dilation disk size for morphometric watershed segmentation
- **patch_size**: Inference patch size in pixels (256 for both 40X and 100X)

### Pixel Size Calibration (Olympus BX63)

| Objective | Scale bar | px measured | µm/px (1440px wide) | µm/px (2880px wide) |
|-----------|-----------|-------------|----------------------|----------------------|
| 4X        | 500 µm    | 230         | 1.087                | —                    |
| 10X       | 100 µm    | 130         | 0.769                | —                    |
| 20X       | 100 µm    | 230         | 0.435                | —                    |
| 40X       | 50 µm     | 230         | 0.217                | —                    |
| 100X      | —         | —           | **TBD**              | **TBD**              |

> **100X not yet calibrated.** Physical unit output (µm, µm²) will be unavailable
> for 100X images until `pixel_size_um["100X"]` is set in `config.json`.
> Pixel-unit measurements will still be reported.

To calibrate: measure a known scale bar in Fiji at the target magnification,
then set `pixel_size_um` in `config.json` using `image_width_px: µm_per_px`.

---

## Models

Pre-trained `.keras` models are stored in `models/`. The `python -m segment` command
presents an interactive selection menu of available models.

Current models:

| File | Magnification | Notes |
|------|--------------|-------|
| `p_100x_v2r_256.keras` | 100X | Most recent 100X model |
| `p_100x_v2_256.keras`  | 100X | 100X base v2 |
| `p_100x_v2_ctrl_256.keras` | 100X | 100X control variant |
| `p_40x_v2_256.keras`   | 40X  | Current 40X production model |
| `rb_40x_v1_256.keras`  | 40X  | Legacy — superseded by v2 |

All models use UNet++ architecture, 256×256×1 input, 3-class softmax output.
See `models/model_registry.json` for full training metadata.

---

## Prerequisites for batch_axon

Before running `python -m batch_axon`, the following must be complete for each nerve:

1. ✅ `python -m segment` — Segmented images in `Segmented\` folder
2. ✅ `python -m morphometrics` — Per-image `.xlsx` files in `Morphometrics\` folder
3. ✅ CSA overlays traced in Fiji and saved in `CSA\` folder
4. ✅ Fiji executable path set in `config.json` (auto-prompted on first run)

`python -m batch_axon` will skip any nerve missing morphometrics and warn in the log.

---

## CSA Overlay Workflow

1. Open image in Fiji
2. Manually trace the nerve/axonal area using the **polygon tool**
3. Add to overlay: `Image > Overlay > Add Selection`
4. Save file as `{image_name}_CSA.tif` in the nerve's `CSA\` folder
5. For 4X whole-nerve CSA: save as `{nerve_name}_4X_CSA.tif`

> **Important:** Use the **polygon tool** only. Freehand ROIs are not reliably
> supported by the automated ROI export macro.

> **Multi-part nerves:** If a nerve spans multiple 4X images, trace each part
> separately and save as `{nerve_name}_4X_001_CSA.tif`, `{nerve_name}_4X_002_CSA.tif` etc.
> `python -m batch_axon` will sum the areas automatically.

`python -m batch_axon` processes these overlays using Fiji in headless mode.
The Fiji executable path is set once in `config.json` and auto-prompted on first run.

---

## Output

`python -m batch_axon` produces `{study_name}_Data.xlsx` in the study folder with:
- One worksheet per animal
- Per-nerve blocks with per-image rows: CSA (µm²), axon count, g-ratio, axon diameter (µm), axon density (axons/µm²)
- Totals row per nerve with estimated full axon count (extrapolated from sampled area to whole-nerve 4X CSA)
- Conditional formatting for quick visual QC of axon count and g-ratio

---

## Sharing Images with Remote Collaborators

Place shared `.tif` images in the `sample_images/` folder in the repo root.
This folder is the designated landing spot for images shared by colleagues
without access to the lab imaging drive.

---

## Development

```
deepaxon/
├── config.json
├── requirements.txt
├── sample_images/       # Shared test images for remote collaborators
├── notebooks/           # Experimental and QC notebooks
├── models/              # Production .keras model files + model_registry.json
├── utils/               # Shared: console, resize, helpers, metrics, gpu
├── segment/             # python -m segment entry point
├── morphometrics/       # python -m morphometrics entry point
├── batch_axon/          # python -m batch_axon entry point
│   └── overlay/         # Fiji ROI processing (process_overlay.py, export_roi.ijm)
└── train/               # python -m train entry point
    ├── data/            # augment, preprocess, data_loader
    └── models/          # unet, unet_plus_plus architecture definitions
```

### Adding a New Model

1. Train using `python -m train` — model saved to `train/models/`
2. Evaluate against held-out test images
3. Copy best `.keras` file to `deepaxon/models/` using naming convention:
   `{cohort}_{mag}_v{version}_{patch_size}_{YYYYMMDD}.keras`
4. Add entry to `models/model_registry.json`

### Key Configuration Reference

| config.json key | Default | Effect |
|----------------|---------|--------|
| `clahe.enabled` | false | Enable CLAHE contrast enhancement before segmentation |
| `clahe.clip_limit` | 1.0 | CLAHE clip limit — higher = more contrast enhancement |
| `patch_size.40X` | 256 | Inference patch size for 40X images |
| `patch_size.100X` | 256 | Inference patch size for 100X images |
| `watershed.distance_threshold` | 0.1 | Watershed seed threshold — lower = more seeds |
| `watershed.dilation_disk` | 5 | Background marker dilation radius |
| `logging` | false | Write log files for each run |
| `timing` | false | Write timing CSV during segmentation |
