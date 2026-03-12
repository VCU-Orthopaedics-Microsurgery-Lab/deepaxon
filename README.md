# DeepAxon

Automated nerve cross-section segmentation and morphometric analysis using UNet++.

## Entry Points

```bash
python segment        # Segment a study folder of nerve images
python morphometrics  # Run per-image morphometric analysis
python batch_axon     # Compile study-level morphometric summary
python train          # Train a new segmentation model
```

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
        ├── Segmented\             # Created by python segment
        └── Morphometrics\         # Created by python morphometrics
```

Study-level outputs created by `python batch_axon`:
```
NT_Validation_Study4\
    NT_Validation_Study4_Data.xlsx
    batch_axon_log_YYYYMMDD_HHMMSS.txt
```

---

## Configuration

Edit `config.json` to set:
- **pixel_size_um**: Calibration values for your microscope (um/pixel per magnification and image width)
- **fiji_executable**: Path to your Fiji executable (prompted on first run if blank)

If your magnification is not listed, physical unit outputs will be unavailable and pixel measurements will be reported instead.

### Current Calibration (Olympus BX63)

| Objective | Scale bar | Pixels | um/px (1440) | um/px (2880) |
|-----------|-----------|--------|--------------|--------------|
| 4X        | 500 um    | 230    | 2.174        | 1.087        |
| 10X       | 100 um    | 115    | 0.769        | 0.385        |
| 20X       | 100 um    | 230    | 0.435        | 0.217        |
| 40X       | 50 um     | 230    | 0.217        | 0.109        |
| 100X      | -         | -      | TBD          | TBD          |

---

## Models

Pre-trained `.keras` models are stored in `models/`. The `python segment` command presents a selection menu of available models.

**Compatibility:** Models were trained on TensorFlow 2.10.1. Use `tensorflow>=2.10,<2.16` to ensure compatibility.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## CSA Overlay Workflow

1. Open image in Fiji
2. Manually trace the axonal area using the polygon or freehand tool
3. Save as overlay: Image > Overlay > Add Selection
4. Save file as `{image_name}_CSA.tif` in the nerve's `CSA\` folder
5. For 4X whole-nerve CSA: save as `{nerve_name}_4X_CSA.tif`

`python batch_axon` processes these overlays automatically using Fiji in headless mode.
The Fiji executable path is set once in `config.json`.

**Note:** Fiji is only required for `python batch_axon`. The other entry points do not need it.

---

## Output

`python batch_axon` produces `{study_name}_Data.xlsx` in the study folder with:
- One worksheet per animal
- Per-nerve blocks with per-image rows (CSA, axon count, g-ratio, axon diameter, axon density)
- Totals row with estimated full axon count (extrapolated using 4X CSA)
- Conditional formatting for axon count and g-ratio

---

## Development

```
deepaxon/
├── config.json
├── models/              # Published .keras files
├── utils/               # Shared: console, resize, helpers, metrics, gpu
├── segment/             # python segment entry point
├── morphometrics/       # python morphometrics entry point
├── batch_axon/          # python batch_axon entry point
│   └── overlay/         # Fiji ROI processing
├── train/               # python train entry point
│   ├── data/            # augment, preprocess, data_loader
│   └── models/          # unet, unet_plus_plus
└── tools/               # Lab-specific utilities
```
