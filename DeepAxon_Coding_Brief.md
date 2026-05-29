# DeepAxon Paper 1 — Analysis Pipeline: Coding Brief

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
- Repo: `kushsavsani/deepaxon-legacy`, branch `v5_pytorch`
- Data: `~/rb40x_v2_hprc` (flat `images/` + `masks/`, `val_` prefix for validation patches)
- Outputs: `~/deepaxon/models/` and `~/deepaxon/logs/`
- Training time: ~7 min per run on H100
- Always use `sbatch` for cluster runs — never run training interactively

---

## Dataset (current state)
- 30 annotated images, 22 animals, 3 independent studies
- 9 animals contributed 2 images each; remaining contributed 1
- 1:1 phenotype ratio: 15 control, 15 regenerating
- 40X rabbit sciatic nerve, Toluidine Blue brightfield
- **Pipeline must be dataset-size agnostic** — adding images requires only a config value change
- Phenotype label must be tracked per image throughout all analyses
- Target: 40 images (10 more masks needed); interim analysis at 30 images is valid

---

## Current Model
- Architecture: UNet++, Encoder: ResNet34 (segmentation-models-pytorch)
- 3-class segmentation: axon interior, myelin sheath, background
- Loss: Dice + SoftCrossEntropy, class weights [3.0, 1.0, 1.0]
- Checkpointing and early stopping monitor `val_loss`
- Learning rate scheduling: standard (not permuted — industry standard, mentioned once in methods)

### Augmentation
- 5 types: 3 geometric, 2 photometric
- Doubly stochastic: global random trigger rate + each augmentation has its own independent probability
- Photometric augmentations applied to **images only** — never to masks
- Original unaugmented images excluded from augmented training set (avoids inflating dataset size / overfitting)
- All parameters user-configurable via JSON config file, empirically tuned to minimal

---

## Split Strategy
- **Stratified random splitting** — phenotype balance (1:1 control/regen) maintained within each split
- Pure random splitting risks imbalance on small datasets — stratified sampling prevents this
- 5 fixed seeds per condition (seeds 1–5) — reproducibility controls, **never optimized**
- Same seeds used across all architectures, encoders, augmentation conditions

| Dataset size | Viable splits | Test n | Per-phenotype test n | Observations ×5 seeds |
|---|---|---|---|---|
| 30 images | 80/20 | 6 | 3 | 30 |
| 30 images | 70/30 | 9* | 4/5 | 45 |
| 32 images | 70/30 | 10 | 5 | 50 |
| 32 images | 80/20 | 6 | 3 | 30 |
| 40 images | 70/30 | 12 | 6 | 60 |
| 40 images | 80/20 | 8 | 4 | 40 |
| 40 images | 90/10 | 4 | 2 | 20 |

*Odd test set at 30 images 70/30 — handle via stratified sampling (4 control + 5 regen or 5+4)

**Current plan (30 images):** run 70/30 and 80/20. Add 90/10 automatically when dataset reaches 40 images.

---

## Metrics — Computed Every Run
**Primary (main tables):** Dice, IoU, F1 — mean ± SD, per image
**Secondary (reported selectively):** Recall, Precision, AUC, Jaccard, Hausdorff Distance, Sensitivity, Specificity, Cohen's Kappa

### Per-class reporting
- 3 classes: axon interior, myelin sheath, background
- 2 phenotypes: control, regenerating
- = 6 values per metric per model per condition
- Mean across classes as headline metric in main table
- Per-class breakdown in supplementary tables
- **Background class inflates mean Dice** — report mean with and without background; flag in paper
- **Myelin class is hardest and most scientifically meaningful** — most likely to differentiate architectures
- **Hausdorff Distance** — boundary accuracy; directly relevant since myelin boundary precision affects downstream g-ratio calculation; most likely reviewer request at Medical Image Analysis / IEEE TMI

---

## Architecture List (Wave 1)
The paper's goal is to identify the **best architecture for Toluidine Blue brightfield peripheral nerve segmentation** — DeepAxon's backbone will be updated to the winning architecture. The story is validation and optimization, not defense of a predetermined choice.

All smp architectures are plug-and-play:

```python
# Architecture comparison (encoder fixed at ResNet34)
smp.Unet(encoder_name="resnet34")                    # UNet — ADS equivalent, field standard baseline
smp.AttentionUnet(encoder_name="resnet34")           # Attention UNet — Preetam's suggestion
smp.UnetPlusPlus(encoder_name="resnet34")            # UNet++ — current DeepAxon
smp.MAnet(encoder_name="resnet34")                   # MAnet — multi-scale attention

# Encoder comparison (within winning architecture)
winning_arch(encoder_name="resnet34")                # current encoder
winning_arch(encoder_name="resnet50")                # Preetam's suggestion; deeper, modest compute increase
winning_arch(encoder_name="efficientnet-b4")         # strong perf/param ratio; good for small datasets
```

### External benchmarks (no training — separate inference pipelines)
- **ADS** — run out-of-the-box on all images, metrics computed against same ground truth masks. Real-world benchmark reflecting how labs actually use it today.
- **MedSAM** — zero-shot inference, no fine-tuning (bowang-lab/MedSAM). Modern foundation model baseline.

### Possible future addition (not currently in scope)
- **SegFormer** — transformer-based, increasingly expected in 2025/2026 papers; available via HuggingFace `transformers`. Requires training loop wrapper and different input/output handling — 1-2 days integration. Flag as future direction in paper; add if reviewers request it.
- **DeepLab v3+** — occasional reviewer request
- **DenseNet / VGG encoders** — occasional reviewer request

---

## Two-Wave Analysis Structure

### Wave 1 — One Massive Parallel SLURM Job Array (aug OFF)

**Full permutation matrix:**
```
Architectures:  UNet, AttentionUnet, UNet++ (ResNet34/50/EfficientNet-B4), MAnet (ResNet34/50/EfficientNet-B4)
TT splits:      70/30, 80/20 (add 90/10 when dataset ≥ 40 images)
Dataset sizes:  10, 20, 30 (add 40 when available)
Seeds:          1–5
Augmentation:   OFF
```

Each SLURM array job = one permutation. Results written to structured output directory. Aggregated after all jobs complete.

**Tables extracted by slicing results:**

**Table 1 — Joint TT Split × Learning Curve**
Performance surface: split ratio × dataset size × Dice/IoU/F1. Reference model: DeepAxon (UNet++ ResNet34). Locks best split and confirms dataset saturation simultaneously — avoids circularity of locking split before saturation confirmed. Rerun on winning architecture if it differs from reference.

**Table 2 — Architecture Comparison**
Encoder fixed at ResNet34 (isolates architecture as only variable). Best split, full dataset, aug off.
Rows: UNet, Attention UNet, UNet++ ResNet34, MAnet ResNet34, ADS (external), MedSAM (external)

**Table 3 — Encoder Comparison**
Within winning architecture from Table 2. Best split, full dataset, aug off.
Rows: ResNet34, ResNet50, EfficientNet-B4

---

### Wave 2 — Augmentation (sequential, on winning model only)

Triggered via `--dependency=afterok` after Wave 1 completes.

**Step 2a — Per-augmentation parameter sweep**
- Winning architecture + encoder from Wave 1
- Sweep probability × intensity independently per augmentation type
- Fixed seed during sweep
- Best split, full dataset
- Output: optimized probability + intensity per augmentation type

**Step 2b — Aug on vs. aug off**
- Winning architecture + encoder
- Optimized aug parameters (from 2a) vs. no aug
- Best split, 5 seeds

**Table 4 — Augmentation Sweep Matrix**
Probability × intensity per augmentation type, F1 as primary metric

**Table 5 — Augmentation Effect**
Winning model aug on (optimized params) vs. aug off, disaggregated by phenotype (control vs. regen)

---

## Circularity — Resolved
- TT split and learning curve entangled in small datasets → joint Table 1 analysis solves this
- Aug parameters require trained model → Wave 1 runs first, aug sweep on winner only
- Architecture unknown before aug sweep → Wave 1 aug-off establishes winner before Wave 2
- No circular dependencies anywhere in the plan

---

## SLURM Notes
- Each job = one permutation (architecture × encoder × split × dataset size × seed)
- Use `--array=0-N` where N = total permutation count
- Each job reads config from pre-generated JSON or derives from array index
- Results written to structured output: `~/deepaxon/logs/results_{arch}_{encoder}_{split}_{n}_{seed}.json`
- Final aggregation script collects all outputs into master table after Wave 1 completes
- Wave 2 triggered with `--dependency=afterok:[Wave1_JobID]`
- Estimated Wave 1 wall time: ~2–3 hours on 4 H100 nodes (parallelized)

---

## Code Style
- Terse and direct
- Mark changed lines `# ← CHANGED` / `# ← NEW`
- Flag out-of-scope issues — never fix silently
- Never regenerate entire files — patch specific sections only
- All hyperparameters and config in JSON — never hardcoded
- Pipeline must be fully reproducible from config file alone

---

## Deferred to Paper 2 (Protocol Paper)
- Hold-out test set (4 images, 2 control, 2 regen — never seen during development)
- Control/regen training ratio analysis (1:1 vs. 2:1 vs. 1:2)
- Watershed sensitivity analysis (4 ROIs ~50 axons each)
- Inter-rater variability on masks (already completed)
- Bland-Altman morphometric comparisons (DeepAxon vs. ADS vs. manual)
- BatchAxon demonstration
- Multi-species and multi-magnification generalization
- Open-source release (GitHub, MIT license)

---

## Known Limitations (state in paper)
- Dataset limited to 40X rabbit; 30 images current, 40 target
- 90/10 split deferred until dataset reaches 40 images
- Aug parameters optimized on winning architecture only
- Control/regen ratio fixed at 1:1 — ratio analysis deferred to protocol paper
- Osmium tetroxide staining not used — higher contrast myelin definition achievable but less accessible
- SegFormer not included in primary analysis — flagged as future direction
