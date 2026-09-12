# Dataset Documentation

> Fill in every field below once the dataset has been downloaded and the EDA notebook has run.
> This file is graded directly against the rubric's "Description of the dataset" criterion — completeness here matters.

## Source Metadata

| Field | Value |
|---|---|
| Dataset name | |
| Source URL | |
| Creator / Author | |
| License | |
| Date accessed | |
| Original number of classes | |
| Original total image count | |

## Class Mapping

Document how the original dataset's classes were consolidated into this project's 6 target classes.

| Target class | Source folder(s) | Rationale |
|---|---|---|
| Plastic | | |
| Paper/Cardboard | | |
| Metal | | |
| Glass | | |
| Organic | | |
| Residual | | |

## Post-Consolidation Statistics

| Field | Value |
|---|---|
| Total images (after consolidation) | |
| Image resolution range | |
| Image formats present (jpg/png/etc.) | |
| Color mode (RGB/grayscale/mixed) | |

## Class Distribution

| Class | Image count | % of total |
|---|---|---|
| Plastic | | |
| Paper/Cardboard | | |
| Metal | | |
| Glass | | |
| Organic | | |
| Residual | | |

> Insert `results/figures/class_distribution.png` reference here once generated.

## Data Quality Issues Identified

| Issue | Count found | Action taken |
|---|---|---|
| Corrupted / unreadable files | | |
| Duplicate images (exact) | | |
| Near-duplicate images (perceptual hash) | | |
| Mislabeled images (manually spot-checked) | | |

## Access Instructions (for reproducibility)

```bash
pip install kaggle
# Place your kaggle.json API token in ~/.kaggle/ (or %USERPROFILE%\.kaggle\ on Windows)
kaggle datasets download -d <dataset-slug>
unzip <dataset-file>.zip -d data/raw/
python preprocessing/remap_classes.py
```

## Fixed Split Configuration

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Train ratio | 0.70 |
| Validation ratio | 0.15 |
| Test ratio | 0.15 |
| Split script | `preprocessing/preprocess.py` |

**Important:** This split is generated once and shared across all four models. Do not regenerate it independently — doing so risks data leakage and breaks fair comparison between models.
