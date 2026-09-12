# Smart Waste Classification Using Deep Learning

**Module:** SE4050 – Deep Learning
**Project type:** Supervised Deep Learning — 4-model comparison
**Team size:** 3 members

## 1. Overview

This project builds an image-classification system that assigns a waste item to one of six categories, then compares four distinct deep-learning architectures under identical, fair experimental conditions (same dataset, same fixed train/validation/test split, same preprocessing policy, same evaluation metrics).

### Target Classes

| Class | Scope |
|---|---|
| Plastic | Bottles, containers, packaging |
| Paper / Cardboard | Paper, newspapers, books, cardboard packaging |
| Metal | Aluminium/steel/tin cans, metal containers |
| Glass | Glass bottles and containers |
| Organic | Food, kitchen, and biodegradable waste |
| Residual | Waste outside the above categories |

## 2. Model Comparison Strategy

Four distinct architectures are implemented and compared, satisfying the module's minimum requirement of four deep-learning models for supervised projects:

| # | Model | Type | Ownership |
|---|---|---|---|
| 1 | Custom CNN | Built from scratch | Member 1 (primary) |
| 2 | ResNet50 | Transfer learning (deep residual) | Member 2 (primary) |
| 3 | MobileNetV2 | Transfer learning (lightweight) | Member 3 (primary) |
| 4 | EfficientNetB0 | Transfer learning (shared benchmark) | All 3 members (1/3 each) |

All four models are trained and evaluated on the same fixed data split and scored with the same metrics (accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix), plus efficiency measures (training time, parameter count, model size, inference time).

## 3. Group Responsibilities

| Work area | Member 1 | Member 2 | Member 3 |
|---|---|---|---|
| Primary model | Custom CNN | ResNet50 | MobileNetV2 |
| EfficientNetB0 (shared) | 1/3 | 1/3 | 1/3 |
| Dataset + EDA | Lead | Support | Support |
| Preprocessing | Lead | Support | Support |
| Evaluation framework | Support | Lead | Support |
| Final comparison + critical analysis | Support | Support | Lead |
| Report | Equal | Equal | Equal |
| GitHub evidence | Equal | Equal | Equal |
| Presentation + viva | Equal | Equal | Equal |

Each member commits to their own model folder under `models/`; the shared `EfficientNetB0` folder is co-owned with each member's contribution clearly attributed in commit messages.

## 4. Repository Structure

```
smart-waste-classification/
├── data_documentation/     # Dataset metadata and sourcing record
├── eda/                    # Exploratory data analysis notebook(s)
├── preprocessing/          # Data cleaning, splitting, augmentation code
├── models/
│   ├── custom_cnn/
│   ├── resnet50/
│   ├── mobilenetv2/
│   └── efficientnetb0_shared/
├── evaluation/             # Shared metrics module used by all models
├── experiments/            # Training logs, configs, random seeds
├── results/figures/        # Generated charts, confusion matrices, curves
├── requirements.txt
└── README.md
```

## 5. Setup Instructions

### Prerequisites
- Python 3.11 (avoid 3.13 — several dependencies lack stable wheels for it)
- Git
- (Optional, recommended) NVIDIA GPU with CUDA support for faster training

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd smart-waste-classification

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify GPU availability (optional but recommended)
python -c "import torch; print(torch.cuda.is_available())"
```

### Dataset

Dataset access instructions and licensing are documented in [`data_documentation/dataset_info.md`](data_documentation/dataset_info.md). The dataset is not committed to this repository; download it via the Kaggle CLI as described there.

## 6. Reproducibility

- Fixed random seed: `42` (used for the train/val/test split and model initialization where applicable)
- Fixed split ratios: 70% train / 15% validation / 15% test
- All preprocessing and hyperparameter decisions are made using only training and validation data; the test set is held out until final evaluation.

## 7. License

This repository is submitted as coursework for SE4050 – Deep Learning. Dataset licensing terms are documented separately in `data_documentation/dataset_info.md`.
