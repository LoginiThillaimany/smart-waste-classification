"""
preprocess.py

Shared preprocessing pipeline for the Smart Waste Classification project.

This module is the single source of truth for:
  1. Consolidating the raw dataset into the project's 6 target classes.
  2. Creating a fixed, reproducible train/validation/test split.
  3. Providing PyTorch DataLoaders with a consistent transform policy.

All four models (Custom CNN, ResNet50, MobileNetV2, EfficientNetB0) must
import `get_loaders()` from this module rather than writing their own
splitting/loading logic, to guarantee a fair, leakage-free comparison.

Usage:
    python preprocessing/preprocess.py --remap --split
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import argparse
import random
import shutil
from pathlib import Path

# PyTorch utilities for handling datasets and data batches
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42           # Fixed random seed for reproducible data splits
IMG_SIZE = 224             # Target image resolution (224x224 px) for all models
DEFAULT_BATCH_SIZE = 16    # Number of images processed together in one batch

# Directory paths for raw, consolidated, and train/val/test split datasets
RAW_DIR = Path("data/raw/garbage_classification")
CONSOLIDATED_DIR = Path("data/consolidated")
SPLIT_DIR = Path("data/split")

# Target split proportions: 70% train, 15% validation, 15% test
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Dictionary mapping raw Kaggle dataset folders into the 6 standardized classes
CLASS_MAPPING = {
    "plastic": "Plastic",
    "paper": "Paper_Cardboard",
    "cardboard": "Paper_Cardboard",
    "metal": "Metal",
    "brown-glass": "Glass",
    "green-glass": "Glass",
    "white-glass": "Glass",
    "biological": "Organic",
    "trash": "Residual",
    "clothes": "Residual",
    "shoes": "Residual",
    "battery": "Residual",
}


# ---------------------------------------------------------------------------
# Step 1 — Class Consolidation
# ---------------------------------------------------------------------------
def remap_classes(source_dir: Path = RAW_DIR, dest_dir: Path = CONSOLIDATED_DIR) -> None:
    """Copy raw dataset images into the 6 consolidated target-class folders."""
    # Ensure source folder exists before starting
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {source_dir}. "
            "Download it first (see data_documentation/dataset_info.md)."
        )

    # Loop through each mapped folder and copy images into corresponding target folder
    for src_folder, target_class in CLASS_MAPPING.items():
        src = source_dir / src_folder
        if not src.exists():
            print(f"  [skip] source folder not found: {src}")
            continue

        # Create target class directory if not present
        target = dest_dir / target_class
        target.mkdir(parents=True, exist_ok=True)

        # Copy each image with a new prefix to prevent name collisions
        for i, img_path in enumerate(src.glob("*.*")):
            shutil.copy(img_path, target / f"{src_folder}_{i}{img_path.suffix}")

    # Display final image count per class
    print("Class consolidation complete. Counts:")
    for cls_dir in sorted(dest_dir.iterdir()):
        print(f"  {cls_dir.name}: {len(list(cls_dir.glob('*.*')))} images")


# ---------------------------------------------------------------------------
# Step 2 — Fixed, Reproducible Split
# ---------------------------------------------------------------------------
def create_split(
    source_dir: Path = CONSOLIDATED_DIR,
    dest_dir: Path = SPLIT_DIR,
    ratios: dict = SPLIT_RATIOS,
    seed: int = RANDOM_SEED,
) -> None:
    """
    Create one fixed train/val/test split, shared by all four models.

    This must only be run ONCE. Re-running with a different seed invalidates
    any prior training results and breaks fair comparison across models.
    """
    random.seed(seed)  # Fix random seed for exact reproducibility

    # Iterate through each class directory
    for cls_dir in source_dir.iterdir():
        images = list(cls_dir.glob("*.*"))
        random.shuffle(images)  # Shuffle images randomly within the class

        # Compute number of images for train and val splits
        n = len(images)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])

        # Slice the shuffled list into train, val, and test groups
        subsets = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        # Copy files to corresponding split subdirectories (train/val/test)
        for split_name, files in subsets.items():
            out_dir = dest_dir / split_name / cls_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, out_dir / f.name)

    print(f"Split complete (seed={seed}, ratios={ratios}). Test set is now locked.")


# ---------------------------------------------------------------------------
# Step 3 — Shared Transforms and DataLoaders
# ---------------------------------------------------------------------------
# Standard ImageNet RGB mean and standard deviation for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training transformations: data augmentation (flip, rotation, color jitter) + normalization
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),            # Resize image to 224x224
    transforms.RandomHorizontalFlip(),                  # Flip image horizontally with 50% probability
    transforms.RandomRotation(15),                       # Random rotation within +/- 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Slight color adjustment
    transforms.ToTensor(),                              # Convert PIL image to PyTorch tensor (0 to 1)
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), # Standardize pixel distribution
])

# Evaluation / testing transformations: only resize and normalize (no random augmentations)
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),            # Resize image to 224x224
    transforms.ToTensor(),                              # Convert to tensor
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), # Standardize pixel distribution
])


def get_loaders(root: Path = SPLIT_DIR, batch_size: int = DEFAULT_BATCH_SIZE, num_workers: int = 0):
    """
    Returns (train_loader, val_loader, test_loader, class_names).

    All models must call this function rather than building their own
    DataLoader, to guarantee identical preprocessing across the comparison.
    """
    # Create PyTorch datasets from folder structure
    train_ds = datasets.ImageFolder(str(root / "train"), transform=train_transform)
    val_ds = datasets.ImageFolder(str(root / "val"), transform=eval_transform)
    test_ds = datasets.ImageFolder(str(root / "test"), transform=eval_transform)

    # Wrap datasets in DataLoaders for batched training and evaluation
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)   # Shuffle for training
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)     # No shuffle for validation
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)   # No shuffle for test

    return train_loader, val_loader, test_loader, train_ds.classes


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Command line argument parser
    parser = argparse.ArgumentParser(description="Smart Waste Classification preprocessing pipeline")
    parser.add_argument("--remap", action="store_true", help="Consolidate raw dataset into 6 target classes")
    parser.add_argument("--split", action="store_true", help="Create the fixed train/val/test split")
    args = parser.parse_args()

    # Execute selected tasks
    if args.remap:
        remap_classes()
    if args.split:
        create_split()
    if not args.remap and not args.split:
        parser.print_help()
