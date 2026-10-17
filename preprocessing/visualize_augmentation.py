"""
visualize_augmentation.py

Generates a visual comparison of original vs augmented images
to demonstrate the data augmentation pipeline applied during training.

Saves: results/figures/augmentation_examples.png

Usage:
    python preprocessing/visualize_augmentation.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Use headless backend for saving figure directly to disk
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Add project root directory to path to import preprocessing settings
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.preprocess import train_transform, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

# Directories for training split and saving output images
SPLIT_DIR = Path("data/split/train")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Augmentation pipeline for display (without normalization so colors look natural)
display_augment = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),              # Resize to 224x224
    transforms.RandomHorizontalFlip(),                    # Random horizontal flip
    transforms.RandomRotation(15),                         # Random rotation up to 15 deg
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Slight brightness and contrast change
])

# Baseline transform for display (resize only, no augmentation)
display_original = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
])


def main():
    """Pick sample images from train split and display original vs augmented versions."""
    random.seed(42)  # Set seed for reproducible image selection
    classes = sorted([d for d in SPLIT_DIR.iterdir() if d.is_dir()])

    # Pick 1 sample image from each of the first 4 classes
    selected = []
    for cls_dir in classes[:4]:
        imgs = list(cls_dir.glob("*.*"))
        selected.append((cls_dir.name, random.choice(imgs)))

    # Create a 4x4 subplot grid (4 classes x 4 columns: 1 original + 3 augmented)
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))

    for row, (cls_name, img_path) in enumerate(selected):
        # Open image and convert to standard RGB
        img = Image.open(img_path).convert("RGB")

        # Column 0: Original image
        orig = display_original(img)
        axes[row][0].imshow(orig)
        axes[row][0].set_title(f"{cls_name}\n(Original)", fontsize=9, fontweight="bold")
        axes[row][0].axis("off")  # Hide axis borders

        # Columns 1 to 3: Three randomly augmented variations of the same image
        for col in range(1, 4):
            aug = display_augment(img)
            axes[row][col].imshow(aug)
            axes[row][col].set_title(f"Augmented #{col}", fontsize=9)
            axes[row][col].axis("off")

    # Add overall chart title
    plt.suptitle(
        "Data Augmentation Examples\n(RandomHorizontalFlip + RandomRotation(15°) + ColorJitter)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    # Save output plot
    plt.savefig(FIG_DIR / "augmentation_examples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'augmentation_examples.png'}")


# Execute main if script is run directly
if __name__ == "__main__":
    main()
