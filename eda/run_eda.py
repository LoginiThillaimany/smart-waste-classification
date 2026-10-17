"""
run_eda.py

Exploratory Data Analysis script for the Smart Waste Classification project.
Generates all EDA figures and prints statistics for the dataset documentation.

Outputs:
    - results/figures/class_distribution.png
    - results/figures/sample_grid.png
    - results/figures/image_dimensions.png
    - Console output with corruption/duplicate/dimension stats

Usage:
    python eda/run_eda.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for saving plots without GUI
import matplotlib.pyplot as plt
import numpy as np

# Pillow is needed to load and inspect images
try:
    from PIL import Image
except ImportError:
    print("Pillow not installed. Run: pip install pillow")
    sys.exit(1)

# imagehash is optional, used for finding duplicate images
try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("Warning: imagehash not installed — skipping duplicate detection.")

# ---------------------------------------------------------------------------
# Folder Paths
# ---------------------------------------------------------------------------
CONSOLIDATED_DIR = Path("data/consolidated")  # Folder where merged class images are stored
FIG_DIR = Path("results/figures")             # Folder to save output charts and plots
FIG_DIR.mkdir(parents=True, exist_ok=True)    # Create the figures folder if it does not exist


def check_dataset_exists():
    """Verify that the consolidated dataset folder exists before running EDA."""
    if not CONSOLIDATED_DIR.exists():
        print(f"ERROR: {CONSOLIDATED_DIR} not found.")
        print("Run `python preprocessing/preprocess.py --remap` first.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 1. Class Distribution
# ---------------------------------------------------------------------------
def plot_class_distribution():
    """Count images in each class, print stats, and save a bar chart."""
    counts = {}

    # Count how many images are in each class folder
    for cls_dir in sorted(CONSOLIDATED_DIR.iterdir()):
        if cls_dir.is_dir():
            counts[cls_dir.name] = len(list(cls_dir.glob("*.*")))

    # Calculate total image count
    total = sum(counts.values())

    # Print summary table to console
    print("\n" + "=" * 50)
    print("CLASS DISTRIBUTION")
    print("=" * 50)
    for cls, n in counts.items():
        pct = n / total * 100  # Percentage of this class
        print(f"  {cls:<20} {n:>6}  ({pct:.1f}%)")
    print(f"  {'TOTAL':<20} {total:>6}")

    # Set distinct colors for each bar
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#607D8B"]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(counts.keys(), counts.values(), color=colors, edgecolor="white", linewidth=0.5)

    # Put exact count value above each bar
    for bar, count in zip(bars, counts.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 str(count), ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Add plot labels and titles
    plt.title("Class Distribution — Smart Waste Dataset (15,515 images)", fontsize=14, fontweight="bold")
    plt.ylabel("Image Count", fontsize=12)
    plt.xlabel("Waste Category", fontsize=12)
    plt.xticks(rotation=20, fontsize=10)
    plt.grid(axis="y", alpha=0.3)  # Light horizontal grid lines
    plt.tight_layout()

    # Save figure to disk and close
    plt.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close()
    print(f"\nSaved: {FIG_DIR / 'class_distribution.png'}")

    return counts


# ---------------------------------------------------------------------------
# 2. Image Dimensions & Corruption Check
# ---------------------------------------------------------------------------
def check_images():
    """Scan all images to verify readability, find corrupted files, and collect sizes."""
    records = []       # Stores valid image details (width, height, format, mode)
    corrupted = []     # Stores list of corrupted files and their errors
    formats = set()    # Set of unique image formats (e.g., JPEG, PNG)
    modes = set()      # Set of unique color modes (e.g., RGB)

    print("\n" + "=" * 50)
    print("IMAGE QUALITY CHECK")
    print("=" * 50)

    # Loop through each category folder
    for cls_dir in sorted(CONSOLIDATED_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue

        # Check each image file inside the category
        for img_path in cls_dir.glob("*.*"):
            try:
                # Test if the file is readable and not broken
                with Image.open(img_path) as im:
                    im.verify()  # Verify image integrity

                # Re-open to read width, height, format, and color mode
                with Image.open(img_path) as im:
                    records.append({
                        "class": cls_dir.name,
                        "width": im.width,
                        "height": im.height,
                        "mode": im.mode,
                        "format": im.format,
                    })
                    formats.add(im.format)
                    modes.add(im.mode)
            except Exception as e:
                # Record error if image is damaged or cannot be read
                corrupted.append((str(img_path), str(e)))

    # Print scan results to console
    print(f"  Total images scanned: {len(records) + len(corrupted)}")
    print(f"  Valid images: {len(records)}")
    print(f"  Corrupted/unreadable: {len(corrupted)}")
    if corrupted:
        for path, err in corrupted[:10]:
            print(f"    - {path}: {err}")

    # Calculate dimension statistics (min, max, average)
    widths = [r["width"] for r in records]
    heights = [r["height"] for r in records]
    print(f"\n  Image formats: {', '.join(str(f) for f in formats)}")
    print(f"  Color modes: {', '.join(str(m) for m in modes)}")
    print(f"  Width  — min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.0f}")
    print(f"  Height — min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.0f}")

    # Create a scatter plot of image resolutions (width vs height)
    plt.figure(figsize=(8, 6))
    plt.scatter(widths, heights, alpha=0.1, s=5, color="teal")
    plt.title("Image Dimensions — Width vs Height", fontsize=13, fontweight="bold")
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save scatter plot
    plt.savefig(FIG_DIR / "image_dimensions.png", dpi=150)
    plt.close()
    print(f"\n  Saved: {FIG_DIR / 'image_dimensions.png'}")

    return records, corrupted, formats, modes


# ---------------------------------------------------------------------------
# 3. Duplicate Detection
# ---------------------------------------------------------------------------
def detect_duplicates():
    """Find near-duplicate images using perceptual image hashing."""
    if not HAS_IMAGEHASH:
        print("\n  Skipping duplicate detection (imagehash not installed).")
        return 0

    print("\n" + "=" * 50)
    print("DUPLICATE DETECTION (perceptual hash)")
    print("=" * 50)

    hashes = defaultdict(list)

    # Compute hash for every image
    for cls_dir in sorted(CONSOLIDATED_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img_path in cls_dir.glob("*.*"):
            try:
                # Average hash generates a fingerprint based on visual content
                h = imagehash.average_hash(Image.open(img_path))
                hashes[str(h)].append(str(img_path))
            except Exception:
                pass

    # Filter out hashes that appear more than once (duplicates)
    dupes = {h: paths for h, paths in hashes.items() if len(paths) > 1}
    total_dupe_images = sum(len(p) for p in dupes.values())
    print(f"  Duplicate groups found: {len(dupes)}")
    print(f"  Total images in duplicate groups: {total_dupe_images}")

    # Print a few sample duplicate paths
    if dupes:
        print(f"  Sample duplicates (first 3 groups):")
        for i, (h, paths) in enumerate(list(dupes.items())[:3]):
            print(f"    Group {i+1} (hash={h}): {len(paths)} images")
            for p in paths[:3]:
                print(f"      - {p}")

    return len(dupes)


# ---------------------------------------------------------------------------
# 4. Sample Image Grid
# ---------------------------------------------------------------------------
def plot_sample_grid():
    """Plot 2 sample images per waste category in a side-by-side grid."""
    import random
    random.seed(42)  # Set seed for reproducible random selection

    classes = sorted([d for d in CONSOLIDATED_DIR.iterdir() if d.is_dir()])
    n_classes = len(classes)

    # Create a 2-row subplot grid (2 samples per class column)
    fig, axes = plt.subplots(2, n_classes, figsize=(3 * n_classes, 6))

    for i, cls_dir in enumerate(classes):
        imgs = list(cls_dir.glob("*.*"))
        samples = random.sample(imgs, min(2, len(imgs)))  # Pick 2 random images

        for j in range(2):
            ax = axes[j][i] if n_classes > 1 else axes[j]
            try:
                # Open image and convert to RGB format
                img = Image.open(samples[j]).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "Error", ha="center", va="center")
            ax.axis("off")  # Hide axis ticks and lines

            # Add class title on top row only
            if j == 0:
                ax.set_title(cls_dir.name, fontsize=10, fontweight="bold")

    plt.suptitle("Sample Images per Class", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Save the sample grid image
    plt.savefig(FIG_DIR / "sample_grid.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {FIG_DIR / 'sample_grid.png'}")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    """Main workflow to run all EDA steps."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS — Smart Waste Classification")
    print("=" * 60)

    # Step 1: Ensure dataset folder is present
    check_dataset_exists()

    # Step 2: Plot and print class balance
    counts = plot_class_distribution()

    # Step 3: Check dimensions, color modes, and corruption
    records, corrupted, formats, modes = check_images()

    # Step 4: Check for near-duplicate images
    n_dupes = detect_duplicates()

    # Step 5: Generate visual grid of sample images
    plot_sample_grid()

    # Step 6: Print overall dataset summary
    widths = [r["width"] for r in records]
    heights = [r["height"] for r in records]

    print("\n" + "=" * 60)
    print("SUMMARY FOR dataset_info.md")
    print("=" * 60)
    print(f"  Total images: {len(records)}")
    print(f"  Resolution range: {min(widths)}×{min(heights)} to {max(widths)}×{max(heights)}")
    print(f"  Formats: {', '.join(str(f) for f in formats)}")
    print(f"  Color modes: {', '.join(str(m) for m in modes)}")
    print(f"  Corrupted files: {len(corrupted)}")
    print(f"  Duplicate groups: {n_dupes}")
    print("\nEDA complete. All figures saved to results/figures/")


# Run the script when executed from terminal
if __name__ == "__main__":
    main()
