"""
verify_split.py

Utility to print train/val/test split statistics per class.
Run after `preprocess.py --split` to confirm the fixed split is correct
and that train/val/test ratios match the 70/15/15 target.

Usage:
    python preprocessing/verify_split.py
"""

from pathlib import Path


SPLIT_DIR = Path("data/split")


def verify_split(split_dir: Path = SPLIT_DIR) -> None:
    """Print per-class image counts for train, val and test splits."""
    if not split_dir.exists():
        print(f"Split directory not found: {split_dir}")
        print("Run `python preprocessing/preprocess.py --split` first.")
        return

    splits = ["train", "val", "test"]
    all_classes = set()
    counts = {}

    for split in splits:
        split_path = split_dir / split
        if not split_path.exists():
            print(f"  [missing] {split_path}")
            continue
        counts[split] = {}
        for cls_dir in sorted(split_path.iterdir()):
            if cls_dir.is_dir():
                n = len(list(cls_dir.glob("*.*")))
                counts[split][cls_dir.name] = n
                all_classes.add(cls_dir.name)

    all_classes = sorted(all_classes)

    # Header
    header = f"{'Class':<20}" + "".join(f"{s:>10}" for s in splits) + f"{'Total':>10}" + f"{'Train%':>10}" + f"{'Val%':>10}" + f"{'Test%':>10}"
    print("=" * len(header))
    print("SPLIT VERIFICATION REPORT")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    grand = {s: 0 for s in splits}
    for cls in all_classes:
        row = f"{cls:<20}"
        total = 0
        cls_counts = []
        for s in splits:
            n = counts.get(s, {}).get(cls, 0)
            cls_counts.append(n)
            total += n
            grand[s] += n
            row += f"{n:>10}"
        row += f"{total:>10}"
        for n in cls_counts:
            pct = (n / total * 100) if total > 0 else 0
            row += f"{pct:>9.1f}%"
        print(row)

    # Grand total
    print("-" * len(header))
    grand_total = sum(grand.values())
    row = f"{'TOTAL':<20}"
    for s in splits:
        row += f"{grand[s]:>10}"
    row += f"{grand_total:>10}"
    for s in splits:
        pct = (grand[s] / grand_total * 100) if grand_total > 0 else 0
        row += f"{pct:>9.1f}%"
    print(row)
    print("=" * len(header))

    print(f"\nExpected ratios: Train=70.0% | Val=15.0% | Test=15.0%")
    print("If the percentages above are close to these targets, the split is correct.")


if __name__ == "__main__":
    verify_split()
