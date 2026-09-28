"""
generate_plots.py

Generates visualization figures for MobileNetV2:
1. Training and validation loss curve
2. Training and validation accuracy curve
3. Test confusion matrix heatmap

Saves all figures to results/figures/
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORY_PATH = PROJECT_ROOT / "models/mobilenetv2/checkpoints/training_history.json"
RESULTS_PATH = PROJECT_ROOT / "models/mobilenetv2/results.json"
FIGURES_DIR = PROJECT_ROOT / "results/figures"


def plot_metrics():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(HISTORY_PATH, "r") as f:
        history = json.load(f)

    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Plot Accuracy Curve
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Accuracy", linewidth=2, markersize=4)
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], "g-s", label="Val Accuracy", linewidth=2, markersize=4)
    plt.title("MobileNetV2: Training vs Validation Accuracy", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mobilenetv2_accuracy.png")
    plt.close()
    print("Saved mobilenetv2_accuracy.png")

    # 2. Plot Loss Curve
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2, markersize=4)
    plt.plot(epochs, history["val_loss"], "r-s", label="Val Loss", linewidth=2, markersize=4)
    plt.title("MobileNetV2: Training vs Validation Loss", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mobilenetv2_loss.png")
    plt.close()
    print("Saved mobilenetv2_loss.png")

    # 3. Plot Confusion Matrix
    cm = np.array(results["confusion_matrix"])
    class_names = results["class_names"]

    plt.figure(figsize=(9, 7), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
        annot_kws={"size": 11, "weight": "bold"},
    )
    plt.title("MobileNetV2: Test Confusion Matrix (97.30% Accuracy)", fontsize=14, fontweight="bold", pad=14)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("True Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mobilenetv2_confusion_matrix.png")
    plt.close()
    print("Saved mobilenetv2_confusion_matrix.png")


if __name__ == "__main__":
    plot_metrics()
