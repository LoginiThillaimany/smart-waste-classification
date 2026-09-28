"""
evaluate.py

Evaluation script for the Custom CNN model.
Loads the best checkpoint, runs inference on the held-out test set using
the shared evaluation module, generates visualisations (confusion matrix,
learning curves) and saves all results.

Usage:
    python models/custom_cnn/evaluate.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.metrics import evaluate_model, summarize_efficiency
from models.custom_cnn.config import BATCH_SIZE, BEST_MODEL_PATH, CHECKPOINT_DIR, NUM_CLASSES
from models.custom_cnn.custom_cnn_model import CustomCNN
from preprocessing.preprocess import get_loaders

# Output directories
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_learning_curves(history: dict, save_dir: Path = FIG_DIR) -> None:
    """Plot and save training/validation accuracy and loss curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    # --- Accuracy ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy", marker="o", markersize=3)
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy", marker="s", markersize=3)
    plt.title("Custom CNN — Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "custom_cnn_accuracy.png", dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'custom_cnn_accuracy.png'}")

    # --- Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    plt.plot(epochs, history["val_loss"], label="Validation Loss", marker="s", markersize=3)
    plt.title("Custom CNN — Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "custom_cnn_loss.png", dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'custom_cnn_loss.png'}")


def plot_confusion_matrix(cm: list, class_names: list, save_dir: Path = FIG_DIR) -> None:
    """Plot and save the confusion matrix as a heatmap."""
    cm_array = np.array(cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_array, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title("Custom CNN — Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_dir / "custom_cnn_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'custom_cnn_confusion_matrix.png'}")


def evaluate():
    """Load best model, evaluate on test set, save results and plots."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    _, _, test_loader, class_names = get_loaders(batch_size=BATCH_SIZE)
    print(f"Test set size: {len(test_loader.dataset)}")

    # Load model
    model = CustomCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    print(f"Loaded best model from {BEST_MODEL_PATH}")

    # Evaluate using shared metrics module
    results = evaluate_model(model, test_loader, device, class_names)

    # Print results
    print(f"\n{'='*50}")
    print("Custom CNN — Test Set Results")
    print(f"{'='*50}")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-score:  {results['f1']:.4f}")
    print(f"  ROC-AUC:   {results['roc_auc']}")
    print(f"  Inference: {results['inference_time_per_image_sec']*1000:.2f} ms/image")

    # Load training history for learning curves and efficiency
    history_path = Path(CHECKPOINT_DIR) / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        plot_learning_curves(history)
        training_time = history.get("total_training_time_sec", 0)
    else:
        print("Warning: training_history.json not found — skipping learning curves.")
        training_time = 0

    # Efficiency metrics
    efficiency = summarize_efficiency(model, training_time)
    print(f"\n  Parameters: {efficiency['parameter_count']:,}")
    print(f"  Model size: {efficiency['model_size_mb']} MB")
    print(f"  Training time: {efficiency['training_time_sec']}s")

    # Confusion matrix plot
    plot_confusion_matrix(results["confusion_matrix"], class_names)

    # Save complete results to JSON
    all_results = {**results, **efficiency}
    # Convert numpy types for JSON serialization
    all_results["confusion_matrix"] = results["confusion_matrix"]

    results_path = Path("models/custom_cnn/results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {results_path}")

    return all_results


if __name__ == "__main__":
    evaluate()
