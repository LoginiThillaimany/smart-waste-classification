"""
evaluate.py

Evaluation script for the ResNet50 model.
Loads the best checkpoint, runs inference on the held-out test set using
the shared evaluation module, generates visualisations (confusion matrix,
learning curves for both training phases) and saves all results.

Usage:
    python models/resnet50/evaluate.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.metrics import evaluate_model, measure_inference_time, summarize_efficiency
from models.resnet50.config import BATCH_SIZE, CHECKPOINT_DIR, NUM_CLASSES, PHASE2_BEST_PATH
from models.resnet50.resnet50_model import ResNet50Classifier
from preprocessing.preprocess import get_loaders

# Output directories
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_learning_curves(history: dict, save_dir: Path = FIG_DIR) -> None:
    """Plot and save two-phase training/validation accuracy and loss curves."""
    p1 = history.get("phase1", {})
    p2 = history.get("phase2", {})

    train_acc = p1.get("train_acc", []) + p2.get("train_acc", [])
    val_acc = p1.get("val_acc", []) + p2.get("val_acc", [])
    train_loss = p1.get("train_loss", []) + p2.get("train_loss", [])
    val_loss = p1.get("val_loss", []) + p2.get("val_loss", [])

    epochs = range(1, len(train_loss) + 1)
    p1_len = len(p1.get("train_loss", []))

    # --- Accuracy ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy", marker="o", markersize=3)
    plt.plot(epochs, val_acc, label="Validation Accuracy", marker="s", markersize=3)
    if p1_len > 0 and p1_len < len(epochs):
        plt.axvline(x=p1_len + 0.5, color="red", linestyle="--", label="Phase 2 Start (Fine-Tuning)")
    plt.title("ResNet50 — Two-Phase Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "resnet50_accuracy.png", dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'resnet50_accuracy.png'}")

    # --- Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss", marker="o", markersize=3)
    plt.plot(epochs, val_loss, label="Validation Loss", marker="s", markersize=3)
    if p1_len > 0 and p1_len < len(epochs):
        plt.axvline(x=p1_len + 0.5, color="red", linestyle="--", label="Phase 2 Start (Fine-Tuning)")
    plt.title("ResNet50 — Two-Phase Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "resnet50_loss.png", dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'resnet50_loss.png'}")


def plot_confusion_matrix(cm: list, class_names: list, save_dir: Path = FIG_DIR) -> None:
    """Plot and save the confusion matrix using matplotlib."""
    cm_array = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_array, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm_array.shape[1]),
        yticks=np.arange(cm_array.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="ResNet50 — Confusion Matrix (Test Set)",
        ylabel="True",
        xlabel="Predicted",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm_array.max() / 2.0
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(
                j, i, format(cm_array[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_array[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.savefig(save_dir / "resnet50_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'resnet50_confusion_matrix.png'}")


def plot_misclassified_examples(model, loader, device: str, class_names: list, num_examples: int = 12, save_path: Path = FIG_DIR / "resnet50_misclassified_examples.png") -> None:
    """Find and save a grid image of misclassified test examples (true vs predicted label)."""
    model.eval()
    misclassified = []

    inv_mean = np.array([0.485, 0.456, 0.406])
    inv_std = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for imgs, labels in loader:
            imgs_dev = imgs.to(device)
            outputs = model(imgs_dev)
            preds = outputs.argmax(dim=1).cpu()

            wrong_indices = (preds != labels).nonzero(as_tuple=True)[0]
            for idx in wrong_indices:
                img_tensor = imgs[idx]
                # Unnormalize: (C, H, W) -> (H, W, C)
                img_np = img_tensor.permute(1, 2, 0).numpy() * inv_std + inv_mean
                img_np = np.clip(img_np, 0.0, 1.0)
                misclassified.append((img_np, int(labels[idx]), int(preds[idx])))
                if len(misclassified) >= num_examples:
                    break
            if len(misclassified) >= num_examples:
                break

    if not misclassified:
        print("No misclassified examples found on test set!")
        return

    n_samples = len(misclassified)
    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        if i < n_samples:
            img_np, true_label, pred_label = misclassified[i]
            axes[i].imshow(img_np)
            true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
            pred_name = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)
            axes[i].set_title(f"True: {true_name}\nPred: {pred_name}", color="crimson", fontsize=10, fontweight="bold")
            axes[i].axis("off")
        else:
            axes[i].axis("off")

    plt.suptitle("ResNet50 — Misclassified Test Examples", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def evaluate():
    """Load best model, evaluate on test set, save results and plots."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    _, _, test_loader, class_names = get_loaders(batch_size=BATCH_SIZE)
    print(f"Test set size: {len(test_loader.dataset)}")

    # Load model
    model = ResNet50Classifier(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(PHASE2_BEST_PATH, map_location=device, weights_only=True))
    print(f"Loaded best model from {PHASE2_BEST_PATH}")

    # Evaluate using shared metrics module
    results = evaluate_model(model, test_loader, device, class_names)

    # Isolated inference time measurement (batch size 1, GPU synchronization)
    isolated_lat = measure_inference_time(model, device)
    results["inference_time_isolated_sec"] = isolated_lat

    # Print results
    print(f"\n{'='*55}")
    print("ResNet50 — Test Set Results")
    print(f"{'='*55}")
    print(f"  Accuracy:         {results['accuracy']:.4f}")
    print(f"  Weighted Prec:    {results['precision']:.4f}")
    print(f"  Weighted Recall:  {results['recall']:.4f}")
    print(f"  Weighted F1:      {results['f1']:.4f}")
    print(f"  Macro Precision:  {results['macro_precision']:.4f}")
    print(f"  Macro Recall:     {results['macro_recall']:.4f}")
    print(f"  Macro F1-score:   {results['macro_f1']:.4f}")
    print(f"  Macro ROC-AUC:    {results['roc_auc']:.4f}")
    print(f"  Inference (loader): {results['inference_time_per_image_sec']*1000:.2f} ms/image")
    print(f"  Inference (single): {results['inference_time_isolated_sec']*1000:.2f} ms/image")

    print("\nPer-class Metrics:")
    for name, m in results["per_class"].items():
        print(f"  {name:<16} Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1']:.4f} | Support: {m['support']}")

    print("\nROC-AUC per class (One-vs-Rest):")
    for name, auc_val in zip(class_names, results["roc_auc_per_class"]):
        print(f"  {name:<16} ROC-AUC: {auc_val:.4f}")

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
    print(f"\nEfficiency Measures:")
    print(f"  Total Parameters:     {efficiency['total_parameters']:,}")
    print(f"  Trainable Parameters: {efficiency['trainable_parameters']:,}")
    print(f"  Model size:           {efficiency['model_size_mb']} MB")
    print(f"  Training time:        {efficiency['training_time_sec']}s ({efficiency['training_time_sec']/60:.1f} min)")

    # Confusion matrix plot
    plot_confusion_matrix(results["confusion_matrix"], class_names)

    # Save misclassified examples plot
    plot_misclassified_examples(model, test_loader, device, class_names, num_examples=12)

    # Save complete results to JSON
    all_results = {**results, **efficiency}
    all_results["confusion_matrix"] = results["confusion_matrix"]

    results_path = Path("models/resnet50/results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {results_path}")

    return all_results


if __name__ == "__main__":
    evaluate()
