"""
metrics.py

Shared evaluation module for the Smart Waste Classification project.

Every model (Custom CNN, ResNet50, MobileNetV2, EfficientNetB0) must be
scored using `evaluate_model()` so that accuracy, precision, recall, F1,
ROC-AUC, confusion matrix, and efficiency measures are computed identically
across the four-model comparison.
"""

import time
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def count_parameters(model: torch.nn.Module) -> int:
    """Total trainable parameter count — used for model complexity comparison."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module, tmp_path: str = "_tmp_model_size.pt") -> float:
    """Serializes the model to disk to measure its on-disk footprint in MB."""
    torch.save(model.state_dict(), tmp_path)
    size_mb = Path(tmp_path).stat().st_size / (1024 ** 2)
    Path(tmp_path).unlink()
    return round(size_mb, 2)


def evaluate_model(model: torch.nn.Module, loader, device: str, class_names: list) -> dict:
    """
    Runs inference over `loader` and returns a dictionary of standard
    classification metrics plus per-image inference time.

    Args:
        model: trained PyTorch model in eval mode.
        loader: DataLoader (typically the held-out test loader).
        device: "cuda" or "cpu".
        class_names: list of class labels, e.g. train_ds.classes.

    Returns:
        dict with accuracy, precision, recall, f1, roc_auc (if applicable),
        confusion_matrix, and inference_time_per_image (seconds).
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    start_time = time.time()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())
    elapsed = time.time() - start_time

    results = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "inference_time_per_image_sec": elapsed / max(len(loader.dataset), 1),
        "class_names": class_names,
    }

    # ROC-AUC requires one-vs-rest probability scores; skip gracefully if it fails
    # (e.g. a class is entirely absent from a small evaluation batch).
    try:
        results["roc_auc"] = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except ValueError:
        results["roc_auc"] = None

    return results


def summarize_efficiency(model: torch.nn.Module, training_time_sec: float) -> dict:
    """Collects the non-accuracy comparison measures the rubric requires."""
    return {
        "parameter_count": count_parameters(model),
        "model_size_mb": get_model_size_mb(model),
        "training_time_sec": round(training_time_sec, 2),
    }
