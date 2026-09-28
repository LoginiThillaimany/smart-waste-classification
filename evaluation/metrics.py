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

import numpy as np
import torch


def count_parameters(model: torch.nn.Module) -> int:
    """Total trainable parameter count — used for model complexity comparison."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module, tmp_path: str = "_tmp_model_size.pt") -> float:
    """Serializes the model to disk to measure its on-disk footprint in MB."""
    torch.save(model.state_dict(), tmp_path)
    size_mb = Path(tmp_path).stat().st_size / (1024 ** 2)
    if Path(tmp_path).exists():
        Path(tmp_path).unlink()
    return round(size_mb, 2)


def compute_classification_metrics(all_labels: list, all_preds: list, num_classes: int):
    """Compute accuracy, weighted precision, recall, F1, and confusion matrix using NumPy."""
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    n_samples = len(y_true)

    # Accuracy
    acc = float(np.mean(y_true == y_pred)) if n_samples > 0 else 0.0

    # Confusion matrix (rows: true, cols: pred)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    # Per-class metrics
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = cm[c, :].sum()

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        supports.append(support)

    total_support = sum(supports)
    if total_support > 0:
        weighted_p = sum(p * s for p, s in zip(precisions, supports)) / total_support
        weighted_r = sum(r * s for r, s in zip(recalls, supports)) / total_support
        weighted_f1 = sum(f1 * s for f1, s in zip(f1s, supports)) / total_support
    else:
        weighted_p = weighted_r = weighted_f1 = 0.0

    return {
        "accuracy": acc,
        "precision": float(weighted_p),
        "recall": float(weighted_r),
        "f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_model(model: torch.nn.Module, loader, device: str, class_names: list) -> dict:
    """
    Runs inference over `loader` and returns a dictionary of standard
    classification metrics plus per-image inference time.
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

    num_classes = len(class_names)
    results = compute_classification_metrics(all_labels, all_preds, num_classes)
    results["inference_time_per_image_sec"] = elapsed / max(len(loader.dataset), 1)
    results["class_names"] = class_names

    # ROC-AUC calculation (optional fallback)
    try:
        from sklearn.metrics import roc_auc_score
        results["roc_auc"] = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except Exception:
        results["roc_auc"] = None

    return results


def summarize_efficiency(model: torch.nn.Module, training_time_sec: float) -> dict:
    """Collects the non-accuracy comparison measures the rubric requires."""
    return {
        "parameter_count": count_parameters(model),
        "model_size_mb": get_model_size_mb(model),
        "training_time_sec": round(training_time_sec, 2),
    }
