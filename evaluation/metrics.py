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


class ParameterCountDict(dict):
    """Dictionary holding total and trainable parameter counts with backward-compatible format support."""
    def __format__(self, format_spec: str) -> str:
        if format_spec:
            return format(self["total"], format_spec)
        return super().__format__(format_spec)


def count_parameters(model: torch.nn.Module) -> dict:
    """Total and trainable parameter count — used for model complexity comparison."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ParameterCountDict({"total": total, "trainable": trainable})


def get_model_size_mb(model: torch.nn.Module, tmp_path: str = "_tmp_model_size.pt") -> float:
    """Serializes the model to disk to measure its on-disk footprint in MB."""
    torch.save(model.state_dict(), tmp_path)
    size_mb = Path(tmp_path).stat().st_size / (1024 ** 2)
    if Path(tmp_path).exists():
        Path(tmp_path).unlink()
    return round(size_mb, 2)


def compute_roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute binary ROC-AUC using trapezoidal integration in pure NumPy."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return 0.0

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0 or tps[-1] <= 0:
        return 0.0

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    # Trapezoidal integration
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))
    return auc


def compute_roc_auc_ovr(y_true: list, y_probs: list, num_classes: int) -> tuple[float, list[float]]:
    """Compute multi-class One-vs-Rest (OvR) ROC-AUC in pure NumPy."""
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    per_class_auc = []
    for c in range(num_classes):
        binary_true = (y_true == c).astype(int)
        binary_score = y_probs[:, c] if y_probs.ndim == 2 else y_probs
        per_class_auc.append(compute_roc_auc_binary(binary_true, binary_score))
    macro_auc = float(np.mean(per_class_auc)) if per_class_auc else 0.0
    return macro_auc, per_class_auc


def compute_classification_metrics(all_labels: list, all_preds: list, num_classes: int, class_names: list = None):
    """Compute accuracy, weighted and macro precision/recall/F1, per_class breakdown, and confusion matrix using NumPy."""
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

    macro_p = float(np.mean(precisions)) if len(precisions) > 0 else 0.0
    macro_r = float(np.mean(recalls)) if len(recalls) > 0 else 0.0
    macro_f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0

    if class_names is None:
        class_names = [f"Class_{c}" for c in range(num_classes)]

    per_class = {}
    for c in range(num_classes):
        name = class_names[c] if c < len(class_names) else f"Class_{c}"
        per_class[name] = {
            "precision": float(precisions[c]),
            "recall": float(recalls[c]),
            "f1": float(f1s[c]),
            "support": int(supports[c]),
        }

    return {
        "accuracy": acc,
        "precision": float(weighted_p),
        "recall": float(weighted_r),
        "f1": float(weighted_f1),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def measure_inference_time(
    model: torch.nn.Module,
    device: str,
    input_shape: tuple = (1, 3, 224, 224),
    warmup: int = 20,
    runs: int = 100,
) -> float:
    """
    Measures isolated single-image inference latency (in seconds) without DataLoader overhead.
    Uses torch.cuda.synchronize() when running on CUDA for accurate hardware timing.
    """
    model.eval()
    dummy_input = torch.randn(*input_shape, device=device)
    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
        if is_cuda:
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(runs):
            _ = model(dummy_input)
        if is_cuda:
            torch.cuda.synchronize()
        total_time = time.perf_counter() - start_time

    return total_time / max(runs, 1)


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
    results = compute_classification_metrics(all_labels, all_preds, num_classes, class_names=class_names)
    results["inference_time_per_image_sec"] = elapsed / max(len(loader.dataset), 1)
    results["class_names"] = class_names

    macro_auc, per_class_auc = compute_roc_auc_ovr(all_labels, all_probs, num_classes)
    results["roc_auc"] = macro_auc
    results["roc_auc_per_class"] = per_class_auc

    return results


def summarize_efficiency(model: torch.nn.Module, training_time_sec: float) -> dict:
    """Collects the non-accuracy comparison measures the rubric requires."""
    params = count_parameters(model)
    return {
        "parameter_count": params,
        "total_parameters": params["total"],
        "trainable_parameters": params["trainable"],
        "model_size_mb": get_model_size_mb(model),
        "training_time_sec": round(training_time_sec, 2),
    }
