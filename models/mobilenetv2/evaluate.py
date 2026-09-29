"""
evaluate.py

Evaluation script for the MobileNetV2 waste classification model.

Computes:
    - Overall Test Loss & Accuracy
    - Per-class Precision, Recall, and F1-Score
    - Macro and Weighted averages
    - Saves evaluation results to results.json

Usage:
    python models/mobilenetv2/evaluate.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.mobilenetv2.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    NUM_CLASSES,
    RESULTS_PATH,
)
from models.mobilenetv2.mobilenetv2_model import MobileNetV2Classifier
from preprocessing.preprocess import get_loaders


def evaluate():
    """Evaluate MobileNetV2 model on test split."""
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Evaluating MobileNetV2 on device: {device}")

    best_ckpt = Path(BEST_MODEL_PATH)
    if not best_ckpt.exists():
        print(f"Error: Model checkpoint not found at {BEST_MODEL_PATH}")
        print("Please run `python models/mobilenetv2/train.py` first.")
        return

    # Load data loaders
    train_loader, val_loader, test_loader, class_names = get_loaders(batch_size=BATCH_SIZE)

    # Load model architecture & weights
    model = MobileNetV2Classifier(num_classes=NUM_CLASSES, freeze_backbone=True)
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)

    test_loss = running_loss / total
    test_acc = (np.array(all_preds) == np.array(all_labels)).mean()

    print("\n" + "=" * 60)
    print("MOBILENETV2 TEST EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%\n")

    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        output_dict=True,
    )
    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
    )

    print("Classification Report:")
    print(report_text)

    cm = confusion_matrix(all_labels, all_preds).tolist()

    results = {
        "model_name": "MobileNetV2",
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "per_class_metrics": report_dict,
        "confusion_matrix": cm,
        "class_names": class_names,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved evaluation results to {RESULTS_PATH}")


if __name__ == "__main__":
    evaluate()
