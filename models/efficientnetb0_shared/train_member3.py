"""
train_member3.py

EfficientNetB0 — Member 3's 1/3 contribution:
Fine-Tuning, Full Evaluation, and Computational Efficiency Benchmark.

Member 3 Responsibilities:
    1. Unfreeze top feature extraction blocks of EfficientNetB0 for fine-tuning.
    2. Train fine-tuned model with reduced learning rate.
    3. Run final evaluation on unseen test dataset.
    4. Benchmark inference latency (ms/image), parameter count, and model size.
    5. Save results to models/efficientnetb0_shared/results_member3.json.

Usage:
    python models/efficientnetb0_shared/train_member3.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.efficientnetb0_shared.efficientnet_model import EfficientNetB0Classifier
from preprocessing.preprocess import get_loaders

# Configuration for Member 3's Fine-Tuning Block
NUM_CLASSES = 6
BATCH_SIZE = 16
LEARNING_RATE = 1e-4     # Reduced LR for fine-tuning pretrained weights
WEIGHT_DECAY = 1e-4
EPOCHS = 15
EARLY_STOP_PATIENCE = 5
RANDOM_SEED = 42

CHECKPOINT_DIR = Path("models/efficientnetb0_shared/checkpoints")
MEMBER1_MODEL_PATH = CHECKPOINT_DIR / "best_efficientnet_member1.pt"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_efficientnet_member3.pt"
HISTORY_PATH = CHECKPOINT_DIR / "training_history_member3.json"
RESULTS_PATH = Path("models/efficientnetb0_shared/results_member3.json")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def measure_inference_efficiency(model, test_loader, device, num_runs=50):
    """Measure model size, parameter counts, and per-image inference latency."""
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Measure latency
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    total_time = time.time() - start
    avg_latency_ms = (total_time / num_runs) * 1000

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "avg_latency_ms_per_image": avg_latency_ms,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }


def train_and_evaluate():
    """Member 3 training, evaluation and efficiency benchmarking routine."""
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[Member 3] Using device: {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Load shared DataLoaders
    train_loader, val_loader, test_loader, class_names = get_loaders(batch_size=BATCH_SIZE)

    # Instantiate EfficientNetB0
    model = EfficientNetB0Classifier(num_classes=NUM_CLASSES, freeze_backbone=False)

    # Unfreeze last 3 feature blocks for fine-tuning
    model.unfreeze_backbone(num_layers_to_unfreeze=3)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    no_improve = 0

    print("\n[Member 3] Starting EfficientNetB0 Fine-Tuning Block...")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        dt = time.time() - t0
        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({dt:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_names": class_names,
                },
                BEST_MODEL_PATH,
            )
            print(f"  --> Best EfficientNetB0 Member 3 checkpoint saved! (Val Acc: {val_acc*100:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)

    # ---------------------------------------------------------------------------
    # Final Test Set Evaluation & Computational Efficiency Benchmark
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[Member 3] Final Test Set Evaluation & Efficiency Benchmark")
    print("=" * 70)

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    running_loss, total = 0.0, 0

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

    report_dict = classification_report(all_labels, all_preds, target_names=class_names, digits=4, output_dict=True)
    report_text = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    efficiency = measure_inference_efficiency(model, test_loader, device)
    model_size_mb = BEST_MODEL_PATH.stat().st_size / (1024 * 1024)

    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Inference Latency: {efficiency['avg_latency_ms_per_image']:.2f} ms / image")
    print(f"Model File Size  : {model_size_mb:.2f} MB")
    print("\nClassification Report:\n", report_text)

    results = {
        "member": 3,
        "model_name": "EfficientNetB0 (Fine-Tuned)",
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "model_size_mb": model_size_mb,
        "inference_latency_ms": efficiency["avg_latency_ms_per_image"],
        "total_parameters": efficiency["total_parameters"],
        "trainable_parameters": efficiency["trainable_parameters"],
        "per_class_metrics": report_dict,
        "confusion_matrix": cm,
        "class_names": class_names,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved Member 3 EfficientNetB0 results to {RESULTS_PATH}")


if __name__ == "__main__":
    train_and_evaluate()
