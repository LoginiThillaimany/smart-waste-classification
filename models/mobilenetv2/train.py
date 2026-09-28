"""
train.py

Training pipeline for the MobileNetV2 model.

Features:
    - Full training loop with train/validation phases per epoch.
    - Early stopping based on validation loss.
    - Learning rate scheduling via ReduceLROnPlateau.
    - Best-model checkpointing (saves when val loss improves).
    - Training history logged to JSON.
    - Reproducible via fixed random seeds.

Usage:
    python models/mobilenetv2/train.py
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path so we can import shared modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.mobilenetv2.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CHECKPOINT_DIR,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    HISTORY_PATH,
    LEARNING_RATE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    NUM_CLASSES,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from models.mobilenetv2.mobilenetv2_model import MobileNetV2Classifier
from preprocessing.preprocess import get_loaders


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (imgs, labels) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
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

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion, device, max_batches=None):
    """Evaluate model on validation loader. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train(max_batches=None):
    """Main training routine for MobileNetV2."""
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    # Get data loaders
    train_loader, val_loader, test_loader, class_names = get_loaders(batch_size=BATCH_SIZE)

    print(f"Loaded dataset: {len(class_names)} classes: {class_names}")

    # Instantiate MobileNetV2
    model = MobileNetV2Classifier(num_classes=NUM_CLASSES, freeze_backbone=True)
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
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
    }

    best_val_loss = float("inf")
    no_improve_epochs = 0
    start_time = time.time()

    print("\nStarting MobileNetV2 Training...")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, max_batches)
        val_loss, val_acc = validate(model, val_loader, criterion, device, max_batches)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rates"].append(current_lr)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | LR: {current_lr:.2e}"
        )

        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_names": class_names,
                },
                BEST_MODEL_PATH,
            )
            print(f"  --> Best model saved! (Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%)")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch} (no val loss improvement for {EARLY_STOP_PATIENCE} epochs).")
                break

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Training completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

    # Save training history JSON
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Saved training history to {HISTORY_PATH}")


if __name__ == "__main__":
    train()
