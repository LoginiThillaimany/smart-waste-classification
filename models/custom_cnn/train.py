"""
train.py

Training pipeline for the Custom CNN model.

Features:
    - Full training loop with train/validation phases per epoch.
    - Early stopping based on validation loss (patience from config).
    - Learning rate scheduling via ReduceLROnPlateau.
    - Best-model checkpointing (saves only when val loss improves).
    - Training history saved to JSON for later plotting.
    - Reproducible via fixed random seeds.

Usage:
    python models/custom_cnn/train.py
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path so we can import shared modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.custom_cnn.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CHECKPOINT_DIR,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    NUM_CLASSES,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from models.custom_cnn.custom_cnn_model import CustomCNN
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
    """Run validation. Returns (avg_loss, accuracy)."""
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


def train(epochs=None, max_batches=None, batch_size=None):
    """Main training function."""
    set_seed(RANDOM_SEED)

    num_epochs = epochs if epochs is not None else EPOCHS
    b_size = batch_size if batch_size is not None else BATCH_SIZE

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, _, class_names = get_loaders(batch_size=b_size, num_workers=0)
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Model
    model = CustomCNN(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )

    # Checkpoint directory
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Training Custom CNN for up to {num_epochs} epochs")
    print(f"Early stopping patience: {EARLY_STOP_PATIENCE}")
    if max_batches:
        print(f"Max batches per epoch: {max_batches}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, max_batches)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, max_batches)

        # Get current LR
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch:>3}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        # Step scheduler
        scheduler.step(val_loss)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  [+] Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")

        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save training history
    history["total_training_time_sec"] = total_time
    history["best_val_loss"] = best_val_loss
    history["epochs_trained"] = len(history["train_loss"])
    history["device"] = device

    history_path = Path(CHECKPOINT_DIR) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    return model, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Custom CNN")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches per epoch")
    args = parser.parse_args()
    train(epochs=args.epochs, max_batches=args.max_batches, batch_size=args.batch_size)
