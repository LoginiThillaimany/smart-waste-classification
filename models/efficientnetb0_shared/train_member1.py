"""
train_member1.py

EfficientNetB0 — Member 1's 1/3 contribution.

Responsibilities:
    1. Validate that the shared data pipeline works with EfficientNetB0.
    2. Run the initial training block (frozen backbone, feature extraction only).
    3. Record experiment logs and observations.

This script trains the model with the backbone frozen (feature extraction mode)
to establish a baseline. Members 2 and 3 will handle fine-tuning and final evaluation.

Usage:
    python models/efficientnetb0_shared/train_member1.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Add project root directory to path to import shared modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.efficientnetb0_shared.efficientnet_model import EfficientNetB0Classifier
from preprocessing.preprocess import get_loaders

# ---------------------------------------------------------------------------
# Configuration for Member 1's Training Block
# ---------------------------------------------------------------------------
NUM_CLASSES = 6         # 6 waste categories
BATCH_SIZE = 16         # Batch size for DataLoader
LEARNING_RATE = 1e-3    # Learning rate for training the new classifier head
WEIGHT_DECAY = 1e-4     # L2 regularization weight
EPOCHS = 20             # Maximum epochs for initial feature-extraction phase
EARLY_STOP_PATIENCE = 5 # Stop if validation loss does not improve for 5 epochs
RANDOM_SEED = 42        # Fixed seed for reproducibility

# Directories and paths for saving checkpoints and logs
CHECKPOINT_DIR = Path("models/efficientnetb0_shared/checkpoints")
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_efficientnet_member1.pt"
HISTORY_PATH = CHECKPOINT_DIR / "training_history_member1.json"


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training results."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    """Run one epoch of training over all batches. Returns (avg_loss, accuracy)."""
    model.train()         # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (imgs, labels) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()               # Clear prior gradients
        outputs = model(imgs)               # Forward pass
        loss = criterion(outputs, labels)   # Calculate loss
        loss.backward()                     # Backpropagate gradients
        optimizer.step()                    # Update classifier weights

        # Track batch statistics
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion, device, max_batches=None):
    """Run validation loop without computing gradients. Returns (avg_loss, accuracy)."""
    model.eval()          # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)               # Forward pass
            loss = criterion(outputs, labels)   # Calculate loss

            # Track batch statistics
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def train(epochs=None, max_batches=None, batch_size=None):
    """Member 1's training block: frozen backbone, feature extraction."""
    set_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = epochs if epochs is not None else EPOCHS
    b_size = batch_size if batch_size is not None else BATCH_SIZE

    print(f"Using device: {device}")
    print(f"{'='*60}")
    print(f"EfficientNetB0 — Member 1: Initial Training (Frozen Backbone, {num_epochs} epochs)")
    print(f"{'='*60}")

    # Load preprocessed datasets
    train_loader, val_loader, test_loader, class_names = get_loaders(batch_size=b_size, num_workers=0)
    print(f"Classes: {class_names}")

    # Validate data pipeline: inspect first batch shapes and value ranges
    sample_batch, sample_labels = next(iter(train_loader))
    print(f"\n--- Data Pipeline Validation ---")
    print(f"Batch shape: {sample_batch.shape}")  # Should be (B, 3, 224, 224)
    print(f"Label shape: {sample_labels.shape}")
    print(f"Pixel range: [{sample_batch.min():.3f}, {sample_batch.max():.3f}]")
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Val set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    print("Data pipeline validated successfully [OK]\n")

    # Initialize EfficientNetB0 with frozen backbone (feature extraction)
    model = EfficientNetB0Classifier(num_classes=NUM_CLASSES, freeze_backbone=True).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters (frozen backbone): {trainable:,}")

    # Set loss function and AdamW optimizer only for trainable parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )

    # Ensure checkpoint folder exists
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Dictionary to track training metrics per epoch
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    # Training loop across epochs
    for epoch in range(1, num_epochs + 1):
        # Step 1: Train for 1 epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, max_batches)

        # Step 2: Validate for 1 epoch
        val_loss, val_acc = validate(model, val_loader, criterion, device, max_batches)

        # Step 3: Record metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch:>2}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Step 4: Checkpoint best weights when validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("  [+] Best model saved")
        else:
            patience_counter += 1
            # Step 5: Check early stopping patience
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    total_time = time.time() - start_time

    # Record summary metadata
    history["total_training_time_sec"] = total_time
    history["best_val_loss"] = best_val_loss
    history["epochs_trained"] = len(history["train_loss"])
    history["device"] = device
    history["phase"] = "feature_extraction"
    history["member"] = "Member 1"

    # Save training history to JSON
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nMember 1 training complete in {total_time:.1f}s")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"History saved to {HISTORY_PATH}")


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 (Member 1)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches per epoch")
    args = parser.parse_args()

    # Launch training
    train(epochs=args.epochs, max_batches=args.max_batches, batch_size=args.batch_size)
