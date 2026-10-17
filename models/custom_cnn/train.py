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

# Add project root directory to path to import configuration and preprocessing
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
    """Set random seeds to make training reproducible."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    """Run one training epoch over all batches. Returns (avg_loss, accuracy)."""
    model.train()         # Set model to training mode (enables dropout and batchnorm updates)
    running_loss = 0.0    # Accumulated loss across batches
    correct = 0           # Count of correct predictions
    total = 0             # Total images processed

    # Iterate over training batches
    for i, (imgs, labels) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        imgs, labels = imgs.to(device), labels.to(device)  # Move data to GPU or CPU

        optimizer.zero_grad()               # Reset gradients from previous step
        outputs = model(imgs)               # Forward pass: get model predictions
        loss = criterion(outputs, labels)   # Compute cross-entropy loss
        loss.backward()                     # Backward pass: compute gradients
        optimizer.step()                    # Update model weights

        # Track statistics
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)       # Class with highest predicted probability
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # Compute average loss and accuracy for this epoch
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion, device, max_batches=None):
    """Evaluate model performance on validation data. Returns (avg_loss, accuracy)."""
    model.eval()          # Set model to evaluation mode (disables dropout)
    running_loss = 0.0
    correct = 0
    total = 0

    # No gradients needed during validation (saves memory and computation)
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)               # Forward pass
            loss = criterion(outputs, labels)   # Compute loss

            # Track statistics
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Compute average validation loss and accuracy
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train(epochs=None, max_batches=None, batch_size=None):
    """Main training workflow: loads data, builds model, trains epochs, and saves best weights."""
    set_seed(RANDOM_SEED)  # Ensure reproducibility

    num_epochs = epochs if epochs is not None else EPOCHS
    b_size = batch_size if batch_size is not None else BATCH_SIZE

    # Select GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data using shared preprocessed splits
    train_loader, val_loader, _, class_names = get_loaders(batch_size=b_size, num_workers=0)
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Instantiate Custom CNN model and move it to device
    model = CustomCNN(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Define loss function, AdamW optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )

    # Ensure checkpoint folder exists
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    # Dictionary to record metrics for each epoch
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")  # Track lowest validation loss
    patience_counter = 0          # Count epochs without improvement for early stopping

    print(f"\n{'='*60}")
    print(f"Training Custom CNN for up to {num_epochs} epochs")
    print(f"Early stopping patience: {EARLY_STOP_PATIENCE}")
    if max_batches:
        print(f"Max batches per epoch: {max_batches}")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Main training loop across epochs
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Step 1: Run training phase
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, max_batches)

        # Step 2: Run validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, device, max_batches)

        # Step 3: Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Step 4: Record epoch metrics
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

        # Step 5: Adjust learning rate if validation loss plateaus
        scheduler.step(val_loss)

        # Step 6: Save model checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  [+] Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")

        # Step 7: Check early stopping condition
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Store final summary metrics in history
    history["total_training_time_sec"] = total_time
    history["best_val_loss"] = best_val_loss
    history["epochs_trained"] = len(history["train_loss"])
    history["device"] = device

    # Save training history to JSON for plotting curves
    history_path = Path(CHECKPOINT_DIR) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    return model, history


# ---------------------------------------------------------------------------
# Command Line Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Custom CNN")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches per epoch")
    args = parser.parse_args()

    # Start training with parsed arguments
    train(epochs=args.epochs, max_batches=args.max_batches, batch_size=args.batch_size)
