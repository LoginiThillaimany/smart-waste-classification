"""
train.py

Two-phase training script for the ResNet50 model.

Features:
    - Phase 1: Train classification head while freezing the backbone.
    - Phase 2: Unfreeze layer4 of the backbone and fine-tune with discriminative learning rates.
    - Early stopping & ReduceLROnPlateau scheduler in both phases.
    - Detailed evaluation metric tracking & checkpoint saving.

Usage:
    python models/resnet50/train.py
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.resnet50.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DROPOUT,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    NUM_CLASSES,
    NUM_WORKERS,
    PHASE1_BEST_PATH,
    PHASE1_EPOCHS,
    PHASE1_LR,
    PHASE1_PATIENCE,
    PHASE2_BEST_PATH,
    PHASE2_EPOCHS,
    PHASE2_LR_BACKBONE,
    PHASE2_LR_HEAD,
    PHASE2_PATIENCE,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from models.resnet50.resnet50_model import (
    ResNet50Classifier,
    freeze_backbone,
    unfreeze_layers,
)
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


def run_phase(
    phase_name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs: int,
    patience: int,
    checkpoint_path: str,
    device: str,
    max_batches=None,
):
    """Execute training phase with early stopping and checkpointing."""
    print(f"\n{'='*60}")
    print(f" Starting {phase_name} ({epochs} max epochs, patience={patience})")
    print(f"{'='*60}")

    best_val_loss = float("inf")
    patience_counter = 0
    phase_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, max_batches)
        val_loss, val_acc = validate(model, val_loader, criterion, device, max_batches)

        current_lr = optimizer.param_groups[0]["lr"]

        phase_history["train_loss"].append(train_loss)
        phase_history["train_acc"].append(train_acc)
        phase_history["val_loss"].append(val_loss)
        phase_history["val_acc"].append(val_acc)
        phase_history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start
        print(
            f"[{phase_name}] Epoch [{epoch:>2}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [+] Saved best checkpoint ({checkpoint_path}) with val_loss={val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  [-] No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"Early stopping triggered for {phase_name} at epoch {epoch}.")
            break

    # Load best checkpoint weights from this phase
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return phase_history, best_val_loss


def train(epochs_p1=None, epochs_p2=None, max_batches=None, batch_size=None):
    """Execute two-phase transfer learning pipeline for ResNet50."""
    set_seed(RANDOM_SEED)

    b_size = batch_size if batch_size is not None else BATCH_SIZE
    e_p1 = epochs_p1 if epochs_p1 is not None else PHASE1_EPOCHS
    e_p2 = epochs_p2 if epochs_p2 is not None else PHASE2_EPOCHS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data loaders
    train_loader, val_loader, _, class_names = get_loaders(batch_size=b_size, num_workers=NUM_WORKERS)
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    # --- PHASE 1 ---
    model = ResNet50Classifier(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)
    freeze_backbone(model)

    p1_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_p1 = optim.AdamW(p1_params, lr=PHASE1_LR, weight_decay=WEIGHT_DECAY)
    scheduler_p1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p1, mode="min", factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
    )

    h1, best_val_p1 = run_phase(
        "Phase 1 (Head-only)",
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer_p1,
        scheduler_p1,
        e_p1,
        PHASE1_PATIENCE,
        PHASE1_BEST_PATH,
        device,
        max_batches,
    )

    # --- PHASE 2 ---
    unfreeze_layers(model, layers=("layer4",))

    # Discriminative learning rates: layer4 backbone vs fc head
    layer4_params = [p for name, p in model.backbone.named_parameters() if name.startswith("layer4") and p.requires_grad]
    head_params = [p for name, p in model.backbone.named_parameters() if "fc" in name and p.requires_grad]

    param_groups = [
        {"params": layer4_params, "lr": PHASE2_LR_BACKBONE},
        {"params": head_params, "lr": PHASE2_LR_HEAD},
    ]
    optimizer_p2 = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler_p2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p2, mode="min", factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
    )

    h2, best_val_p2 = run_phase(
        "Phase 2 (Fine-tuning layer4+head)",
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer_p2,
        scheduler_p2,
        e_p2,
        PHASE2_PATIENCE,
        PHASE2_BEST_PATH,
        device,
        max_batches,
    )

    total_time = time.time() - start_time

    # Combine history
    combined_history = {
        "phase1": h1,
        "phase2": h2,
        "best_val_loss_p1": best_val_p1,
        "best_val_loss_p2": best_val_p2,
        "total_training_time_sec": total_time,
        "device": device,
    }

    history_path = Path(CHECKPOINT_DIR) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(combined_history, f, indent=2)
    print(f"\nTraining complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Phase 1 best val loss: {best_val_p1:.4f} | Phase 2 best val loss: {best_val_p2:.4f}")
    print(f"Training history saved to {history_path}")

    return model, combined_history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train ResNet50 model")
    parser.add_argument("--epochs-p1", type=int, default=PHASE1_EPOCHS, help="Phase 1 max epochs")
    parser.add_argument("--epochs-p2", type=int, default=PHASE2_EPOCHS, help="Phase 2 max epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches per epoch (for quick testing)")
    args = parser.parse_args()

    train(
        epochs_p1=args.epochs_p1,
        epochs_p2=args.epochs_p2,
        max_batches=args.max_batches,
        batch_size=args.batch_size,
    )
