"""
config.py

Hyperparameter configuration for the ResNet50 transfer-learning model.
All training parameters are centralised here so they can be documented
in the report and reproduced by other team members.

Two-phase training strategy
----------------------------
Phase 1 (Feature Extraction): Backbone frozen, only the new classification
    head is trained.  A higher learning rate (1e-3) is safe because the
    gradients only flow through a single Linear layer.
Phase 2 (Fine-tuning):  layer4 of the backbone is unfrozen and trained
    alongside the head at a lower learning rate (5e-5 backbone, 1e-4 head)
    so the pretrained features are refined without being destroyed.
"""

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
NUM_CLASSES = 6
IMG_SIZE = 224           # Must match preprocessing/preprocess.py
DROPOUT = 0.4            # Dropout before the classifier head

# ---------------------------------------------------------------------------
# Phase 1 — Head-only training (backbone frozen)
# ---------------------------------------------------------------------------
PHASE1_LR = 1e-3
PHASE1_EPOCHS = 8
PHASE1_PATIENCE = 3      # Early-stop if val loss stalls for 3 epochs

# ---------------------------------------------------------------------------
# Phase 2 — Fine-tuning (layer4 + head unfrozen)
# ---------------------------------------------------------------------------
PHASE2_LR_HEAD = 1e-4    # Higher LR for the randomly-initialised head
PHASE2_LR_BACKBONE = 5e-5  # Lower LR for pretrained layer4 weights
PHASE2_EPOCHS = 15
PHASE2_PATIENCE = 5

# ---------------------------------------------------------------------------
# Shared training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4      # L2 regularisation in AdamW
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 2   # Reduce LR after 2 epochs without improvement
NUM_WORKERS = 2          # DataLoader workers (fits 16 GB RAM)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = "models/resnet50/checkpoints"
PHASE1_BEST_PATH = "models/resnet50/checkpoints/best_resnet50_phase1.pt"
PHASE2_BEST_PATH = "models/resnet50/checkpoints/best_resnet50_phase2.pt"
