"""
config.py

Hyperparameter configuration for the MobileNetV2 model.
All training parameters are centralised here for consistency and reproducibility.
"""

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
NUM_CLASSES = 6
IMG_SIZE = 224           # Must match preprocessing/preprocess.py

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4      # L2 regularisation in AdamW
EPOCHS = 25
EARLY_STOP_PATIENCE = 5  # Stop if val loss does not improve for N epochs
LR_SCHEDULER_PATIENCE = 2  # Reduce LR after N epochs without improvement
LR_SCHEDULER_FACTOR = 0.5  # Multiply LR by this factor on plateau

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = "models/mobilenetv2/checkpoints"
BEST_MODEL_PATH = "models/mobilenetv2/checkpoints/best_mobilenetv2.pt"
HISTORY_PATH = "models/mobilenetv2/checkpoints/training_history.json"
RESULTS_PATH = "models/mobilenetv2/results.json"
