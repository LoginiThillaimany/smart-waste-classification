"""
config.py

Hyperparameter configuration for the Custom CNN model.
All training parameters are centralised here so they can be documented
in the report and reproduced by other team members.
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
EPOCHS = 50
EARLY_STOP_PATIENCE = 7  # Stop if val loss does not improve for N epochs
LR_SCHEDULER_PATIENCE = 3  # Reduce LR after N epochs without improvement
LR_SCHEDULER_FACTOR = 0.5  # Multiply LR by this factor on plateau

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = "models/custom_cnn/checkpoints"
BEST_MODEL_PATH = "models/custom_cnn/checkpoints/best_custom_cnn.pt"
