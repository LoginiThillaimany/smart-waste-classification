"""
config.py

Hyperparameter configuration for the Custom CNN model.
All training parameters are centralised here so they can be documented
in the report and reproduced by other team members.
"""

# ---------------------------------------------------------------------------
# Model Architecture Settings
# ---------------------------------------------------------------------------
NUM_CLASSES = 6          # Number of target waste categories (Plastic, Metal, Glass, etc.)
IMG_SIZE = 224           # Input image size in pixels (224x224, matches preprocessing)

# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 16            # Number of images processed per forward/backward pass
LEARNING_RATE = 1e-3       # Initial learning rate for AdamW optimizer
WEIGHT_DECAY = 1e-4        # L2 regularization factor to prevent overfitting
EPOCHS = 50                # Maximum number of training epochs
EARLY_STOP_PATIENCE = 7    # Stop training if validation loss does not improve for 7 epochs
LR_SCHEDULER_PATIENCE = 3  # Reduce learning rate if validation loss stalls for 3 epochs
LR_SCHEDULER_FACTOR = 0.5  # Halve the learning rate when plateau is reached

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42           # Fixed seed to ensure reproducible training runs

# ---------------------------------------------------------------------------
# File and Directory Paths
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = "models/custom_cnn/checkpoints"                   # Folder to store saved weights & logs
BEST_MODEL_PATH = "models/custom_cnn/checkpoints/best_custom_cnn.pt" # Path to save the best model weights
