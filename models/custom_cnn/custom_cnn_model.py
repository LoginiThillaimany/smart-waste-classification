"""
custom_cnn_model.py

Custom CNN architecture for waste image classification.
Built entirely from scratch (no pretrained weights) — this serves as the
baseline model in the four-model comparison.

Architecture overview:
    4 × (Conv2D → BatchNorm → ReLU → MaxPool2D)
    → AdaptiveAvgPool2D → Flatten
    → FC(512) → ReLU → Dropout(0.5)
    → FC(256) → ReLU → Dropout(0.3)
    → FC(num_classes)

Design rationale:
    - Four convolutional blocks progressively increase channel depth
      (32 → 64 → 128 → 256) to learn hierarchical features from edges
      to textures to object parts.
    - BatchNorm after each Conv stabilises training and allows higher
      learning rates.
    - MaxPool2D(2) halves spatial dimensions after each block, reducing
      computation while preserving the most activated features.
    - AdaptiveAvgPool2D(1) makes the architecture input-size-agnostic
      and reduces the flattened vector to 256 values regardless of the
      input resolution.
    - Two fully connected layers with ReLU provide non-linear decision
      capacity; Dropout (0.5 then 0.3) combats overfitting given the
      moderately-sized dataset (~15 k images).
    - No Softmax in forward() — PyTorch CrossEntropyLoss applies
      LogSoftmax internally, which is numerically more stable.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    """
    A 4-block convolutional neural network for 6-class waste classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 6 for the waste categories).
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()

        # ---------- Feature Extractor (4 Convolutional Blocks) ----------
        self.features = nn.Sequential(
            # Block 1: Extracts low-level features like edges and basic shapes (3 -> 32 channels)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3 input channels (RGB) to 32 feature maps
            nn.BatchNorm2d(32),                          # Normalizes batch activations for stable training
            nn.ReLU(inplace=True),                       # Non-linear activation function
            nn.MaxPool2d(kernel_size=2, stride=2),       # Downsamples spatial size by 2x (224 -> 112)

            # Block 2: Learns simple textures and corners (32 -> 64 channels)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # Downsamples spatial size by 2x (112 -> 56)

            # Block 3: Learns complex textures and object parts (64 -> 128 channels)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # Downsamples spatial size by 2x (56 -> 28)

            # Block 4: Learns high-level semantic patterns (128 -> 256 channels)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # Downsamples spatial size by 2x (28 -> 14)
        )

        # Global average pooling: reduces spatial dimensions to 1x1 regardless of input size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ---------- Classifier Head (Fully Connected Layers) ----------
        self.classifier = nn.Sequential(
            nn.Flatten(),                     # Flatten 2D feature map into 1D vector (256 elements)
            nn.Linear(256, 512),              # First dense layer: 256 -> 512 units
            nn.ReLU(inplace=True),            # Activation
            nn.Dropout(0.5),                  # Drops 50% neurons to prevent overfitting
            nn.Linear(512, 256),              # Second dense layer: 512 -> 256 units
            nn.ReLU(inplace=True),            # Activation
            nn.Dropout(0.3),                  # Drops 30% neurons
            nn.Linear(256, num_classes),      # Final output layer: outputs raw logits for 6 classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of images (Batch, Channels=3, Height=224, Width=224).

        Returns
        -------
        torch.Tensor
            Raw class logits (Batch, num_classes=6).
        """
        x = self.features(x)    # Pass through 4 conv blocks
        x = self.pool(x)        # Pool spatial dimensions to 1x1
        x = self.classifier(x)  # Compute class logits via FC layers
        return x


# ---------------------------------------------------------------------------
# Sanity Check & Architecture Verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Instantiate the model with 6 classes
    model = CustomCNN(num_classes=6)

    # Create dummy batch of 2 images (3 channels, 224x224 pixels)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)

    # Print output shape and parameter counts
    print(f"Output shape: {out.shape}")  # Expected: (2, 6)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")
