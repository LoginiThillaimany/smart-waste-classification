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

        # ---------- Convolutional feature extractor ----------
        self.features = nn.Sequential(
            # Block 1: 3 → 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32 → 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 64 → 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 128 → 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Collapse spatial dims to 1×1 regardless of input size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ---------- Classifier head ----------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (B, 3, H, W).

        Returns
        -------
        torch.Tensor
            Raw logits of shape (B, num_classes).
        """
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = CustomCNN(num_classes=6)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # Expected: (2, 6)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")
