"""
efficientnet_model.py

EfficientNetB0 model for waste image classification.
This is the shared benchmark model — each team member contributes 1/3
of the training, tuning and analysis work.

Member 1 responsibility:
    - Model definition and data pipeline validation (this file).
    - Initial training block and experiment logging.

Architecture:
    - EfficientNetB0 backbone pretrained on ImageNet.
    - Original classifier head replaced with a custom head for 6 classes.
    - Backbone layers frozen initially, then optionally fine-tuned.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
from torchvision import models


class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNetB0 with a custom classification head for waste sorting.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 6).
    freeze_backbone : bool
        If True, freeze all backbone layers (feature extraction only).
        Set to False for fine-tuning.
    """

    def __init__(self, num_classes: int = 6, freeze_backbone: bool = True):
        super().__init__()

        # Load pretrained EfficientNet-B0 weights from ImageNet
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Freeze backbone parameters if requested (for initial feature extraction phase)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace default classifier head with custom layers for our 6 classes
        in_features = self.backbone.classifier[1].in_features  # 1280 features from backbone
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),              # Dropout to prevent overfitting
            nn.Linear(in_features, 512),  # Dense layer: 1280 -> 512
            nn.ReLU(inplace=True),        # Activation
            nn.Dropout(0.2),              # Dropout
            nn.Linear(512, num_classes),  # Final classification layer: 512 -> 6 classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — passes input through backbone and custom classifier."""
        return self.backbone(x)

    def unfreeze_backbone(self, num_layers_to_unfreeze: int = 0) -> None:
        """
        Unfreeze backbone blocks for fine-tuning.
        If num_layers_to_unfreeze is 0, all backbone layers are unfrozen.
        Otherwise, only the last N blocks are unfrozen.
        """
        all_children = list(self.backbone.features.children())
        if num_layers_to_unfreeze == 0:
            # Unfreeze all parameters across the backbone
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze parameters only in the top/last N feature blocks
            for child in all_children[-num_layers_to_unfreeze:]:
                for param in child.parameters():
                    param.requires_grad = True


# ---------------------------------------------------------------------------
# Sanity Check & Verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Create model instance with frozen backbone
    model = EfficientNetB0Classifier(num_classes=6, freeze_backbone=True)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # Expected: (2, 6)

    # Check total vs trainable parameters with backbone frozen
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters (frozen backbone): {trainable:,}")

    # Unfreeze the last 3 blocks and verify increased trainable parameters
    model.unfreeze_backbone(num_layers_to_unfreeze=3)
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (last 3 blocks unfrozen): {trainable_after:,}")
