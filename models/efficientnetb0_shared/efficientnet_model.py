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

        # Load pretrained EfficientNetB0
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the classifier head
        # EfficientNetB0 original: classifier = Sequential(Dropout(0.2), Linear(1280, 1000))
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits."""
        return self.backbone(x)

    def unfreeze_backbone(self, num_layers_to_unfreeze: int = 0) -> None:
        """
        Unfreeze the last `num_layers_to_unfreeze` blocks of the backbone
        for fine-tuning. If 0, unfreeze everything.
        """
        all_children = list(self.backbone.features.children())
        if num_layers_to_unfreeze == 0:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze last N blocks
            for child in all_children[-num_layers_to_unfreeze:]:
                for param in child.parameters():
                    param.requires_grad = True


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = EfficientNetB0Classifier(num_classes=6, freeze_backbone=True)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # Expected: (2, 6)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters (frozen backbone): {trainable:,}")

    model.unfreeze_backbone(num_layers_to_unfreeze=3)
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (last 3 blocks unfrozen): {trainable_after:,}")
