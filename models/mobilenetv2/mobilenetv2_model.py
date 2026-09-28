"""
mobilenetv2_model.py

MobileNetV2 model for smart waste image classification.

Architecture:
    - MobileNetV2 backbone pretrained on ImageNet (inverted residual blocks).
    - Custom classification head tailored for 6 waste categories.
    - Support for feature extraction (frozen backbone) and fine-tuning.
"""

import torch
import torch.nn as nn
from torchvision import models


class MobileNetV2Classifier(nn.Module):
    """
    MobileNetV2 classifier for 6 waste sorting classes.

    Parameters
    ----------
    num_classes : int
        Number of target output classes (default 6).
    freeze_backbone : bool
        If True, freeze all backbone layers initially.
    """

    def __init__(self, num_classes: int = 6, freeze_backbone: bool = True):
        super().__init__()

        # Load pretrained MobileNetV2
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final classifier head
        # MobileNetV2 original classifier: Sequential(Dropout(0.2), Linear(1280, 1000))
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw output logits."""
        return self.backbone(x)

    def unfreeze_backbone(self, num_layers_to_unfreeze: int = 0) -> None:
        """
        Unfreeze backbone layers for fine-tuning.
        If num_layers_to_unfreeze == 0, unfreeze all layers.
        """
        if num_layers_to_unfreeze == 0:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            all_features = list(self.backbone.features.children())
            for child in all_features[-num_layers_to_unfreeze:]:
                for param in child.parameters():
                    param.requires_grad = True


if __name__ == "__main__":
    model = MobileNetV2Classifier(num_classes=6, freeze_backbone=True)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # Expected: (2, 6)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters (frozen backbone): {trainable:,}")
