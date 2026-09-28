"""
resnet50_model.py

Pretrained ResNet50 with a custom 6-class classification head for
waste image classification.

Architecture overview (for viva / report)
-----------------------------------------
ResNet50 is a 50-layer deep CNN built from *bottleneck blocks*, each
containing three convolutions (1×1, 3×3, 1×1) with a *skip / residual
connection* that adds the block's input directly to its output.  This
shortcut solves the vanishing-gradient problem: gradients can flow
through the identity path during back-propagation, allowing the
network to be very deep without degrading.

Layer structure (frozen during phase 1):
    conv1  → bn1 → relu → maxpool        # 7×7, stride 2
    layer1  (3 bottleneck blocks,   64 channels)
    layer2  (4 bottleneck blocks,  128 channels)
    layer3  (6 bottleneck blocks,  256 channels)
    layer4  (3 bottleneck blocks,  512 channels)  ← unfrozen in phase 2
    avgpool → flatten → fc (2048 → NUM_CLASSES)   ← always trainable

We replace the original 1000-class fc head with:
    Dropout(DROPOUT) → Linear(2048, NUM_CLASSES)

Using IMAGENET1K_V2 weights (acc@1 80.858) gives strong low- and mid-level
feature representations (edges, textures, shapes) that transfer well to
waste images, even though the domain is different from ImageNet.

Helper functions
----------------
- freeze_backbone(model): freezes everything except the fc head and sets
  frozen BatchNorm layers to eval mode so running stats are not updated.
- unfreeze_layers(model, layers): selectively unfreezes named groups
  (e.g. "layer4") for fine-tuning in phase 2.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Classifier(nn.Module):
    """
    ResNet50 transfer-learning model for 6-class waste classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 6).
    dropout : float
        Dropout probability before the final Linear layer.
    """

    def __init__(self, num_classes: int = 6, dropout: float = 0.4):
        super().__init__()

        # Load pretrained backbone (IMAGENET1K_V2 is the best available)
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Replace the 1000-class head with our custom head
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (B, 3, 224, 224).

        Returns
        -------
        torch.Tensor
            Raw logits of shape (B, num_classes).
            No softmax — CrossEntropyLoss applies LogSoftmax internally.
        """
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def freeze_backbone(model: ResNet50Classifier) -> None:
    """
    Freeze the entire backbone so only the fc head is trainable.

    Also sets frozen BatchNorm layers to eval mode so their running mean/var
    statistics are not corrupted by the new domain during phase 1.
    """
    for name, param in model.backbone.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

    # Keep frozen BN layers in eval mode
    for module in model.backbone.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            # Only freeze BN if all its params are frozen
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()


def unfreeze_layers(model: ResNet50Classifier, layers: tuple = ("layer4",)) -> None:
    """
    Selectively unfreeze named layer groups for fine-tuning.

    Parameters
    ----------
    layers : tuple of str
        Names of backbone children to unfreeze (e.g. ("layer4",) or
        ("layer3", "layer4")).
    """
    for name, param in model.backbone.named_parameters():
        for layer_name in layers:
            if name.startswith(layer_name):
                param.requires_grad = True

    # Also set the unfrozen BN layers back to train mode
    for layer_name in layers:
        layer = getattr(model.backbone, layer_name, None)
        if layer is not None:
            for module in layer.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.train()


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = ResNet50Classifier(num_classes=6, dropout=0.4)

    # Check parameter counts before freezing
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Before freeze — Total: {total:,} | Trainable: {trainable:,}")

    # Phase 1: freeze backbone
    freeze_backbone(model)
    trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 1 (head only) — Trainable: {trainable_p1:,}")

    # Phase 2: unfreeze layer4
    unfreeze_layers(model, layers=("layer4",))
    trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 2 (layer4+head) — Trainable: {trainable_p2:,}")

    # Forward pass test
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # Expected: (2, 6)
