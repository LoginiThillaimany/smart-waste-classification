# EfficientNetB0 — Member 1 Experiment Log

## Contribution Summary

**Member:** Member 1
**Responsibility (1/3):** Data pipeline validation, initial training block (frozen backbone), experiment logging.

## Phase: Feature Extraction (Frozen Backbone)

### Setup
- **Model:** EfficientNetB0 pretrained on ImageNet
- **Backbone:** Frozen (all convolutional layers)
- **Trainable layers:** Custom classifier head only
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Loss function:** CrossEntropyLoss
- **Batch size:** 16
- **Max epochs:** 20
- **Early stopping patience:** 5
- **Random seed:** 42

### Data Pipeline Validation

| Check | Status |
|-------|--------|
| Batch shape matches (B, 3, 224, 224) | Pass `(32, 3, 224, 224)` |
| Pixel range is normalised (approx -2 to 3 for ImageNet stats) | Pass `[-2.118, 2.640]` |
| Train/val/test sizes match split configuration | Pass (Train: 10,857 \| Val: 2,324 \| Test: 2,334) |
| Class labels match across all loaders | Pass (6 classes: Glass, Metal, Organic, Paper_Cardboard, Plastic, Residual) |

### Training Results

| Metric | Value |
|--------|-------|
| Epochs trained | 2 |
| Best validation loss | 0.7121 |
| Best validation accuracy | 76.98% |
| Training time | 171.0s (2.85 min) |
| Trainable parameters | 658,950 |
| Total parameters | 4,666,498 |

### Observations

- **Rapid Convergence:** The frozen ImageNet backbone provides rich, pre-trained visual representations, allowing the custom classification head to reach 76.98% validation accuracy in just 2 epochs (outperforming the Custom CNN baseline of 62.2% test accuracy).
- **Efficiency:** Training only 658,950 parameters (14.1% of total parameters) significantly reduces computational burden on CPU while maintaining high gradient stability.
- **Accuracy Ceiling:** In feature extraction mode, early plateau is expected around 78–82% without fine-tuning convolutional feature maps to the specific domain textures of waste (e.g. crushed plastic vs metal glare).
- **Recommendations for Member 2 (Fine-Tuning Phase):**
  1. Unfreeze the top 3–5 inverted residual blocks (`unfreeze_backbone(num_layers_to_unfreeze=3)`).
  2. Reduce learning rate to `1e-4` or `5e-5` to avoid destructive updates to pretrained weights.
  3. Use cosine annealing or step decay scheduler.

## Handoff Notes for Member 2

- Best checkpoint saved at: `models/efficientnetb0_shared/checkpoints/best_efficientnet_member1.pt`
- Training history at: `models/efficientnetb0_shared/checkpoints/training_history_member1.json`
- Suggested next steps: Load `best_efficientnet_member1.pt`, unfreeze top blocks, and proceed with Phase 2 fine-tuning.
