# Custom CNN — Architecture Design Document

## Overview

The Custom CNN is the **baseline model** in the four-model comparison. It is built entirely from scratch (no pretrained weights) to establish how well a simple convolutional architecture can learn waste-classification features from this dataset alone.

## Architecture Diagram

```
Input (224 × 224 × 3)
       │
       ▼
┌──────────────────────────┐
│ Block 1                  │
│ Conv2D(3→32, 3×3, pad=1) │
│ BatchNorm2D(32)          │
│ ReLU                     │
│ MaxPool2D(2×2)           │
│ Output: 112 × 112 × 32  │
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Block 2                  │
│ Conv2D(32→64, 3×3, pad=1)│
│ BatchNorm2D(64)          │
│ ReLU                     │
│ MaxPool2D(2×2)           │
│ Output: 56 × 56 × 64    │
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Block 3                  │
│ Conv2D(64→128, 3×3, pad=1│
│ BatchNorm2D(128)         │
│ ReLU                     │
│ MaxPool2D(2×2)           │
│ Output: 28 × 28 × 128   │
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Block 4                  │
│ Conv2D(128→256,3×3,pad=1)│
│ BatchNorm2D(256)         │
│ ReLU                     │
│ MaxPool2D(2×2)           │
│ Output: 14 × 14 × 256   │
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ AdaptiveAvgPool2D(1×1)   │
│ Output: 1 × 1 × 256     │
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Flatten → 256            │
│ FC(256→512) → ReLU       │
│ Dropout(0.5)             │
│ FC(512→256) → ReLU       │
│ Dropout(0.3)             │
│ FC(256→6) [logits]       │
└──────────────────────────┘
```

## Design Justifications

### Convolutional Blocks

| Decision | Justification |
|----------|---------------|
| **4 blocks** | Sufficient depth to learn hierarchical features (edges → textures → parts → objects) without excessive parameters for a ~15k-image dataset. |
| **3×3 kernels with padding=1** | Standard receptive-field expansion; padding preserves spatial dimensions before pooling. |
| **Progressive channels (32→64→128→256)** | Doubling channels at each block is a well-established pattern (VGG, ResNet) that allows the network to learn increasingly complex feature maps. |
| **BatchNorm after Conv** | Stabilises training by normalising activations, permits higher learning rates, and acts as mild regularisation. |
| **ReLU activation** | Simple, effective, avoids vanishing gradients. Used throughout for consistency. |
| **MaxPool2D(2×2)** | Halves spatial dimensions, reducing computation and introducing translational invariance. |

### Global Pooling

| Decision | Justification |
|----------|---------------|
| **AdaptiveAvgPool2D(1×1)** | Replaces the large flatten-after-conv approach (which would produce a 14×14×256 = 50,176-dim vector). Reduces to 256 values, cutting parameters by ~98% and making the model input-size-agnostic. |

### Classifier Head

| Decision | Justification |
|----------|---------------|
| **FC(512) → FC(256) → FC(6)** | Two hidden layers provide enough non-linear capacity for 6-class classification without excessive parameters. |
| **Dropout(0.5) then Dropout(0.3)** | Aggressive dropout on the first FC layer prevents co-adaptation of features; lighter dropout on the second layer maintains expressiveness. |
| **No Softmax in forward()** | PyTorch `CrossEntropyLoss` applies `LogSoftmax` internally for numerical stability. Softmax is applied only during inference/evaluation. |

### Training Configuration

| Hyperparameter | Value | Justification |
|---|---|---|
| Optimizer | AdamW | Combines Adam's adaptive learning rates with proper weight decay (L2 regularisation). |
| Learning rate | 1e-3 | Standard starting point for Adam-family optimisers on image tasks. |
| Weight decay | 1e-4 | Mild L2 regularisation to reduce overfitting. |
| Batch size | 16 | Balances memory usage and gradient noise; the dataset is moderate-sized. |
| Epochs | 50 (max) | Upper bound; early stopping will terminate training sooner if validation loss plateaus. |
| Early stopping patience | 7 | Allows the model to recover from temporary validation dips before stopping. |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) | Halves the learning rate if validation loss stalls, helping the model converge to a better minimum. |
| Random seed | 42 | Fixed for reproducibility across all experiments. |

### Loss Function

- **CrossEntropyLoss** — standard for multi-class classification. Combines `LogSoftmax` and `NLLLoss`. If class imbalance is severe (identified in EDA), class weights can be added.

## Expected Strengths and Limitations

### Strengths
- Full control over architecture design.
- No dependency on pretrained weights (ImageNet).
- Smaller model size and faster inference compared to deep transfer-learning models.

### Expected Limitations
- Likely lower accuracy than transfer-learning models (ResNet50, MobileNetV2, EfficientNetB0) because it learns features from scratch on only ~15k images.
- More susceptible to overfitting without the feature-learning head start that pretrained models provide.
- No residual connections — deep gradient flow relies entirely on BatchNorm.

These limitations are expected and form the basis of the critical comparison in the report.
