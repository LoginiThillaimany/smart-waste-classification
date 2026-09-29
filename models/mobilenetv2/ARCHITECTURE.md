# MobileNetV2 Architecture Specification

## Overview

MobileNetV2 is an efficient, light-weight deep neural network architecture designed for resource-constrained vision applications. It uses depthwise separable convolutions combined with inverted residual blocks and linear bottlenecks.

## Key Design Features

1. **Inverted Residual Blocks**:
   - Standard residual blocks connect layers with high channel counts. Inverted residual blocks instead create shortcut connections between narrow bottleneck layers.
   - Expansion ratio of 6 expands input channels before applying 3x3 depthwise convolution, followed by 1x1 projection back to low dimensions.

2. **Linear Bottlenecks**:
   - Non-linear activation functions (ReLU6) are omitted in the final layer of each bottleneck block to prevent destroying non-linear manifold information in lower dimensions.

3. **Pretrained Transfer Learning**:
   - Pretrained on ImageNet (1,000 classes, 1.28M images).
   - Custom classifier head designed specifically for 6-class waste sorting (`Glass`, `Metal`, `Organic`, `Paper_Cardboard`, `Plastic`, `Residual`).

## Custom Classifier Head Specification

```
Input Tensor: [Batch, 3, 224, 224]
     │
[MobileNetV2 Feature Extractor Backbone] -> Output: [Batch, 1280, 7, 7]
     │
[Adaptive Avg Pooling] -> Output: [Batch, 1280]
     │
[Dropout (p=0.3)]
     │
[Linear (1280 -> 512)]
     │
[ReLU Activation]
     │
[Dropout (p=0.2)]
     │
[Linear (512 -> 6)] -> Output Logits: [Batch, 6]
```

## Hyperparameter Summary

| Parameter | Value |
|---|---|
| Input Resolution | 224 × 224 |
| Batch Size | 16 |
| Initial Learning Rate | 1e-3 |
| Optimizer | AdamW (`weight_decay=1e-4`) |
| LR Scheduler | `ReduceLROnPlateau(factor=0.5, patience=2)` |
| Epochs | 25 (with Early Stopping patience=5) |
