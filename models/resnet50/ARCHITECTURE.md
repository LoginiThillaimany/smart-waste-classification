# ResNet50 Model Architecture & Defense Notes

## 1. Overview
ResNet50 is a 50-layer deep Convolutional Neural Network (CNN) architecture introduced by He et al. (2015). It addresses the **degrading accuracy / vanishing gradient problem** in ultra-deep networks through **Residual Learning (skip connections)**.

In our Smart Waste Classification benchmark, ResNet50 serves as the primary **deep transfer learning baseline**, leveraging features pretrained on ImageNet (`IMAGENET1K_V2`) and adapting them to a 6-class waste image taxonomy.

---

## 2. Key Architecture Components

### A. Bottleneck Building Blocks
Unlike standard 3×3 convolutional blocks, ResNet50 uses **3-layer Bottleneck Blocks**:
1. **1×1 Conv**: Reduces feature depth (e.g., 256 → 64 channels) to lower computational cost.
2. **3×3 Conv**: Performs spatial feature extraction on compressed channels.
3. **1×1 Conv**: Restores feature depth (e.g., 64 → 256 channels).

### B. Skip Connections (Residual Mapping)
Instead of trying to learn an unassisted target mapping \( H(x) \), each block learns a residual mapping \( F(x) = H(x) - x \).
The final output is:
\[
y = F(x) + x
\]
During backpropagation, gradients flow directly through the identity shortcut (\( +x \)), enabling deep feature learning without vanishing gradients.

### C. Layer Structure & Channels
- **Conv1 + MaxPool**: 7×7 convolution (stride 2) + 3×3 MaxPool → 64 channels.
- **Layer 1**: 3 Bottleneck blocks (256 channels).
- **Layer 2**: 4 Bottleneck blocks (512 channels).
- **Layer 3**: 6 Bottleneck blocks (1024 channels).
- **Layer 4**: 3 Bottleneck blocks (2048 channels).
- **Classification Head**: Global Average Pooling → Dropout(0.4) → Linear(2048, 6).

---

## 3. Two-Phase Transfer Learning Strategy

| Phase | Unfrozen Layers | Learning Rates | Epochs | Rationale |
|---|---|---|---|---|
| **Phase 1: Feature Extraction** | Only `fc` Head | `1e-3` (Head) | 8 (patience=3) | Protect pretrained ImageNet weights while adapting top classification head to waste classes. |
| **Phase 2: Fine-Tuning** | `layer4` + `fc` Head | `5e-5` (Backbone), `1e-4` (Head) | 15 (patience=5) | Fine-tune high-level semantic representations in `layer4` using discriminative learning rates to avoid catastrophic forgetting. |

---

## 4. Key Viva Questions & Answers

**Q1: Why use ResNet50 instead of a custom CNN?**
> *Answer:* Custom CNNs train features from scratch and require massive data. ResNet50 brings pretrained representations from 1.2M ImageNet images (edges, textures, shapes), giving higher accuracy and faster convergence even under severe class imbalance.

**Q2: What is the purpose of the 1×1 convolutions in the bottleneck block?**
> *Answer:* They act as dimension reduction and expansion layers. The first 1×1 reduces channel dimension before the 3×3 convolution, and the second 1×1 expands it back, dramatically reducing FLOPs while maintaining representation power.

**Q3: What are Discriminative Learning Rates?**
> *Answer:* Pretrained lower layers represent general feature detectors (edges/textures) that need minimal tuning (`5e-5`), while the newly initialised classification head requires a higher learning rate (`1e-4`) to learn task-specific decision boundaries.
