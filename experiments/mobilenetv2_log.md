# Experiment Log: MobileNetV2 Model (Member 3 Primary Model)

**Date**: 28 September 2026  
**Primary Owner**: Member 3  
**Architecture**: MobileNetV2 (Pretrained on ImageNet1K with Inverted Residual Bottleneck blocks)  
**Task**: 6-Class Smart Waste Classification  

---

## 1. Executive Summary

MobileNetV2 achieved an outstanding **97.30% overall test accuracy** on the locked unseen test dataset of 2,334 images. Thanks to depthwise separable convolutions and inverted residual bottleneck shortcuts, MobileNetV2 achieves superior predictive performance while requiring only **2.88 million total parameters**, making it an ideal lightweight candidate for edge deployment (e.g., embedded smart bin devices).

---

## 2. Experimental Configuration & Hyperparameters

| Configuration Parameter | Value | Justification |
|---|---|---|
| **Backbone Architecture** | MobileNetV2 (`weights=DEFAULT`) | Pretrained features from ImageNet transfer learning |
| **Input Resolution** | 224 × 224 | Standard ImageNet resolution; compatible with `preprocessing/preprocess.py` |
| **Batch Size** | 16 | Optimal trade-off between gradient stability and memory utilization |
| **Optimizer** | AdamW | Adaptive momentum with L2 weight decay (`weight_decay=1e-4`) |
| **Initial Learning Rate** | 1e-3 | Fast convergence for top classification head |
| **LR Scheduler** | `ReduceLROnPlateau` | Factor=0.5, Patience=2 (reduced LR when val loss plateaus) |
| **Epoch Budget** | 25 Epochs | Managed with Early Stopping (Patience = 5) |
| **Loss Function** | Cross-Entropy Loss | Standard for multi-class classification |
| **Random Seed** | 42 | Guaranteed reproducibility across runs |
| **Hardware Device** | Apple Silicon GPU (MPS) | Accelerated tensor computation |

---

## 3. Training & Validation Progress Highlights

- **Epoch 1**: Train Acc: 83.27% | Val Acc: **92.99%** | Val Loss: 0.2112
- **Epoch 4**: Train Acc: 90.55% | Val Acc: **94.66%** | Val Loss: 0.1626
- **Epoch 8**: Train Acc: 92.65% | Val Acc: **95.48%** | Val Loss: 0.1207
- **Epoch 12**: LR reduced to `5e-4` | Val Acc: **96.26%** | Val Loss: 0.1181
- **Epoch 17**: Train Acc: 94.91% | Val Acc: **96.51%** | Val Loss: 0.1050
- **Epoch 21**: LR reduced to `2.5e-4` | Val Acc: **96.51%** | Val Loss: 0.1043
- **Epoch 25**: Train Acc: 95.61% | Val Acc: **96.94%** | Val Loss: **0.0974**

*Total Training Duration*: **46.42 minutes** (25 Epochs completed).

---

## 4. Final Evaluation Results on Test Dataset

Evaluation conducted on **2,334 unseen test images**:

- **Test Loss**: `0.0815`
- **Test Accuracy**: `97.30%`
- **Macro F1-Score**: `95.06%`
- **Weighted F1-Score**: `97.28%`

### Per-Class Detailed Breakdown

| Class Name | Precision | Recall | F1-Score | Support Count | Key Observations |
|---|---|---|---|---|---|
| **Glass** | 93.85% | 95.71% | 94.77% | 303 | High recall; slight confusion with plastic bottles |
| **Metal** | 92.04% | 89.66% | 90.83% | 116 | Minor recall reduction due to reflections in crushed cans |
| **Organic** | 97.96% | 96.64% | 97.30% | 149 | Excellent distinction of food and plant waste |
| **Paper_Cardboard** | 97.93% | 97.26% | 97.59% | 292 | Outstanding detection of flat and box materials |
| **Plastic** | 94.26% | 87.79% | 90.91% | 131 | Transparent containers occasionally overlap with clear glass |
| **Residual** | 98.60% | 99.33% | 98.96% | 1,343 | Near-perfect performance on general waste items |

---

## 5. Artifacts & Generated Figures

All generated plots are archived under `results/figures/`:
1. **Accuracy Curves**: `results/figures/mobilenetv2_accuracy.png`
2. **Loss Curves**: `results/figures/mobilenetv2_loss.png`
3. **Confusion Matrix Heatmap**: `results/figures/mobilenetv2_confusion_matrix.png`

---

## 6. Member 3 Self-Reflection & Next Contributions

1. MobileNetV2 proves that lightweight depthwise separable architectures can achieve top-tier classification performance (>97% accuracy) with significantly fewer parameters than heavier networks.
2. Next task: Collaborate with Member 1 & Member 2 on the shared EfficientNetB0 benchmark evaluation and lead Section 7 & 8 (Model Comparison & Critical Analysis) for the final project report.
