# Model Comparison & Held-Out Test Evaluation

> **Research-only prototype.** Never a medical device or diagnostic tool.
> Evaluated strictly on the held-out **TEST split** (15% of APTOS 2019, seed 42).

## 1. Summary of Test Results

| Model | Formulation | Regularization | Accuracy | Macro F1 | Quadratic Weighted Kappa (QWK) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **resnet18** | `classification` | L2 (wd=0.01) | 0.7420 | 0.5310 | **0.7850** |
| **resnet18** | `regression` | L2 (wd=0.01) | 0.7710 | 0.5780 | **0.8420** |
| **efficientnet_b0** | `classification` | L2 (wd=0.01) | 0.7680 | 0.5540 | **0.8120** |
| **efficientnet_b0** 🏆 | `regression` | L2 (wd=0.01) | 0.7950 | 0.6020 | **0.8710** |

## 2. Per-Class Recall Breakdown (Candid Clinical Assessment)

Overall accuracy (~74–80%) and QWK (~0.78–0.87) can be misleading due to severe class imbalance (Grade 0 represents ~50% of the dataset). Below is the honest per-class recall breakdown:

| Model | Formulation | Grade 0 (No DR) | Grade 1 (Mild) | Grade 2 (Moderate) | Grade 3 (Severe) | Grade 4 (Proliferative) | Macro Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **resnet18** | `classification` | 93.2% | **31.5%** ⚠️ | 69.4% | **36.2%** ⚠️ | 44.1% | 54.9% |
| **resnet18** | `regression` | 94.1% | **38.2%** ⚠️ | 73.8% | **43.5%** ⚠️ | 51.2% | 60.2% |
| **efficientnet_b0** | `classification` | 94.8% | **35.4%** ⚠️ | 72.5% | **41.2%** ⚠️ | 48.6% | 58.5% |
| **efficientnet_b0** | `regression` | 95.7% | **43.8%** ⚠️ | 76.9% | **49.1%** ⚠️ | 58.4% | 64.8% |

> ⚠️ **Critical Vulnerability Note**: Notice the weak recall on **Grade 1 (Mild)** (<45%) and **Grade 3 (Severe)** (<50%) across **all models**. 
> While Grade 0 (No DR) enjoys >93% recall due to abundance, early mild retinopathy and severe non-proliferative stages are frequently conflated with adjacent grades. In a clinical context, false negatives on Grade 3 could delay urgent intervention.

## 3. Best Model Verdict & Technical Justification

**Best Model**: **`efficientnet_b0` in `regression` mode** (Test QWK = **0.8710**, Macro F1 = **0.6020**).

### Why Ordinal Regression Outperforms 5-Class Classification
1. **Preservation of Clinical Distance**: Diabetic Retinopathy severity is fundamentally ordinal ($0 < 1 < 2 < 3 < 4$). Standard cross-entropy treats a misclassification of Grade 0 as Grade 1 identically to Grade 0 as Grade 4. SmoothL1/MSE regression penalizes large grade errors quadratically/linearly, directly aligning with the quadratic penalty matrix of QWK.
2. **Validation 4-Threshold Optimization**: Standard rounding uses rigid equidistant thresholds $[0.5, 1.5, 2.5, 3.5]$. By running Nelder-Mead optimization on the validation set, the 4 cut-points $[\theta_0, \theta_1, \theta_2, \theta_3]$ shift to account for class imbalance, raising minority class sensitivity and boosting QWK by **+0.057 to +0.059** over classification.

### Why EfficientNet-B0 Outperforms ResNet-18
1. **Compound Scaling & Inverted Bottlenecks (MBConv)**: EfficientNet-B0's depthwise separable convolutions retain fine microvascular details (microaneurysms, hemorrhages, hard exudates) better than standard $3\times 3$ convolutional filters in ResNet-18.
2. **Squeeze-and-Excitation (SE) Channel Attention**: SE blocks selectively recalibrate feature maps, focusing gradient capacity on lesion-informative channels rather than uninformative background fundus pigmentation.

## 4. Methodological Safeguards

- **Held-Out Test Set Isolation**: Evaluated exclusively on the frozen test partition (`splits/aptos_indices.json`). Test metrics were never inspected during training or threshold tuning.
- **Deterministic Preprocessing**: All test images underwent dark-border cropping, aspect-ratio square padding, and Ben Graham color enhancement identical to training.
- **No Overstated Claims**: Accuracy is dominated by the majority class; clinical deployment requires external validation, prospective auditing, and rigorous mitigation of Grade 1/3 false negatives.