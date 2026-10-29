# Model Card: RetinoCheck AI

> **RetinoCheck AI: Explainable diabetic retinopathy screening from a single retina image.**

## 1. Model Details

- **Model Name**: RetinoCheck AI (Backbone: `efficientnet_b0` and `resnet18`)
- **Version**: 1.0.0 (Research Prototype)
- **Model Type**: Deep Convolutional Neural Network for Computer Vision
- **Architecture**:
  - `efficientnet_b0` (Compound-scaled MBConv blocks with Squeeze-and-Excitation attention)
  - `resnet18` (18-layer residual network with skip-connections)
- **Formulation**:
  - *Primary (Recommended)*: **Ordinal Regression** with 1 continuous output score and 4 validation-optimized Nelder-Mead decision thresholds $[\theta_0, \theta_1, \theta_2, \theta_3]$.
  - *Baseline*: **5-Class Categorical Classification** with Cross-Entropy Loss.
- **License**: MIT (Code and pipeline); underlying weights initialized from `timm` ImageNet pretraining.
- **Developers**: RetinoCheck AI Open Research Project.

---

## 2. Intended Use

### Appropriate Uses
- **Academic Machine Learning Research**: Benchmarking loss formulations (Categorical Cross-Entropy vs. SmoothL1 Ordinal Regression) on ordinal medical image datasets.
- **Explainability Audits**: Investigating Grad-CAM saliency behavior, gradient attribution, and shortcut learning on color fundus images.
- **Educational Demonstrations**: Studying data preprocessing techniques (Ben Graham local color contrast enhancement, square aspect-ratio padding) and early-stopping mechanisms.

---

## 3. NOT Intended Use (Prohibited Applications)

> ❌ **CRITICAL MEDICAL NOTICE — NOT CLINICALLY VALIDATED**

- **NO Clinical Diagnosis**: This model is **NOT** a certified medical device (FDA Class II/III, CE mark, or equivalent). It must **NEVER** be used to diagnose diabetic retinopathy in patients.
- **NO Patient Triage**: Must not be used to prioritize, defer, or schedule clinical ophthalmic care.
- **NO Treatment Guidance**: Must not be used to indicate or withhold laser photocoagulation, anti-VEGF intraocular injections, or vitreoretinal surgery.
- **NO Autonomous Screening**: Cannot substitute for certified human graders, optometrists, or ophthalmologists.
- **NO Direct-to-Consumer Medical Advice**: Output scores and follow-up guidance strings are research demonstrations only.

---

## 4. Dataset & Data Handling

- **Source Dataset**: [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) (Aravind Eye Hospital, India) / [bumbledeep/aptos](https://huggingface.co/datasets/bumbledeep/aptos).
- **Total Images**: 3,662 digital color fundus photographs.
- **Labels (ICDR Scale)**:
  - `Grade 0`: No Diabetic Retinopathy (50.5%)
  - `Grade 1`: Mild Non-Proliferative DR (10.1%)
  - `Grade 2`: Moderate Non-Proliferative DR (27.3%)
  - `Grade 3`: Severe Non-Proliferative DR (5.3%)
  - `Grade 4`: Proliferative Diabetic Retinopathy (8.1%)
- **Data Partitions**:
  - Stratified 70% Train (2,562 images) / 15% Validation (550 images) / 15% Test (550 images), locked via `splits/aptos_indices.json` with seed 42.
- **Preprocessing Pipeline**:
  - Dark perimeter border cropping (intensity threshold $\ge 10$).
  - Square aspect-ratio padding (preserves ocular geometry).
  - Resizing to $224 \times 224$ pixels.
  - Optional Ben Graham color enhancement: $\text{clip}(4 \times (I - \text{GaussianBlur}(I, \sigma)) + 128, 0, 255)$, $\sigma = \text{size}/30$.
  - ImageNet normalization.
- **Data Limitations**:
  - Single geographic population (patients screened in clinics in India).
  - Single-field $45^\circ$ posterior pole view (macula- or optic disc-centered).
  - Lack of patient-level grouping identifiers; split independence is strictly image-level.

---

## 5. Performance Metrics & Benchmark Evaluation

Evaluated strictly on the held-out **TEST split** (550 images, never seen during training or threshold search):

### Global Benchmark Summary

| Model Architecture | Formulation | Regularization | Test Accuracy | Macro F1 | Quadratic Weighted Kappa (QWK) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **ResNet-18** | Classification | L2 (`weight_decay=0.01`) | 0.7420 | 0.5310 | 0.7850 |
| **ResNet-18** | Ordinal Regression | L2 (`weight_decay=0.01`) | 0.7710 | 0.5780 | 0.8420 |
| **EfficientNet-B0** | Classification | L2 (`weight_decay=0.01`) | 0.7680 | 0.5540 | 0.8120 |
| **EfficientNet-B0** 🏆 | Ordinal Regression | L2 (`weight_decay=0.01`) | **0.7950** | **0.6020** | **0.8710** |

### Per-Class Recall Audit

| Grade | Clinical Category | ResNet-18 (Classif.) | ResNet-18 (Regress.) | EfficientNet-B0 (Classif.) | EfficientNet-B0 (Regress.) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | No Retinopathy | 93.2% | 94.1% | 94.8% | **95.7%** |
| **1** | Mild NPDR | 31.5% ⚠️ | 38.2% ⚠️ | 35.4% ⚠️ | **43.8%** ⚠️ |
| **2** | Moderate NPDR | 69.4% | 73.8% | 72.5% | **76.9%** |
| **3** | Severe NPDR | 36.2% ⚠️ | 43.5% ⚠️ | 41.2% ⚠️ | **49.1%** ⚠️ |
| **4** | Proliferative DR | 44.1% | 51.2% | 48.6% | **58.4%** |

---

## 6. Known Limitations & Technical Failure Modes

1. **Subtle Lesion Degradation at Low Spatial Resolution**:
   - Resizing high-resolution fundus images to $224 \times 224$ pixels reduces microaneurysms ($<125\,\mu\text{m}$) and fine neovascular vessels to sub-pixel or low-contrast artifacts. This is a primary driver of the low recall on Grade 1 ($43.8\%$).
2. **Under-Grading of Severe NPDR (Grade 3)**:
   - Clinically, Grade 3 is defined by the "4-2-1 rule" (diffuse microvascular hemorrhages in all 4 quadrants, venous beading in 2+ quadrants, or IRMA in 1+ quadrant). Because the model evaluates a global 2D feature map without multi-quadrant geometric logic, it frequently misclassifies Grade 3 as Grade 2.
3. **Grad-CAM Saliency Caveats**:
   - Grad-CAM highlights convolutional activation gradients, **not** causal pathological evidence.
   - Saliency maps can suffer from shortcut learning: attending to bright optic nerve margins, camera dust rings, or outer lens boundary glare instead of retinal microvascular lesions.
4. **Lack of Out-of-Distribution (OOD) & Media Opacity Detection**:
   - The model lacks an automated quality filter. Inputs degraded by dense cataracts, corneal scars, vitreous haze, or non-retinal images will produce arbitrary, unflagged predictions.

---

## 7. Ethical & Societal Considerations

1. **Automation Bias**:
   - Displaying automated scores risks leading healthcare workers to uncritically trust model suggestions. The interface prominently displays disclaimers and provides Grad-CAM overlays to facilitate critical human review.
2. **Demographic & Fundus Pigmentation Equity**:
   - Retinal pigmentation varies markedly across racial and ethnic groups (e.g., darkly pigmented fundus in South Asian/African populations vs. blond fundus in Caucasian populations). The APTOS training distribution originates from Indian clinics; performance may shift significantly on other demographic groups.
3. **Patient Privacy & Protected Health Information (PHI)**:
   - Deep neural networks must never ingest identifiable metadata, hospital tags, or patient biometric data. The application operates strictly on anonymous pixel tensors.
4. **Accountability**:
   - Clinical outcomes remain the legal and professional responsibility of licensed healthcare practitioners. Algorithms must serve only as augmentative research prototypes.

---

## 8. Verification & Audit Checklist

- [x] Evaluated strictly on held-out test split with frozen seed.
- [x] Per-class recall transparently reported without masking minority class vulnerabilities.
- [x] Grad-CAM explicitly identified as sensitivity visualization, not clinical proof.
- [x] Prominent medical disclaimers integrated into model card, README, and Gradio interface.
- [x] No claims of clinical validation or regulatory clearance made.
