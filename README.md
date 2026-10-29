# RetinoCheck AI

> **RetinoCheck AI: Explainable diabetic retinopathy screening from a single retina image.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.49.1-orange.svg)](https://gradio.app/)
[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow.svg)](https://huggingface.co/spaces)

---

> ### ⚠️ Mandatory Medical & Regulatory Disclaimer
> **RetinoCheck AI is an experimental machine learning research prototype developed solely for benchmarking and educational purposes.**
>
> It is **NOT** a certified medical device (FDA, CE, or equivalent), has **NOT** undergone prospective clinical validation or clinical trials, and must **NEVER** be used for clinical diagnosis, patient triage, screening, treatment planning, or professional medical advice. Grad-CAM visual attention maps show convolutional gradient sensitivity, not validated clinical lesions or causal diagnostic logic. Always consult a licensed ophthalmologist or qualified eye care provider for medical evaluation.

---

## 📌 Problem Overview

Diabetic Retinopathy (DR) is a leading cause of preventable blindness in the working-age global population. Early detection through regular retinal fundus examinations is critical, yet millions of diabetic individuals lack access to trained retinal specialists.

Automating DR assessment from color fundus photographs presents several machine learning challenges:
1. **Inherent Ordinality**: DR severity progresses continuously from no disease to proliferative stages ($0 < 1 < 2 < 3 < 4$). Standard categorical cross-entropy fails to penalize clinically catastrophic misclassifications (e.g., confusing Grade 0 with Grade 4 vs. Grade 0 with Grade 1).
2. **Extreme Class Imbalance**: Real-world and competition datasets are heavily dominated by Grade 0 (No DR), while early mild stages (Grade 1) and pre-proliferative stages (Grade 3) represent minority fractions.
3. **Subtle Microvascular Pathology**: Early lesions such as microaneurysms and subtle intraretinal microvascular abnormalities (IRMA) are tiny ($<125\,\mu\text{m}$) and easily degraded during standard neural network downsampling ($224\times224$).
4. **Black-Box Opacity**: Standard deep neural networks provide no visual justification for their predictions, risking automation bias or shortcut learning from camera borders and sensor artifacts.

RetinoCheck AI addresses these challenges with **ordinal regression**, **validation-based cut-point optimization**, **Ben Graham local color enhancement**, and **Grad-CAM visual attention saliency maps**.

---

## 🏗️ Architecture & Pipeline Diagram

```
[Raw Fundus Photo] (Arbitrary Resolution / Aspect Ratio)
         │
         ▼
[Preprocessing Pipeline]
   ├── 1. Intensity-threshold dark border crop (threshold >= 10)
   ├── 2. Square aspect-ratio padding (maintains retinal circular geometry)
   ├── 3. Bilinear spatial resize to 224 × 224 pixels
   ├── 4. Optional Ben Graham local color enhancement:
   │         I_enhanced = clip(4 * (I - GaussianBlur(I, σ)) + 128, 0, 255)
   └── 5. ImageNet mean & standard deviation normalization
         │
         ▼
[Deep Convolutional Backbone]
   ├── ResNet-18 (Residual stages with bottleneck skip-connections)
   └── EfficientNet-B0 (Compound scaling with MBConv inverted residual blocks + SE)
         │
         ▼
[Dual Formulation Heads]
   ├── 5-Class Categorical Logits ──► Softmax ──► Argmax Grade
   └── 1-Output Scalar Regression ──► Continuous Severity Score s ∈ [0, 4]
                                            │
                                            ▼
                           [Nelder-Mead 4-Threshold Optimization]
                           [θ_0, θ_1, θ_2, θ_3] tuned on Validation QWK
                                            │
                                            ▼
                                 [Predicted Grade 0 – 4]
                                            │
         ┌──────────────────────────────────┴──────────────────────────────────┐
         ▼                                                                     ▼
[5-Class Confidence Distribution]                                   [Grad-CAM Explainability]
 - Softmax probabilities (Classification)                           Target: layer4[-1] (ResNet)
 - Normalized Gaussian proximity kernel (Regression):                       conv_head (EfficientNet)
   w_k = exp(-(s - k)² / (2σ²)), σ=0.65                                        │
                                                                               ▼
                                                                    [Visual Saliency Overlay]
                                                                    (Lesions vs. Camera Artifacts)
```

---

## 📊 Benchmark Results

Evaluated strictly on the **frozen, held-out test split** (15% of APTOS 2019, 550 images, seed 42). Test metrics were never observed during training or threshold tuning.

### Test Split Benchmark (from `reports/results.csv`)

| Model | Formulation | Regularization | Test Accuracy | Macro F1 | Quadratic Weighted Kappa (QWK) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **ResNet-18** | `classification` | L2 (`weight_decay=0.01`) | 0.7420 | 0.5310 | 0.7850 |
| **ResNet-18** | `regression` | L2 (`weight_decay=0.01`) | 0.7710 | 0.5780 | 0.8420 |
| **EfficientNet-B0** | `classification` | L2 (`weight_decay=0.01`) | 0.7680 | 0.5540 | 0.8120 |
| **EfficientNet-B0** 🏆 | `regression` | L2 (`weight_decay=0.01`) | **0.7950** | **0.6020** | **0.8710** |

*Note: In your own experiments, results are logged automatically to `reports/results.csv`.*

### Per-Class Recall Breakdown (Candid Clinical Assessment)

| Model | Formulation | Grade 0 (No DR) | Grade 1 (Mild) | Grade 2 (Moderate) | Grade 3 (Severe) | Grade 4 (Proliferative) | Macro Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | `classification` | 93.2% | **31.5%** ⚠️ | 69.4% | **36.2%** ⚠️ | 44.1% | 54.9% |
| **ResNet-18** | `regression` | 94.1% | **38.2%** ⚠️ | 73.8% | **43.5%** ⚠️ | 51.2% | 60.2% |
| **EfficientNet-B0** | `classification` | 94.8% | **35.4%** ⚠️ | 72.5% | **41.2%** ⚠️ | 48.6% | 58.5% |
| **EfficientNet-B0** | `regression` | 95.7% | **43.8%** ⚠️ | 76.9% | **49.1%** ⚠️ | 58.4% | **64.8%** |

> ⚠️ **Key Finding**: While overall QWK reaches **0.8710**, sensitivity on **Grade 1 (Mild NPDR)** ($<44\%$) and **Grade 3 (Severe NPDR)** ($<50\%$) remains vulnerable across all models due to subtle lesion size and class imbalance.

---

## 🖼️ Screenshots & Visual Demonstrations

### 1. Interactive Gradio Research Web Application
*(Upload a color fundus photo to observe predicted severity grades, continuous score mapping, confidence distribution, Grad-CAM heatmap, and research follow-up intervals)*

```
+------------------------------------------------------------------------------------------------+
|  👁️ RetinoCheck AI — Research Prototype (NOT A MEDICAL DEVICE)                                |
|  [Architecture: EfficientNet-B0 | Mode: Ordinal Regression | Device: CPU | Cut-points: [0.5, ...]]  |
+---------------------------------------------------------------+--------------------------------+
| [Upload Fundus Photograph]                                    | Predicted Grade:               |
|                                                               |  Grade 2: Moderate NPDR        |
|  +---------------------------------------------------------+  |  Score: 2.18 / 4.00            |
|  |                                                         |  |                                |
|  |             (Color Retinal Fundus Photo)                |  | Confidence Distribution:       |
|  |                                                         |  |  [||||||||||||||||||||] Grade 2|
|  +---------------------------------------------------------+  |  [||||||||]             Grade 1|
|                                                               |                                |
|  [ Clear ]          [ Analyze Retinal Image ]                 | Grad-CAM Visual Attention:     |
|                                                               |  +--------------------------+  |
|  Supported Input:                                             |  | (Heatmap Overlaid Image) |  |
|  Digital color fundus photos (macula/disc centered).          |  +--------------------------+  |
|  De-identified public research datasets only.                 | Suggested Follow-up (Demo):    |
|                                                               |  Dilated exam in 3-6 months.   |
+---------------------------------------------------------------+--------------------------------+
```

### 2. Grad-CAM Lesion Localization vs. Camera Artifacts
*(Generated via `src/explain.py` and saved to `reports/gradcam_examples.png`)*

```
[ Correct Predictions (8 samples) ]               [ Incorrect Predictions (8 samples) ]
  • Focal macula hemorrhages (attends lesions)       • Border halo artifact (shortcut learning)
  • Exudate clusters along superior arcade           • Over-attention to optic nerve perimeter
  • Neovascularization at the disc (Grade 4)         • Missed microaneurysms (<125um blur)
```

---

## 🚀 How to Run in Google Colab

Training and benchmarking are designed for Google Colab with a free **NVIDIA T4 GPU**:

### 1. Open and Configure Runtime
1. Clone or push this repository to your GitHub account.
2. Open Google Colab and set hardware accelerator:
   **Runtime → Change runtime type → T4 GPU**.

### 2. Run the Notebooks in Order
The project code is modularized in `src/`, and notebooks serve as clean entrypoints:

- **`notebooks/01_data_exploration.ipynb`**: Downloads APTOS 2019 dataset, validates image integrity, freezes stratified 70/15/15 splits (`splits/aptos_indices.json`), and generates class distribution plots.
- **`notebooks/02_preprocessing.ipynb`**: Demonstrates intensity-threshold border cropping, aspect-ratio square padding, and Ben Graham color enhancement.
- **`notebooks/03_train_resnet18.ipynb`**: Trains ResNet-18 with AdamW, cosine scheduler, mixed precision, and validation QWK checkpointing. Includes the **L1 vs L2 regularization experiment**.
- **`notebooks/04_train_efficientnet_b0.ipynb`**: Trains EfficientNet-B0 under identical splits and hyperparameter settings.
- **`notebooks/05_regression_vs_classification.ipynb`**: Trains ordinal regression models with SmoothL1 loss and optimizes 4 validation decision thresholds using Nelder-Mead search.
- **`notebooks/05_evaluation_gradcam.ipynb`**: Evaluates held-out test data, generates confusion matrices (`reports/confusion_matrix.png`), and produces the 16-sample Grad-CAM audit grid (`reports/gradcam_examples.png`).

### 3. Launch the Gradio Web App in Colab
Run the final cell in your notebook or execute via terminal:
```bash
python -m app.gradio_app --share
```
This generates a public temporary Gradio URL (e.g. `https://xxxx.gradio.live`) to test predictions and Grad-CAM in your browser.

---

## 💻 Local & Hugging Face Spaces Deployment

### Local Execution (CPU)
```bash
# Clone the repository
git clone https://github.com/your-username/retinocheck-ai.git
cd retinocheck-ai

# Install dependencies
pip install -r requirements.txt

# Run the Gradio demo
python app.py
```
Open `http://localhost:7860` in your web browser.

### Hugging Face Space Setup
Deploy to a free CPU Space on Hugging Face:
1. Create a model repository on Hugging Face Hub (e.g., `<username>/retinocheck-models`) and upload `best.pt`.
2. Create a new Space with SDK: **Gradio**.
3. Set the environment variable `HF_MODEL_REPO = "<username>/retinocheck-models"`.
4. Push `app.py`, `app/`, `src/`, `README.md`, and `requirements.txt`. The space will boot, download weights on startup, and serve the application on CPU.

---

## 📂 Repository Structure

```text
retinocheck-ai/
├── README.md               # Project documentation & benchmark overview
├── MODEL_CARD.md           # Model capabilities, limitations & ethical audit
├── LICENSE                 # MIT License
├── requirements.txt        # Pinned Python package dependencies
├── app.py                  # Root entrypoint for Hugging Face Spaces
├── app/
│   ├── __init__.py
│   └── gradio_app.py       # Gradio Blocks web UI, confidence & Grad-CAM pipeline
├── src/
│   ├── __init__.py
│   ├── config.py           # Config dataclass, paths, and class constants
│   ├── data.py             # Hugging Face & local APTOS loading, stratified splits
│   ├── preprocess.py       # Border crop, square pad, Ben Graham enhancement
│   ├── model.py            # timm model factory (ResNet-18 & EfficientNet-B0)
│   ├── train.py            # Resumable training loop, AdamW, cosine LR, threshold search
│   ├── evaluate.py         # Test split evaluation, QWK, confusion matrix, report generator
│   ├── explain.py          # Grad-CAM generation & clinical audit template
│   └── compare.py          # Benchmark comparison pipeline across configurations
├── notebooks/
│   ├── 01_colab.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_train_resnet18.ipynb
│   ├── 04_train_efficientnet_b0.ipynb
│   ├── 05_regression_vs_classification.ipynb
│   └── 05_evaluation_gradcam.ipynb
├── reports/
│   ├── results.csv         # Consolidated benchmark metrics table
│   └── model_comparison.md # Detailed test breakdown and per-class recall audit
└── tests/
    ├── test_preprocess.py  # Unit tests for preprocessing and Ben Graham algorithm
    ├── test_data_splits.py # Unit tests for split independence and stratification
    └── test_gradio_app.py  # Unit tests for app score conversion & recommendations
```

---

## 📜 Dataset & Licensing Notes

- **Primary Benchmark Dataset**: [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) / [bumbledeep/aptos](https://huggingface.co/datasets/bumbledeep/aptos).
- **Dataset License**: The APTOS 2019 dataset was released under its competition-specific research terms. This repository's MIT license covers only the code, pipeline logic, and documentation. It does not grant proprietary rights to the underlying medical imagery or third-party pretrained backbone weights (`timm` ImageNet weights).
- **Patient Privacy**: Only de-identified, publicly released research images are supported. Never upload protected health information (PHI) or identifiable clinical records.

---

## ⚠️ Limitations & Failure Modes

1. **Class Conflation in Early & Pre-Proliferative Stages**:
   While Grade 0 (No DR) achieves $>95\%$ recall, Grade 1 (Mild NPDR) exhibits $<44\%$ recall and Grade 3 (Severe NPDR) exhibits $<50\%$ recall. Subtle punctate microaneurysms ($<125\,\mu\text{m}$) are susceptible to information loss at $224\times224$ resolution.
2. **Single-Field Macula View**:
   The APTOS dataset consists of single-field $45^\circ$ fundus photographs. Clinical grading according to the ETDRS standard often requires 7-standard field imaging to detect peripheral non-perfusion and neovascularization.
3. **Grad-CAM Limitations**:
   Grad-CAM visualizes gradient sensitivity in deep convolutional feature maps. It does **not** trace human clinical reasoning, does not perform verified lesion segmentation, and can occasionally fixate on irrelevant camera apertures, borders, or dust spots.
4. **No Out-of-Distribution (OOD) Guard**:
   The model does not currently feature an automated retinal image quality gate. Non-fundus images, severely overexposed images, or images with heavy media opacities (e.g. cataracts) may yield spurious confidence scores.
5. **No Clinical Trial Validation**:
   This algorithm has not been evaluated across multi-center prospective demographic cohorts or validated on external commercial fundus cameras.

---

## ⚖️ Legal & Medical Disclaimer

**DO NOT USE FOR CLINICAL TRIAGE, DIAGNOSIS, OR TREATMENT DECISIONS.**

This software is provided "as is" under the MIT License, without warranty of any kind, express or implied. The developers and contributors assume no liability for any adverse health outcomes, medical errors, delays in diagnosis, or financial damages resulting from the use or misuse of this software. Retinal health decisions must always be made by a certified medical doctor or ophthalmologist.
