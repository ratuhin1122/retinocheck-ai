# Research reports

Save each model's artifacts in separate folders. Evaluation writes `test_metrics.json`, `test_predictions.json` and `test_confusion_matrix.png`. Training history and split indices live alongside Drive checkpoints. Preserve the runtime's package versions and config with reports. Do not commit private images, credentials or large checkpoints.

Report QWK first, followed by accuracy, macro F1, grade 0–4 recall, class counts and confusion matrix. State whether data came from the resized HF mirror or original Kaggle images. Record limitations of image-level splitting and any duplicate/patient audit.

## Benchmark Results (`reports/results.csv`)

Comparison across backbones (ResNet-18 vs. EfficientNet-B0) and training formulations (5-class classification vs. ordinal regression with 4 validation-optimized thresholds):

- Columns: `model, mode, regularization, accuracy, macro_f1, qwk`
- Generated automatically by `src.compare` and `notebooks/05_regression_vs_classification.ipynb`.
