# RetinoCheck AI model card

## Status

Untrained project scaffold; no trained weights or measured performance included. Two independent five-class classifiers: timm `resnet18` and `efficientnet_b0`, initialized with their default ImageNet pretrained weights. Five logits trained with class-weighted cross entropy. Prediction is softmax argmax; no ordinal threshold tuning or calibration is performed.

## Intended use

Educational experiments and research on fundus-image classification. Grades map to the dataset's labels: 0 no DR, 1 mild, 2 moderate, 3 severe, 4 proliferative DR. These are annotation categories, not a diagnosis for an uploaded image. Never a medical device; not for screening, diagnosis, treatment, patient triage, or medical advice.

## Data and processing

Default source: https://huggingface.co/datasets/bumbledeep/aptos . Alternate: original Kaggle APTOS 2019 labeled training images. Record source, immutable revision, counts and split fingerprint for every trained artifact. The hosted copy has 224×224 images. RGB conversion, dark-border cropping, square padding, resizing, optional Ben Graham-style enhancement and ImageNet normalization are shared across splits and inference. Enhancement defaults to `clip(4 * (image - GaussianBlur(image)) + 128, 0, 255)`, with sigma equal to output size divided by 30, preserving black background. Training additionally uses horizontal/vertical flips, small rotations and light brightness/contrast jitter. Optional lossless PNG caching stores only deterministic preprocessing. No evidence is yet provided that enhancement improves performance.

Stratified image-level 70/15/15 splits with seed 42. Patient grouping is unavailable; independence and absence of duplicates are unverified. Dataset population, camera mix, annotation quality and representation across demographic groups are insufficiently characterized here. Review original dataset documentation and permissions before use.

## Evaluation

Select checkpoints using validation QWK, then evaluate the selected model on held-out test images. Report QWK, accuracy, macro F1, recall for grades 0–4, confusion matrix, sample counts, software versions and seeds. Results: **not yet measured**. No external or prospective evaluation, uncertainty intervals, calibration or subgroup evaluation is supplied. A small image-level internal test set cannot support clinical claims.

## Limitations

Class imbalance and annotation noise may bias predictions. Domain shift, artifacts, camera variation, missing retinal detail and low-quality or non-fundus inputs may produce misleading scores. No validated abstention or out-of-distribution detection exists. Grad-CAM is a sensitivity visualization, not validated lesion evidence. Scores are uncalibrated. No clinical safety, fairness, privacy, or regulatory assessment has been performed.

## Release checklist for research weights

Record training date, chosen backbone, source revision, preprocessing, hyperparameters, completed epochs, best validation epoch/QWK, all held-out metrics, duplicate/patient overlap audit, compute and library versions, data/weight usage terms, and any known failures. Keep this card with exported weights and update its untrained status only after actual measurement.
