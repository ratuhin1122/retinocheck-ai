# RetinoCheck AI

Research prototype for five-class diabetic retinopathy severity grading from fundus images, using pretrained timm ResNet18 and EfficientNet-B0. **Never a medical device.** Outputs are experimental model predictions, not medical advice. Do not use for diagnosis, screening, or treatment decisions.

## Start in Google Colab

1. Push this repository to GitHub. Set `REPO_URL` in the notebook setup cell to your repository URL. The setup clones code into `/content/retinocheck-ai`.
2. Open `notebooks/01_colab.ipynb` in Colab. Select **Runtime → Change runtime type → T4 GPU**. Free GPU availability is not guaranteed.
3. Run setup, dataset verification, and training cells in order. Grant Drive mounting access when Colab prompts.
4. Train ResNet18 first. Change `MODEL` to `efficientnet_b0` and rerun the configuration and training cells for a separate experiment with identical splits.
5. Use validation metrics to select the model and settings, then run the held-out test cell once for the selected model. Test results must not guide further tuning.

Logic lives in `src/`; the notebook only performs environment setup and calls project functions. Defaults: 224 px, batch size 32, seed 42, learning rate 1e-4, weight decay 1e-2, 15 epochs, AdamW, training-only class-weighted cross entropy. Reduce batch size to 16 on a GPU memory error; start a separate run when changing settings. Runtime and accuracy are not promised. Class weights are optional and must be chosen using validation results.

## Dataset verification and fallback

On 2026-10-28 the [Hugging Face dataset viewer](https://huggingface.co/datasets/bumbledeep/aptos) listed a single train split with about 3,662 images, `image`, numeric `label_code` (0–4), and text `label`. Hosted images are 224×224, so detail lost during upstream resizing cannot be recovered. The authoring environment could access the web listing but could not resolve the download host and did not have `datasets` installed. **A Python `load_dataset()` download has not been verified here.** The notebook verifies a real download and image decode before training; errors are surfaced instead of silently switching data.

If it fails, obtain the original [Kaggle APTOS 2019 competition data](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data), following its account/access requirements and terms. Extract `train.csv` and `train_images/*.png` to one directory and set `Config(local_data="/content/aptos2019", ...)`. Only labeled training images are used; competition test labels are unavailable. The original-resolution source constitutes a different experiment from the resized HF copy. Keep separate output folders and report the source explicitly. This repository's MIT license does not grant dataset or pretrained-weight rights; review each source's terms.

## Splits and leakage

Run `notebooks/01_data_exploration.ipynb` for columns, image counts, class counts/bar chart, three original images per grade and an all-image dimensions scan. Its cells call `src/data.py`; no training logic is embedded. Replace the setup cell's GitHub URL first.

The exploration notebook and both training backbones share `MyDrive/RetinoCheckAI/splits/aptos_indices.json` (`Config.split_file`). First creation writes the stratified indices; every later run reads the existing file without overwriting it. The file includes the dataset fingerprint, ordered-label hash, seed and ratios. Changed data or settings, corrupt files, incomplete/overlapping indices and conflicting checkpoint splits raise errors. Keep a backup of this file; if it is lost, restore original indices from a checkpoint rather than treating a new split as the same experiment. Switching to the Kaggle source requires a separately named split file and a separate experiment. Integer rounding makes exact split counts approximate.

Fixed seed 42 creates stratified 70% train / 15% validation / 15% test image-level partitions (rounding affects exact sizes). Both backbones share these splits when source and seed agree. Only training images receive augmentation; class weights use training labels only. Checkpoints contain the dataset fingerprint and split indices; resume/evaluation refuse a changed fingerprint. Set `revision` to an immutable Hugging Face commit for final experiments. The local-source identity hashes the path/label manifest, not image bytes: retain immutable local files.

Patient identifiers are not available in this loader. Image-level splitting cannot establish patient independence or exclude duplicates. Audit duplicate/near-duplicate images and patient overlap before reporting strong generalization claims; if identifiers become available, replace splitting with stratified patient-group splitting. No external validation is provided.

## Checkpoints and recovery

The notebook mounts Drive and writes directly to `MyDrive/RetinoCheckAI/checkpoints/<model>/`. Each completed epoch writes `last.pt`, including model, best model, optimizer, LR scheduler, AMP scaler, epoch, config, split manifest and history. `best.pt` tracks the highest validation QWK; ties preserve the earlier model. Files use temporary-file replacement. Drive synchronization is not a transactional backup guarantee: confirm files appear in Drive. Disconnecting during an epoch loses that epoch's work, while the previous completed checkpoint should remain usable.

Rerun setup after a disconnect, rebuild the same configuration, and set `RESUME = True`. Epoch-boundary seeds recreate shuffling/augmentation with the same environment and worker count; GPU results can still vary across hardware/library versions. Increase `epochs` to extend a run. Other training settings must match. Checkpoints intentionally include best weights to preserve model selection when resuming to another directory. Resume rewrites `best.pt` from those weights; consult `history` for the best epoch. Load only project checkpoints you trust.

## Evaluation and explanation

`src.evaluate.evaluate` saves JSON metrics, indexed predictions and a confusion-matrix PNG. Main metric: quadratic weighted Cohen's kappa (QWK); also accuracy, macro F1 and per-class recall, always ordered 0–4. Confusion matrix rows are true grades and columns predicted grades. Undefined QWK is saved as JSON null and cannot win checkpoint selection. No performance results are claimed before training. Report counts, all metrics, seed, dataset revision, model, library versions and limitations; compare several seeds if resources allow.

Grad-CAM targets `layer4[-1]` for ResNet18 and `conv_head` for EfficientNet-B0. It highlights sensitivity for the predicted class. It neither establishes lesion localization nor explains causal reasoning. Softmax scores are uncalibrated and do not measure clinical certainty. The demo accepts arbitrary images and has no validated fundus or image-quality rejection.

## Preprocessing and optional Drive cache

Run `notebooks/02_preprocessing.ipynb` to display six training images before/after deterministic preprocessing. It writes `reports/preprocessing_examples.png` in the clone and a durable copy at `MyDrive/RetinoCheckAI/reports/preprocessing_examples.png`. The figure is generated from real data when the notebook runs; no APTOS example figure is prefilled in this repository.

The common stage converts to RGB, crops pixels outside a bounding box above intensity 10, pads to square, resizes to `config.image_size`, then applies optional Ben Graham-style enhancement: `clip(4 * (image - GaussianBlur(image)) + 128, 0, 255)`. Default Gaussian sigma is `image_size / 30`. Black background stays black. Pillow implements an approximate Gaussian blur; this does not claim pixel parity with an OpenCV implementation. Disable enhancement with `config.ben_graham=False` for a separately tracked baseline. Enhancement can suppress information or amplify artifacts; choose settings with validation data rather than assuming it improves QWK.

After this shared stage, training adds horizontal/vertical flips, rotations up to 15 degrees and light brightness/contrast jitter. Evaluation only resizes, converts to a tensor and applies ImageNet normalization. Training, evaluation, the demo and Grad-CAM use the same configured deterministic preprocessing. Older checkpoints lacking the enhancement setting are loaded without enhancement for inference; use matching settings when resuming experiments.

Set `config.cache_preprocessed=True` to lazily save/reuse deterministic RGB PNGs under `MyDrive/RetinoCheckAI/cache/preprocessed`. The notebook also has a `CACHE_TO_DRIVE` switch to warm the entire cache. Enable caching in the training notebook too if desired. Keys include original pixels/dimensions, settings, algorithm version and Pillow version, so settings changes do not reuse stale images. Augmentation and normalized tensors are never cached, and cached images are not enhanced twice. Source images still need decoding for content hashing, and Drive I/O can be slower than recomputing this lightweight stage.

## Demo

From the repository root: `python -m app.gradio_app --checkpoint /path/to/best.pt`.
In Colab use the final notebook cell. It explicitly opens a temporary public Gradio sharing link; run only when desired and use public or de-identified research images. Close the link/runtime after use. No patient data workflow is implemented.

## Files

- `src/config.py`, `data.py`, `preprocess.py`: settings, source loading, splits, transforms.
- `src/model.py`, `train.py`: timm backbones and resumable training.
- `src/evaluate.py`, `explain.py`: reports and Grad-CAM.
- `app/gradio_app.py`: research demo.
- `notebooks/01_colab.ipynb`: thin Colab entry point.
- `reports/`: reporting instructions, not fabricated metrics.
- `MODEL_CARD.md`: intended use and limitations.

Direct dependencies are pinned for Python 3.11–3.13. Setup installs the official PyTorch 2.6.0 / torchvision 0.21.0 CUDA 12.4 pair; this is not a full transitive lockfile. A complete installation on a live Colab T4 has not been verified here. Restart the session if older package versions were already imported, and save `pip freeze` with each run. Source validation in the authoring environment is documented in `reports/VALIDATION.md`.
