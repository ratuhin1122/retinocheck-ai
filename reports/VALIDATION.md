# Authoring validation — 2026-10-28

Passed locally on CPU (Python 3.13, torch 2.14.0+cpu, torchvision 0.29.0+cpu):

- Compilation and AST parsing of all ten Python source files.
- Syntax parsing of notebook Python cells after removing Colab magic lines; valid notebook JSON.
- Configuration defaults and rejection of unsupported backbones, invalid split fractions and zero batch size.
- Synthetic preprocessing checks: dark-border crop, black-image handling, RGB/grayscale/RGBA inputs, finite normalized tensors, deterministic evaluation output and training augmentation output shape.

Not verified in this environment: Hugging Face Python download, pretrained-weight download, complete model training, numerical metrics with scikit-learn, Drive recovery, Grad-CAM and Gradio launch. The local environment lacks datasets/scikit-learn and the shell could not resolve the dataset server. The HF web viewer was reachable and confirmed the published image/label schema. The Colab notebook includes a real load and image-decode check before training.

Starter-file update: direct dependencies are now pinned, defaults are 224 px / batch 32 / seed 42 / learning rate 1e-4 / weight decay 1e-2 / 15 epochs, and the shared helper is in `src/utils.py`. Updated source and setup-cell syntax, configuration ratios and Python/NumPy/torch CPU seed repeatability passed locally. The pinned environment itself has not been installed or tested here; the existing local torch version differs from the pins. The setup checks imports and a real CUDA tensor operation in Colab.

No GPU training or model performance is claimed. Complete a one-epoch Colab run, confirm Drive files, resume it for another epoch, evaluate, and try the optional demo before starting longer experiments. Use a separate smoke-test output folder to keep these runs separate from reported experiments.

## Data exploration update

`notebooks/01_data_exploration.ipynb` has eight syntactically valid Python code cells and no prefilled outputs. Source compilation passed. Five synthetic split-persistence tests passed: checkpoint-index recovery and unchanged reload, rejection of source/seed/label changes, overlap detection, checkpoint conflict detection, and preservation of a corrupt file without overwriting it. One new-split stratification test was skipped because scikit-learn is unavailable locally. An attempt to fetch test dependencies found no accessible distribution through this environment's pip connection. Full HF loading, chart rendering and new stratified-split creation remain checks to run in Colab; no real-data statistics are fabricated here.
