"""Small, serializable configuration shared by training and inference."""
from dataclasses import dataclass
from pathlib import Path

# Drive holds durable outputs; /content holds disposable downloads and code.
DRIVE_ROOT = Path("/content/drive/MyDrive/RetinoCheckAI")
CHECKPOINT_ROOT = DRIVE_ROOT / "checkpoints"
REPORT_ROOT = DRIVE_ROOT / "reports"
REPO_DIR = Path("/content/retinocheck-ai")
DATA_DIR = Path("/content/data/aptos")
HF_CACHE_DIR = Path("/content/hf_cache")

# Annotation category names, not a diagnosis for an uploaded image.
CLASS_NAMES = ["0 - No DR", "1 - Mild", "2 - Moderate", "3 - Severe", "4 - Proliferative DR"]
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15


@dataclass
class Config:
    model_name: str = "resnet18"
    dataset_name: str = "bumbledeep/aptos"
    revision: str = "main"  # Use an immutable Hub commit for a final experiment.
    local_data: str = ""  # Kaggle directory with train.csv and train_images/.
    split_file: str = str(DRIVE_ROOT / "splits" / "aptos_indices.json")
    output_dir: str = str(CHECKPOINT_ROOT / "resnet18")
    image_size: int = 224
    ben_graham: bool = True
    enhancement_sigma: float | None = None  # None means image_size / 30.
    enhancement_gain: float = 4.0
    enhancement_mean: float = 128.0
    border_threshold: int = 10
    cache_preprocessed: bool = False
    preprocessed_cache_dir: str = str(DRIVE_ROOT / "cache" / "preprocessed")
    batch_size: int = 32
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-2
    seed: int = 42
    num_workers: int = 2
    val_fraction: float = VAL_RATIO
    test_fraction: float = TEST_RATIO
    class_weights: bool = True
    # --- new fields (train.py v2) -------------------------------------------
    # "classification" uses CrossEntropyLoss; "regression" uses SmoothL1Loss or MSELoss.
    mode: str = "classification"
    # Loss function for regression mode: "smooth_l1" or "mse".
    regression_loss: str = "smooth_l1"
    # Number of DR severity grades (classification head width).
    num_classes: int = 5
    # Early-stopping patience in epochs (0 = disabled).
    patience: int = 5
    # Use a WeightedRandomSampler instead of class-weighted loss when True.
    use_weighted_sampler: bool = False
    # Run threshold search on the validation set after every epoch (regression).
    threshold_search: bool = True

    @property
    def train_fraction(self):
        """Derived so the three fractions always sum to one."""
        return 1.0 - self.val_fraction - self.test_fraction

    def __post_init__(self):
        if self.model_name not in ("resnet18", "efficientnet_b0"):
            raise ValueError("Choose resnet18 or efficientnet_b0")
        if self.mode not in ("classification", "regression"):
            raise ValueError("mode must be 'classification' or 'regression'")
        if self.regression_loss not in ("smooth_l1", "mse"):
            raise ValueError("regression_loss must be 'smooth_l1' or 'mse'")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if self.patience < 0:
            raise ValueError("patience must be >= 0")
        if not (0 < self.val_fraction < 1 and 0 < self.test_fraction < 1
                and self.val_fraction + self.test_fraction < 1):
            raise ValueError("Validation and test fractions must leave training data")
        if min(self.batch_size, self.epochs, self.image_size) < 1:
            raise ValueError("Batch size, epochs and image size must be positive")
        if self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("Learning rate must be positive; weight decay cannot be negative")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.enhancement_sigma is not None and self.enhancement_sigma <= 0:
            raise ValueError("Enhancement sigma must be positive")
        if self.enhancement_gain <= 0 or not 0 <= self.enhancement_mean <= 255:
            raise ValueError("Invalid enhancement gain or mean")
        if not 0 <= self.border_threshold <= 255:
            raise ValueError("Border threshold must be in [0, 255]")
