"""Small, serializable configuration shared by training and inference."""
from dataclasses import dataclass

CLASS_NAMES = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]


@dataclass
class Config:
    model_name: str = "resnet18"
    dataset_name: str = "bumbledeep/aptos"
    revision: str = "main"  # Use an immutable Hub commit for a final experiment.
    local_data: str = ""  # Kaggle directory with train.csv and train_images/.
    output_dir: str = "/content/drive/MyDrive/RetinoCheckAI/resnet18"
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 2
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    class_weights: bool = True

    def __post_init__(self):
        if self.model_name not in ("resnet18", "efficientnet_b0"):
            raise ValueError("Choose resnet18 or efficientnet_b0")
        if not (0 < self.val_fraction < 1 and 0 < self.test_fraction < 1
                and self.val_fraction + self.test_fraction < 1):
            raise ValueError("Validation and test fractions must leave training data")
        if min(self.batch_size, self.epochs, self.image_size) < 1:
            raise ValueError("Batch size, epochs and image size must be positive")
