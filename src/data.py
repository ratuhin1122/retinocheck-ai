"""Load APTOS, verify labels, and make fixed stratified image-level splits."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .preprocess import make_transform


class FundusDataset(Dataset):
    def __init__(self, source, indices, transform):
        self.source, self.indices, self.transform = source, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.source[int(self.indices[index])]
        if isinstance(row["image"], str):
            with Image.open(row["image"]) as image:
                tensor = self.transform(image.convert("RGB"))
        else:
            tensor = self.transform(row["image"].convert("RGB"))
        return tensor, int(row["label_code"])


def load_source(config):
    if config.local_data:
        root = Path(config.local_data)
        with (root / "train.csv").open(newline="") as handle:
            source = [{"image": str(root / "train_images" / (r["id_code"] + ".png")),
                       "label_code": int(r["diagnosis"])} for r in csv.DictReader(handle)]
        labels = np.array([row["label_code"] for row in source])
        identity = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    else:
        from datasets import load_dataset
        try:
            source = load_dataset(config.dataset_name, revision=config.revision, split="train")
        except Exception as exc:
            raise RuntimeError("Hugging Face APTOS loading failed. Check network/access; "
                               "alternative: download Kaggle APTOS 2019 train.csv and "
                               "train_images/, then set Config(local_data=...).") from exc
        if not {"image", "label_code"}.issubset(source.column_names):
            raise ValueError(f"Unexpected dataset schema: {source.column_names}")
        labels = np.array(source["label_code"])
        identity = source._fingerprint
    if set(labels.tolist()) != set(range(5)):
        raise ValueError("Expected all five numeric grades, 0 through 4")
    # Decode one image now so broken image dependencies fail before training.
    FundusDataset(source, [0], make_transform(config.image_size))[0]
    return source, labels.astype(np.int64), identity


def make_loaders(config, saved_manifest=None):
    source, labels, identity = load_source(config)
    if saved_manifest is None:
        train, rest = train_test_split(np.arange(len(labels)),
            test_size=config.val_fraction + config.test_fraction,
            stratify=labels, random_state=config.seed)
        val, test = train_test_split(rest,
            test_size=config.test_fraction / (config.val_fraction + config.test_fraction),
            stratify=labels[rest], random_state=config.seed)
        manifest = {"identity": identity, "indices": {
            "train": train.tolist(), "val": val.tolist(), "test": test.tolist()}}
    else:
        manifest = saved_manifest
        if manifest["identity"] != identity:
            raise ValueError("Dataset changed: refusing to reuse checkpoint splits")
    loaders = {}
    for split, indices in manifest["indices"].items():
        dataset = FundusDataset(source, indices,
                                make_transform(config.image_size, train=split == "train"))
        loaders[split] = DataLoader(dataset, batch_size=config.batch_size,
            shuffle=split == "train", num_workers=config.num_workers,
            pin_memory=True)
    return loaders, labels[manifest["indices"]["train"]], manifest
