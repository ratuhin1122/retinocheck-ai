"""Load APTOS, verify labels, and make fixed stratified image-level splits."""
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .preprocess import make_transform
from .config import CLASS_NAMES


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


def read_image(source, index):
    """Read an original RGB image, before cropping/resizing/augmentation."""
    image = source[int(index)]["image"]
    if isinstance(image, str):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    return image.convert("RGB")


def describe_dataset(source, labels):
    """Print the schema, class counts and size statistics for ALL images.

    Images are decoded one at a time to keep Colab memory use low. Any broken
    image raises an error rather than silently changing the indexed dataset.
    """
    from tqdm.auto import tqdm
    columns = source.column_names if hasattr(source, "column_names") else list(source[0])
    print("Columns:", columns)
    print("Number of images:", len(labels))
    for grade, count in enumerate(np.bincount(labels, minlength=5)):
        print(f"{CLASS_NAMES[grade]}: {count:,} ({count / len(labels):.2%})")
    sizes = []
    for index in tqdm(range(len(source)), desc="Reading original image dimensions"):
        try:
            sizes.append(read_image(source, index).size)
        except Exception as exc:
            raise RuntimeError(f"Cannot decode image at source index {index}") from exc
    sizes = np.asarray(sizes, dtype=np.int64)
    print("\nOriginal image dimensions (before project preprocessing):")
    for axis, name in enumerate(("Width", "Height")):
        values = sizes[:, axis]
        print(f"{name}: min={values.min()}, median={np.median(values):.1f}, "
              f"mean={values.mean():.1f}, max={values.max()}, std={values.std():.1f} px")
    print("Distinct (width, height) pairs:", len(np.unique(sizes, axis=0)))
    return sizes


def plot_grade_counts(labels):
    """Show the five-class distribution and return the figure."""
    import matplotlib.pyplot as plt
    counts = np.bincount(labels, minlength=5)
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(range(5), counts, color="steelblue")
    ax.bar_label(bars, padding=3)
    ax.set(xticks=range(5), xticklabels=CLASS_NAMES, ylabel="Number of images",
           xlabel="Dataset annotation grade", title="APTOS class distribution")
    ax.margins(y=0.15)
    fig.tight_layout()
    plt.show()
    return fig


def show_grade_samples(source, labels, seed=42, samples_per_grade=3):
    """Show original images, using a local RNG so training seeds are unaffected."""
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(5, samples_per_grade, figsize=(3 * samples_per_grade, 14),
                             squeeze=False)
    for grade in range(5):
        candidates = np.flatnonzero(labels == grade)
        if len(candidates) < samples_per_grade:
            raise ValueError(f"Grade {grade} has fewer than {samples_per_grade} images")
        for column, index in enumerate(rng.choice(candidates, samples_per_grade, replace=False)):
            axes[grade, column].imshow(read_image(source, index))
            axes[grade, column].set_title(f"{CLASS_NAMES[grade]} | index {index}")
            axes[grade, column].axis("off")
    fig.tight_layout()
    plt.show()
    return fig


def validate_manifest(manifest, labels, identity):
    """Reject a changed source, overlap, missing rows or corrupt indices."""
    if manifest.get("identity") != identity:
        raise ValueError("Dataset fingerprint changed. Refusing to change the saved test set.")
    splits = manifest.get("indices", {})
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Split file must contain train, val and test indices")
    combined = []
    for name in ("train", "val", "test"):
        indices = splits[name]
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"Empty or invalid {name} indices")
        if any(type(index) is not int or index < 0 or index >= len(labels) for index in indices):
            raise ValueError(f"Invalid index in {name}")
        if set(labels[indices].tolist()) != set(range(5)):
            raise ValueError(f"{name} does not contain all five grades")
        combined.extend(indices)
    if len(combined) != len(labels) or len(set(combined)) != len(labels):
        raise ValueError("Splits overlap, contain duplicate indices, or omit dataset rows")
    if manifest.get("num_images", len(labels)) != len(labels):
        raise ValueError("Saved image count differs from the current dataset")
    if "fractions" in manifest:
        fractions = manifest["fractions"]
        if (set(fractions) != set(splits) or
                not all(0 < value < 1 for value in fractions.values()) or
                not np.isclose(sum(fractions.values()), 1.0)):
            raise ValueError("Invalid saved split ratios")
        for name in splits:
            if abs(len(splits[name]) - len(labels) * fractions[name]) > 2:
                raise ValueError(f"{name} size differs from the saved split ratio")
    expected_hash = hashlib.sha256(np.asarray(labels, dtype="<i8").tobytes()).hexdigest()
    if "labels_sha256" in manifest and manifest["labels_sha256"] != expected_hash:
        raise ValueError("Label order changed: refusing to reuse indices")


def get_or_create_splits(config, labels, identity, saved_manifest=None):
    """Use one persistent Drive manifest across notebooks and both backbones.

    An existing file is NEVER regenerated or overwritten. Changed data, seed,
    ratios or conflicting checkpoint indices cause an error instead. Back up
    this file with the research artifacts; a lost file cannot guarantee the
    same test set when upstream data changes. Run creation in one notebook.
    """
    path = Path(config.split_file)
    fractions = {"train": config.train_fraction, "val": config.val_fraction,
                 "test": config.test_fraction}
    if saved_manifest is not None:
        validate_manifest(saved_manifest, labels, identity)
        if (("seed" in saved_manifest and saved_manifest["seed"] != config.seed) or
                ("fractions" in saved_manifest and saved_manifest["fractions"] != fractions)):
            raise ValueError("Checkpoint seed or ratios differ from the requested configuration")
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_manifest(manifest, labels, identity)
        if manifest.get("seed") != config.seed or manifest.get("fractions") != fractions:
            raise ValueError("Seed or ratios differ from the frozen Drive split file")
        if saved_manifest is not None and manifest["indices"] != saved_manifest["indices"]:
            raise ValueError("Checkpoint splits conflict with frozen Drive splits")
        print(f"Reusing frozen splits: {path}")
        return manifest
    if saved_manifest is None:
        from sklearn.model_selection import train_test_split
        train, rest = train_test_split(np.arange(len(labels)),
            test_size=config.val_fraction + config.test_fraction,
            stratify=labels, random_state=config.seed)
        val, test = train_test_split(rest,
            test_size=config.test_fraction / (config.val_fraction + config.test_fraction),
            stratify=labels[rest], random_state=config.seed)
        indices = {"train": train.tolist(), "val": val.tolist(), "test": test.tolist()}
    else:
        # Recover an earlier run's exact test membership when Drive has no file.
        indices = saved_manifest["indices"]
    manifest = {"schema_version": 1, "identity": identity, "dataset": config.dataset_name,
                "revision": config.revision, "local_data": config.local_data,
                "num_images": len(labels), "seed": config.seed, "fractions": fractions,
                "labels_sha256": hashlib.sha256(np.asarray(labels, dtype="<i8").tobytes()).hexdigest(),
                "indices": indices}
    validate_manifest(manifest, labels, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of an existing file.
    # If interrupted while writing, an invalid file is rejected on the next run;
    # restore it from a backup/checkpoint rather than silently creating new splits.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    print(f"Saved frozen splits: {path}")
    return manifest


def verify_split_proportions(labels, manifest):
    """Print counts/proportions and check stratification with rounding tolerance."""
    validate_manifest(manifest, labels, manifest["identity"])
    overall = np.bincount(labels, minlength=5) / len(labels)
    print("Overall grade proportions:", np.round(overall, 4).tolist())
    report = {}
    for name in ("train", "val", "test"):
        indices = manifest["indices"][name]
        counts = np.bincount(labels[indices], minlength=5)
        proportions = counts / len(indices)
        # Two stratified splits introduce small integer-rounding differences.
        tolerance = 2 / len(indices) + 2 / len(labels)
        if np.max(np.abs(proportions - overall)) > tolerance:
            raise ValueError(f"Unexpected class distribution in {name}")
        print(f"\n{name}: {len(indices):,} images ({len(indices) / len(labels):.2%} of data)")
        print("grade | count | split proportion | overall proportion")
        for grade in range(5):
            print(f"{grade:5} | {counts[grade]:5} | {proportions[grade]:16.2%} | {overall[grade]:18.2%}")
        report[name] = {"count": len(indices), "class_counts": counts.tolist(),
                        "class_proportions": proportions.tolist()}
    print("\nPASS: disjoint, complete splits; all grades present; class proportions preserved.")
    return report


def make_loaders(config, saved_manifest=None):
    source, labels, identity = load_source(config)
    manifest = get_or_create_splits(config, labels, identity, saved_manifest)
    loaders = {}
    for split, indices in manifest["indices"].items():
        dataset = FundusDataset(source, indices,
                                make_transform(config.image_size, train=split == "train"))
        loaders[split] = DataLoader(dataset, batch_size=config.batch_size,
            shuffle=split == "train", num_workers=config.num_workers,
            pin_memory=True)
    return loaders, labels[manifest["indices"]["train"]], manifest
