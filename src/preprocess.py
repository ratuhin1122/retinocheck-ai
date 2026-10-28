"""Deterministic preprocessing, optional PNG cache and train/eval transforms.

Raw -> crop/pad -> resize -> enhancement -> optional cache -> train-only
augmentation -> tensor/normalization. Evaluation never applies augmentation.
"""
from dataclasses import dataclass
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, __version__ as PILLOW_VERSION
from torchvision import transforms as T

PREPROCESS_VERSION = "crop-pad-ben-graham-v1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def crop_dark_border(image, threshold=10):
    """Crop the non-dark bounding box, then square-pad without stretching anatomy.

    Entirely dark images and one-pixel foregrounds are not cropped aggressively.
    This is preprocessing, not a validated image-quality test.
    """
    if not 0 <= threshold <= 255:
        raise ValueError("Border threshold must be between 0 and 255")
    image = image.convert("RGB")
    mask = np.asarray(image).max(axis=2) > threshold
    ys, xs = np.where(mask)
    if len(xs) and xs.max() > xs.min() and ys.max() > ys.min():
        image = image.crop((int(xs.min()), int(ys.min()),
                            int(xs.max()) + 1, int(ys.max()) + 1))
    side = max(image.size)
    square = Image.new("RGB", (side, side))
    square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return square


def ben_graham_enhance(image, sigma, gain=4.0, mean=128.0):
    """Compute clip(gain * (image - GaussianBlur(image)) + mean, 0, 255).

    The conventional fixed gray mean is 128; no dataset statistics are fitted.
    Pillow approximates a Gaussian blur with this sigma. Float32 arithmetic
    prevents unsigned-integer subtraction from wrapping around.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be positive and finite")
    if not np.isfinite(gain) or gain <= 0 or not np.isfinite(mean) or not 0 <= mean <= 255:
        raise ValueError("Gain must be positive and finite; mean must be in [0, 255]")
    image = image.convert("RGB")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    enhanced = gain * (np.asarray(image, dtype=np.float32) -
                       np.asarray(blurred, dtype=np.float32)) + mean
    return Image.fromarray(np.clip(enhanced, 0, 255).astype(np.uint8))


def preprocess_image(image, size=224, enhance=True, sigma=None, gain=4.0,
                     mean=128.0, border_threshold=10):
    """Return deterministic RGB pixels; do not augment or normalize here."""
    if not isinstance(size, int) or size < 1:
        raise ValueError("Image size must be a positive integer")
    resized = crop_dark_border(image, border_threshold).resize(
        (size, size), resample=Image.Resampling.BILINEAR)
    if not enhance:
        return resized
    enhanced = ben_graham_enhance(resized, size / 30 if sigma is None else sigma, gain, mean)
    # Keep dark background/padding black instead of turning it into a gray frame.
    pixels = np.array(enhanced)
    pixels[np.asarray(resized).max(axis=2) <= border_threshold] = 0
    return Image.fromarray(pixels)


def preprocessing_options(config):
    """Settings shared by training, evaluation, inference and cache keys."""
    return {"size": config.image_size, "enhance": config.ben_graham,
            "sigma": config.enhancement_sigma, "gain": config.enhancement_gain,
            "mean": config.enhancement_mean, "border_threshold": config.border_threshold}


@dataclass
class CachedPreprocessor:
    """Content-addressed, lossless PNG cache of deterministic pixels only.

    Input pixels/dimensions, settings, algorithm version and Pillow version
    determine each key. Augmentations and normalized tensors are never cached.
    """
    cache_dir: str
    options: dict

    def path_for(self, image):
        image = image.convert("RGB")
        metadata = {"algorithm": PREPROCESS_VERSION, "pillow": PILLOW_VERSION,
                    "options": self.options, "input_size": image.size}
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
        digest.update(image.tobytes())
        key = digest.hexdigest()
        return Path(self.cache_dir) / key[:2] / f"{key}.png"

    def __call__(self, image):
        path = self.path_for(image)
        if path.exists():
            with Image.open(path) as cached:
                cached.load()
                if cached.mode != "RGB" or cached.size != (self.options["size"],) * 2:
                    raise ValueError(f"Invalid cached image: {path}")
                return cached.copy()
        result = preprocess_image(image, **self.options)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unique temp files allow DataLoader workers to share the cache safely.
        descriptor, temporary = tempfile.mkstemp(prefix=".preprocess-", suffix=".tmp", dir=path.parent)
        os.close(descriptor)
        try:
            result.save(temporary, format="PNG")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return result


def build_preprocessor(config):
    options = preprocessing_options(config)
    if config.cache_preprocessed:
        return CachedPreprocessor(config.preprocessed_cache_dir, options)
    return partial(preprocess_image, **options)


def make_transform(size=224, train=False, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                   preprocessed=False, enhancement_mean=128.0, **preprocess_options):
    """Include shared preprocessing for raw inputs; skip it for cached pixels.

    The eval stage only resizes, converts to a tensor and normalizes. Training
    also applies flips, +/-15-degree rotations and light brightness/contrast.
    """
    steps = []
    if not preprocessed:
        steps.append(T.Lambda(partial(preprocess_image, size=size, mean=enhancement_mean,
                                      **preprocess_options)))
    steps.append(T.Resize((size, size), antialias=True))
    if train:
        steps += [T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5),
                  T.RandomRotation(15, interpolation=T.InterpolationMode.BILINEAR, fill=0),
                  T.ColorJitter(brightness=0.1, contrast=0.1)]
    return T.Compose(steps + [T.ToTensor(), T.Normalize(mean, std)])


def build_transform(config, train=False, preprocessed=False):
    options = preprocessing_options(config)
    size = options.pop("size")
    # Enhancement's pixel mean and ImageNet's channel means are different units.
    enhancement_mean = options.pop("mean")
    return make_transform(size, train=train, preprocessed=preprocessed,
                          enhancement_mean=enhancement_mean, **options)


def cache_preprocessed_images(source, config, indices=None):
    """Warm the Drive cache one image at a time; repeated calls reuse files.

    Split membership never changes. Source pixels must still be read for content
    hashing on cache hits. Drive I/O can be slower than local preprocessing.
    """
    from tqdm.auto import tqdm
    from .data import read_image
    processor = CachedPreprocessor(config.preprocessed_cache_dir, preprocessing_options(config))
    indices = range(len(source)) if indices is None else indices
    for index in tqdm(indices, desc="Caching deterministic preprocessing"):
        processor(read_image(source, index))
    print(f"Cache ready: {config.preprocessed_cache_dir}")
    return Path(config.preprocessed_cache_dir)


def save_preprocessing_comparison(source, config, indices, output_path,
                                  count=6, also_save_to=None):
    """Save a labeled before/after PNG, sampling only the supplied indices.

    Originals are scaled only for display. Processed pixels are shown before
    augmentation/normalization. Pillow assembles the comparison without fitting
    any data statistics or requiring an interactive plotting backend.
    """
    from .data import read_image
    if len(indices) < count:
        raise ValueError(f"Need at least {count} candidate images")
    selected = np.random.default_rng(config.seed).choice(indices, count, replace=False)
    width, height = 340, 300
    canvas = Image.new("RGB", (2 * width, count * height + 40), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 12), "Before: original (display scaled)", fill="black")
    draw.text((width + 15, 12), "After: crop / resize / enhancement", fill="black")
    processor = build_preprocessor(config)
    for row, index in enumerate(selected):
        original = read_image(source, index)
        processed = processor(original)
        label = int(source[int(index)]["label_code"])
        for column, image in enumerate((original, processed)):
            thumbnail = ImageOps.contain(image, (width - 20, height - 35))
            x = column * width + (width - thumbnail.width) // 2
            y = 40 + row * height
            canvas.paste(thumbnail, (x, y))
            draw.text((column * width + 10, y + height - 27),
                      f"Index {index} | grade {label} | {image.width}x{image.height}", fill="black")
    for destination in (output_path, also_save_to):
        if destination is not None:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(path, format="PNG")
            print(f"Saved comparison: {path}")
    return canvas
