"""Identical deterministic preprocessing for validation, test and the demo."""
import numpy as np
from PIL import Image
from torchvision import transforms as T


def crop_dark_border(image):
    """Remove dark margins, preserving all content when an image is nearly black."""
    image = image.convert("RGB")
    pixels = np.asarray(image)
    mask = pixels.max(axis=2) > 10
    ys, xs = np.where(mask)
    if len(xs) and xs.max() > xs.min() and ys.max() > ys.min():
        image = image.crop((int(xs.min()), int(ys.min()),
                            int(xs.max()) + 1, int(ys.max()) + 1))
    # Pad to square rather than stretch the retina.
    side = max(image.size)
    square = Image.new("RGB", (side, side))
    square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return square


def make_transform(size=224, train=False, mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)):
    steps = [T.Lambda(crop_dark_border), T.Resize((size, size), antialias=True)]
    if train:
        steps += [T.RandomHorizontalFlip(), T.RandomRotation(15),
                  T.ColorJitter(brightness=0.1, contrast=0.1)]
    return T.Compose(steps + [T.ToTensor(), T.Normalize(mean, std)])
