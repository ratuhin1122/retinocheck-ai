"""Offline preprocessing checks using synthetic pixels, not APTOS images."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image, ImageFilter
import torch
from src.config import Config
from src.data import FundusDataset
from src.preprocess import (CachedPreprocessor, ben_graham_enhance, build_preprocessor,
                            build_transform, crop_dark_border, preprocessing_options,
                            preprocess_image, save_preprocessing_comparison)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.test_root = Path(__file__).resolve().parent
        self.directory = tempfile.TemporaryDirectory(prefix=".preprocess-test-", dir=self.test_root)
        self.addCleanup(self.cleanup_directory)
        self.config = Config(preprocessed_cache_dir=self.directory.name)
        pixels = np.zeros((100, 140, 3), dtype=np.uint8)
        pixels[20:80, 30:110] = (170, 90, 60)
        pixels[35:50, 50:70] = (90, 40, 30)
        self.image = Image.fromarray(pixels)

    def cleanup_directory(self):
        target = Path(self.directory.name).resolve()
        if target.parent != self.test_root or not target.name.startswith(".preprocess-test-"):
            raise RuntimeError("Refusing cleanup outside temporary test directory")
        self.directory.cleanup()

    def test_crop_padding_and_dark_image(self):
        cropped = crop_dark_border(self.image)
        self.assertEqual(cropped.size, (80, 80))
        self.assertEqual(np.asarray(cropped)[0].max(), 0)
        black = preprocess_image(Image.new("RGB", (50, 30)))
        self.assertEqual(black.size, (224, 224))
        self.assertEqual(np.asarray(black).max(), 0)

    def test_enhancement_formula_float_arithmetic_and_clipping(self):
        image = self.image.resize((40, 40))
        blur = image.filter(ImageFilter.GaussianBlur(2))
        expected = np.clip(4 * (np.asarray(image, dtype=np.float32) -
                               np.asarray(blur, dtype=np.float32)) + 128, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(np.asarray(ben_graham_enhance(image, 2)), expected)
        constant = ben_graham_enhance(Image.new("RGB", (20, 20), (90, 60, 40)), 2)
        self.assertTrue(np.all(np.asarray(constant) == 128))

    def test_input_modes_and_invalid_sigma(self):
        for mode in ("L", "RGBA", "RGB"):
            result = preprocess_image(self.image.convert(mode), size=64)
            self.assertEqual((result.mode, result.size), ("RGB", (64, 64)))
        with self.assertRaises(ValueError):
            ben_graham_enhance(self.image, 0)

    def test_cache_matches_uncached_and_keys_track_inputs_and_settings(self):
        processor = CachedPreprocessor(self.directory.name, preprocessing_options(self.config))
        expected = preprocess_image(self.image, **preprocessing_options(self.config))
        actual = processor(self.image)
        path = processor.path_for(self.image)
        before = path.read_bytes()
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
        np.testing.assert_array_equal(np.asarray(processor(self.image)), np.asarray(expected))
        self.assertEqual(path.read_bytes(), before)
        changed = self.image.copy()
        changed.putpixel((0, 0), (1, 2, 3))
        self.assertNotEqual(path, processor.path_for(changed))
        settings = preprocessing_options(self.config)
        settings["gain"] = 2.0
        self.assertNotEqual(path, CachedPreprocessor(self.directory.name, settings).path_for(self.image))

    def test_eval_is_deterministic_and_cache_is_not_enhanced_twice(self):
        raw_transform = build_transform(self.config)
        expected = raw_transform(self.image)
        self.assertTrue(torch.equal(expected, raw_transform(self.image)))
        transform = build_transform(self.config, preprocessed=True)
        self.assertEqual([type(step).__name__ for step in transform.transforms],
                         ["Resize", "ToTensor", "Normalize"])
        self.config.cache_preprocessed = True
        dataset = FundusDataset([{"image": self.image, "label_code": 2}], [0], transform,
                                preprocessor=build_preprocessor(self.config))
        tensor, grade = dataset[0]
        self.assertEqual(grade, 2)
        self.assertTrue(torch.equal(expected, tensor))
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(tensor).all())
        pixels = np.asarray(preprocess_image(self.image), dtype=np.float32) / 255.0
        reference = (torch.from_numpy(pixels.copy()).permute(2, 0, 1) -
                     torch.tensor([0.485, 0.456, 0.406])[:, None, None])
        reference /= torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        torch.testing.assert_close(tensor, reference)

    def test_training_augments_without_changing_shape(self):
        torch.manual_seed(7)
        transform = build_transform(self.config, train=True)
        tensors = [transform(self.image) for _ in range(3)]
        self.assertTrue(all(tuple(tensor.shape) == (3, 224, 224) for tensor in tensors))
        self.assertTrue(any(not torch.equal(tensors[0], tensor) for tensor in tensors[1:]))

    def test_comparison_saves_six_pairs(self):
        source = [{"image": self.image, "label_code": i % 5} for i in range(6)]
        destination = Path(self.directory.name) / "preprocessing_examples.png"
        comparison = save_preprocessing_comparison(source, self.config, list(range(6)), destination)
        self.assertEqual(comparison.size, (680, 1840))
        with Image.open(destination) as saved:
            self.assertEqual(saved.size, comparison.size)
            self.assertEqual(saved.format, "PNG")


if __name__ == "__main__":
    unittest.main()
