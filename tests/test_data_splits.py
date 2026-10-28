"""Run: python -m unittest discover -s tests -v

Persistence checks use synthetic labels and no image or network downloads.
"""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.config import Config
from src.data import get_or_create_splits, validate_manifest, verify_split_proportions


class FrozenSplitsTests(unittest.TestCase):
    def setUp(self):
        self.test_root = Path(__file__).resolve().parent
        self.directory = tempfile.TemporaryDirectory(prefix=".split-test-", dir=self.test_root)
        self.addCleanup(self.cleanup_directory)
        self.config = Config(split_file=str(Path(self.directory.name) / "splits.json"))
        self.labels = np.repeat(np.arange(5), 200)
        self.original = {"identity": "synthetic-v1", "indices": {"train": [], "val": [], "test": []}}
        for grade in range(5):
            rows = list(range(grade * 200, (grade + 1) * 200))
            self.original["indices"]["train"].extend(rows[:140])
            self.original["indices"]["val"].extend(rows[140:170])
            self.original["indices"]["test"].extend(rows[170:])

    def cleanup_directory(self):
        # Verify the exact test workspace before recursively cleaning temp files.
        target = Path(self.directory.name).resolve()
        if target.parent != self.test_root or not target.name.startswith(".split-test-"):
            raise RuntimeError("Refusing cleanup outside the temporary test directory")
        self.directory.cleanup()

    def freeze(self):
        return get_or_create_splits(self.config, self.labels, "synthetic-v1", self.original)

    def test_recover_checkpoint_then_reload_without_rewriting(self):
        frozen = self.freeze()
        path = Path(self.config.split_file)
        before = path.read_bytes()
        reloaded = get_or_create_splits(self.config, self.labels, "synthetic-v1")
        self.assertEqual(frozen["indices"], self.original["indices"])
        self.assertEqual(frozen, reloaded)
        self.assertEqual(before, path.read_bytes())
        report = verify_split_proportions(self.labels, reloaded)
        self.assertEqual([report[name]["count"] for name in ("train", "val", "test")], [700, 150, 150])

    def test_changed_source_seed_and_labels_are_rejected(self):
        self.freeze()
        with self.assertRaises(ValueError):
            get_or_create_splits(self.config, self.labels, "different-source")
        self.config.seed += 1
        with self.assertRaises(ValueError):
            get_or_create_splits(self.config, self.labels, "synthetic-v1")
        self.config.seed -= 1
        with self.assertRaises(ValueError):
            get_or_create_splits(self.config, self.labels[::-1], "synthetic-v1")

    def test_overlap_and_missing_indices_are_rejected(self):
        broken = copy.deepcopy(self.original)
        broken["indices"]["test"][0] = broken["indices"]["train"][0]
        with self.assertRaises(ValueError):
            validate_manifest(broken, self.labels, "synthetic-v1")

    def test_conflicting_checkpoint_is_rejected(self):
        self.freeze()
        conflicting = copy.deepcopy(self.original)
        conflicting["indices"]["val"][0], conflicting["indices"]["test"][0] = (
            conflicting["indices"]["test"][0], conflicting["indices"]["val"][0])
        with self.assertRaises(ValueError):
            get_or_create_splits(self.config, self.labels, "synthetic-v1", conflicting)

    def test_corrupt_file_is_not_replaced(self):
        path = Path(self.config.split_file)
        path.write_text("{interrupted", encoding="utf-8")
        with self.assertRaises(ValueError):
            get_or_create_splits(self.config, self.labels, "synthetic-v1")
        self.assertEqual(path.read_text(encoding="utf-8"), "{interrupted")

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn unavailable locally")
    def test_new_stratified_split(self):
        labels = np.repeat(np.arange(5), [1805, 370, 999, 193, 295])
        manifest = get_or_create_splits(self.config, labels, "imbalanced-fixture")
        verify_split_proportions(labels, manifest)
        self.assertEqual(sum(map(len, manifest["indices"].values())), len(labels))


if __name__ == "__main__":
    unittest.main()
