"""Unit tests for Gradio app utilities and prediction logic."""
import unittest
import numpy as np

from app.gradio_app import (
    GRADE_SHORT_NAMES,
    FOLLOW_UP_RECOMMENDATIONS,
    continuous_score_to_probabilities,
    resolve_checkpoint_path,
)
from src.config import CLASS_NAMES


class TestGradioApp(unittest.TestCase):
    def test_class_names_and_grades_match(self):
        self.assertEqual(len(GRADE_SHORT_NAMES), 5)
        self.assertEqual(len(FOLLOW_UP_RECOMMENDATIONS), 5)
        self.assertEqual(len(CLASS_NAMES), 5)
        for i in range(5):
            self.assertIn(i, GRADE_SHORT_NAMES)
            self.assertIn(i, FOLLOW_UP_RECOMMENDATIONS)

    def test_continuous_score_to_probabilities_normalization(self):
        # Center at grade 2.0
        probs = continuous_score_to_probabilities(2.0, sigma=0.65)
        self.assertEqual(len(probs), 5)
        prob_values = list(probs.values())
        self.assertAlmostEqual(sum(prob_values), 1.0, places=5)
        # Class 2 should have maximum probability
        self.assertEqual(np.argmax(prob_values), 2)
        # Probabilities should be symmetric around 2.0
        self.assertAlmostEqual(prob_values[1], prob_values[3], places=4)
        self.assertAlmostEqual(prob_values[0], prob_values[4], places=4)

    def test_continuous_score_extreme_values(self):
        # Grade 0 floor
        probs_low = continuous_score_to_probabilities(-0.5, sigma=0.65)
        self.assertAlmostEqual(sum(probs_low.values()), 1.0, places=5)
        self.assertEqual(np.argmax(list(probs_low.values())), 0)

        # Grade 4 ceiling
        probs_high = continuous_score_to_probabilities(4.5, sigma=0.65)
        self.assertAlmostEqual(sum(probs_high.values()), 1.0, places=5)
        self.assertEqual(np.argmax(list(probs_high.values())), 4)

    def test_resolve_checkpoint_path_nonexistent(self):
        self.assertIsNone(resolve_checkpoint_path("non_existent_file.pt"))


if __name__ == "__main__":
    unittest.main()

