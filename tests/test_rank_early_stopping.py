import math
import unittest

from recommender.training.rank import EarlyStopping


class EarlyStoppingTest(unittest.TestCase):
    def test_stops_after_configured_number_of_non_improving_epochs(self):
        early_stopping = EarlyStopping(patience=2)

        self.assertEqual(early_stopping.step(0.60), (True, False))
        self.assertEqual(early_stopping.step(0.59), (False, False))
        self.assertEqual(early_stopping.step(0.60), (False, True))

    def test_improvement_resets_patience(self):
        early_stopping = EarlyStopping(patience=2, min_delta=0.01)

        self.assertEqual(early_stopping.step(0.60), (True, False))
        self.assertEqual(early_stopping.step(0.605), (False, False))
        self.assertEqual(early_stopping.step(0.62), (True, False))
        self.assertEqual(early_stopping.epochs_without_improvement, 0)

    def test_zero_patience_disables_stopping(self):
        early_stopping = EarlyStopping(patience=0)

        self.assertEqual(early_stopping.step(0.60), (True, False))
        self.assertEqual(early_stopping.step(math.nan), (False, False))
        self.assertEqual(early_stopping.step(0.50), (False, False))

    def test_rejects_negative_configuration(self):
        with self.assertRaises(ValueError):
            EarlyStopping(patience=-1)
        with self.assertRaises(ValueError):
            EarlyStopping(patience=1, min_delta=-0.1)


if __name__ == "__main__":
    unittest.main()
