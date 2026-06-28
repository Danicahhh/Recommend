import math
import unittest

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from recommender.evaluation import compute_multitask_metrics, gauc_score


def logit(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return np.log(probabilities / (1.0 - probabilities))


class RankEvaluationTest(unittest.TestCase):
    def test_auc_and_logloss_match_sklearn(self):
        targets = {"click": [0, 1, 0, 1]}
        probabilities = np.array([0.1, 0.8, 0.3, 0.9])
        metrics = compute_multitask_metrics(
            targets=targets,
            logits={"click": logit(probabilities)},
            user_ids=[1, 1, 2, 2],
            task_names=("click",),
        )

        self.assertAlmostEqual(
            metrics["click_auc"], roc_auc_score(targets["click"], probabilities)
        )
        self.assertAlmostEqual(
            metrics["click_logloss"],
            log_loss(targets["click"], probabilities, labels=[0, 1]),
        )

    def test_gauc_is_sample_weighted_and_skips_single_label_users(self):
        targets = [0, 1, 0, 1, 0, 1, 1, 1]
        probabilities = [0.1, 0.9, 0.2, 0.8, 0.8, 0.2, 0.7, 0.6]
        user_ids = [1, 1, 1, 1, 2, 2, 3, 3]

        # user 1: AUC=1 with weight 4; user 2: AUC=0 with weight 2;
        # user 3 is skipped because it has only positive labels.
        self.assertAlmostEqual(
            gauc_score(targets, probabilities, user_ids),
            4.0 / 6.0,
        )

    def test_single_label_task_returns_nan_without_breaking_mean(self):
        targets = {
            "click": [1, 1, 1, 1],
            "like": [0, 1, 0, 1],
        }
        logits = {
            "click": logit([0.6, 0.7, 0.8, 0.9]),
            "like": logit([0.1, 0.9, 0.2, 0.8]),
        }
        metrics = compute_multitask_metrics(
            targets=targets,
            logits=logits,
            user_ids=[1, 1, 2, 2],
            task_names=("click", "like"),
        )

        self.assertTrue(math.isnan(metrics["click_auc"]))
        self.assertTrue(math.isnan(metrics["click_gauc"]))
        self.assertTrue(math.isfinite(metrics["click_logloss"]))
        self.assertEqual(metrics["mean_auc"], metrics["like_auc"])
        self.assertEqual(metrics["mean_gauc"], metrics["like_gauc"])


if __name__ == "__main__":
    unittest.main()
