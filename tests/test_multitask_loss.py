import math
import unittest

import pandas as pd
import torch
import torch.nn.functional as F

from recommender.ablation import compute_pos_weights
from recommender.models.mmoe import RankMultiTaskLoss


class RankMultiTaskLossTest(unittest.TestCase):
    def test_pos_weight_can_use_default_sample_mean(self):
        logits = torch.tensor([[0.3], [-0.7], [1.1]])
        targets = torch.tensor([[1.0], [0.0], [0.0]])
        pos_weight = 3.0
        criterion = RankMultiTaskLoss(
            task_names=("share",),
            pos_weights={"share": pos_weight},
            normalize_pos_weight=False,
            auxiliary_weight=0.0,
        )

        total, losses = criterion(
            {"share": logits},
            {"share": targets},
        )
        expected = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=torch.tensor([pos_weight]),
            reduction="mean",
        )

        self.assertTrue(torch.allclose(losses["share"], expected))
        self.assertTrue(torch.allclose(total, expected))

    def test_pos_weight_loss_is_divided_by_effective_weight_sum(self):
        logits = torch.tensor([[0.3], [-0.7], [1.1]])
        targets = torch.tensor([[1.0], [0.0], [0.0]])
        pos_weight = 3.0
        criterion = RankMultiTaskLoss(
            task_names=("share",),
            pos_weights={"share": pos_weight},
            normalize_pos_weight=True,
            auxiliary_weight=0.0,
        )

        total, losses = criterion(
            {"share": logits},
            {"share": targets},
        )
        element_losses = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=torch.tensor([pos_weight]),
            reduction="none",
        )
        expected = element_losses.sum() / torch.tensor(pos_weight + 2.0)

        self.assertTrue(torch.allclose(losses["share"], expected))
        self.assertTrue(torch.allclose(total, expected))

    def test_gradnorm_builds_weight_gradient_and_preserves_weight_sum(self):
        shared = torch.nn.Parameter(torch.tensor(0.5))
        criterion = RankMultiTaskLoss(
            task_names=("click", "share"),
            task_weighting_method="gradnorm",
            auxiliary_weight=0.0,
        )
        losses = {
            "click": (shared - 1.0).pow(2),
            "share": (2.0 * shared + 1.0).pow(2),
        }

        objective = criterion.gradnorm_objective(losses, (shared,))
        objective.backward()
        self.assertTrue(torch.isfinite(criterion.gradnorm_weights.grad).all())

        with torch.no_grad():
            criterion.gradnorm_weights.add_(torch.tensor([-0.6, 0.8]))
        criterion.normalize_gradnorm_weights()
        self.assertAlmostEqual(
            criterion.gradnorm_weights.sum().item(),
            2.0,
            places=5,
        )

    def test_uncertainty_weights_receive_gradients(self):
        criterion = RankMultiTaskLoss(
            task_names=("click", "share"),
            task_weighting_method="uncertainty",
            auxiliary_weight=0.0,
        )
        predictions = {
            "click": torch.tensor([[0.2]], requires_grad=True),
            "share": torch.tensor([[-0.4]], requires_grad=True),
        }
        targets = {
            "click": torch.tensor([[1.0]]),
            "share": torch.tensor([[0.0]]),
        }

        total, _ = criterion(predictions, targets)
        total.backward()
        self.assertTrue(torch.isfinite(criterion.log_task_variances.grad).all())

    def test_pos_weights_use_training_counts_and_leave_click_at_one(self):
        frame = pd.DataFrame(
            {
                "click": [1, 0, 0, 1],
                "follow": [1, 0, 0, 0],
                "like": [1, 1, 0, 0],
                "share": [0, 0, 1, 0],
            }
        )
        weights = compute_pos_weights(
            frame,
            ("click", "follow", "like", "share"),
        )

        self.assertEqual(weights["click"], 1.0)
        self.assertEqual(weights["follow"], 3.0)
        self.assertEqual(weights["like"], 1.0)
        self.assertEqual(weights["share"], 3.0)
        self.assertTrue(all(math.isfinite(value) for value in weights.values()))


if __name__ == "__main__":
    unittest.main()
