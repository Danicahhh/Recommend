import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from recommender.evaluation import compute_topk_retrieval_metrics
from recommender.training.recall import TwoTowerTrainer


class DeterministicTwoTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def get_user_embedding(
        self,
        user_ids,
        behavior_sequence,
        behavior_mask=None,
        gender_ids=None,
        age_ids=None,
    ):
        return torch.nn.functional.one_hot(
            user_ids - 1,
            num_classes=2,
        ).float()

    def get_item_embedding(self, item_ids, video_category_ids=None):
        embeddings = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]],
            device=item_ids.device,
        )
        return embeddings[item_ids - 1]

    def forward(
        self,
        user_ids,
        item_ids,
        behavior_sequence,
        behavior_mask=None,
        gender_ids=None,
        age_ids=None,
        video_category_ids=None,
        return_embeddings=False,
    ):
        user_repr = self.get_user_embedding(user_ids, behavior_sequence)
        item_repr = self.get_item_embedding(item_ids)
        scores = (user_repr * item_repr).sum(dim=1) + self.bias
        embeddings = (
            {"user_repr": user_repr, "item_repr": item_repr}
            if return_embeddings
            else None
        )
        return scores, embeddings


class RecallEvaluationTest(unittest.TestCase):
    def test_topk_metrics_are_macro_averaged_by_user(self):
        recommendations = {
            1: [10, 30, 20],
            2: [50, 60, 40],
        }
        relevant_items = {
            1: {10, 20},
            2: {40},
        }

        metrics = compute_topk_retrieval_metrics(
            recommendations,
            relevant_items,
            ks=[1, 3],
        )

        self.assertEqual(metrics["evaluated_users"], 2.0)
        self.assertAlmostEqual(metrics["recall@1"], 0.25)
        self.assertAlmostEqual(metrics["hit_rate@1"], 0.5)
        self.assertAlmostEqual(metrics["ndcg@1"], 0.5)
        self.assertAlmostEqual(metrics["recall@3"], 1.0)
        self.assertAlmostEqual(metrics["hit_rate@3"], 1.0)
        user_1_ndcg = (
            1.0 + 1.0 / math.log2(4.0)
        ) / (
            1.0 + 1.0 / math.log2(3.0)
        )
        user_2_ndcg = 1.0 / math.log2(4.0)
        self.assertAlmostEqual(metrics["ndcg@3"], (user_1_ndcg + user_2_ndcg) / 2)

    def test_users_without_relevant_items_are_skipped(self):
        metrics = compute_topk_retrieval_metrics(
            {1: [10], 2: [20]},
            {1: set(), 2: {20}},
            ks=[1],
        )

        self.assertEqual(metrics["evaluated_users"], 1.0)
        self.assertEqual(metrics["recall@1"], 1.0)
        self.assertEqual(metrics["hit_rate@1"], 1.0)
        self.assertEqual(metrics["ndcg@1"], 1.0)

    def test_k_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            compute_topk_retrieval_metrics({}, {}, ks=[0])

    def test_trainer_validates_against_the_full_candidate_catalog(self):
        batch = (
            torch.tensor([1, 2]),
            torch.tensor([1, 2]),
            torch.zeros((2, 2), dtype=torch.long),
            torch.ones((2, 2), dtype=torch.bool),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
            torch.ones(2),
        )
        dataset = SimpleNamespace(
            candidate_item_ids=np.asarray([1, 2, 3], dtype=np.int64),
            item_category_by_id=np.zeros(4, dtype=np.int64),
        )
        trainer = TwoTowerTrainer(
            model=DeterministicTwoTower(),
            train_loader=[],
            val_loader=[batch],
            device=torch.device("cpu"),
            learning_rate=1e-3,
            weight_decay=0.0,
            loss_type="bce",
            infonce_temperature=0.07,
            evaluation_dataset=dataset,
            eval_ks=[1, 2],
        )

        metrics = trainer.validate(epoch=1, epochs=1)

        self.assertEqual(metrics["val_evaluated_users"], 2.0)
        self.assertEqual(metrics["val_recall@1"], 1.0)
        self.assertEqual(metrics["val_hit_rate@1"], 1.0)
        self.assertEqual(metrics["val_ndcg@1"], 1.0)


if __name__ == "__main__":
    unittest.main()
