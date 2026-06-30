import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from recommender.data_process.recall import TwoTowerDataset, build_user_samples
from recommender.retrieval import RecallEngine


def recall_frame():
    rows = []
    for item_id in (10, 1_000_000):
        row = {
            "user_id": 1,
            "item_id": item_id,
            "video_category": "news",
            "gender": "F",
            "age": "20",
            "click": 1,
        }
        row.update({f"hist_{index}": 0 for index in range(1, 11)})
        rows.append(row)
    return pd.DataFrame(rows)


class DeterministicRecallModel(nn.Module):
    def get_item_embedding(self, item_ids, video_category_ids=None):
        values = item_ids.float()
        return torch.stack((values, torch.ones_like(values)), dim=1)

    def get_user_embedding(
        self,
        user_ids,
        behavior_sequence,
        behavior_mask=None,
        gender_ids=None,
        age_ids=None,
    ):
        return torch.tensor([[1.0, 0.0]], device=user_ids.device)

    def predict_batch(self, user_repr, item_reprs):
        return torch.matmul(user_repr, item_reprs.t()).squeeze(0)


class RecallIdTest(unittest.TestCase):
    def test_recall_returns_only_real_raw_ids_and_excludes_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            recall_frame().to_csv(path, index=False)
            dataset = TwoTowerDataset(str(path))
            engine = RecallEngine(
                model=DeterministicRecallModel(),
                device=torch.device("cpu"),
                dataset=dataset,
                use_faiss=False,
                ann_nlist=1,
                ann_nprobe=1,
            )
            samples = build_user_samples(dataset, user_ids=[1])
            recalled = engine.recall(samples, top_k=10)[1]

        recalled_ids = [item_id for item_id, _ in recalled]
        self.assertEqual(set(recalled_ids), {10, 1_000_000})
        self.assertNotIn(0, recalled_ids)
        self.assertEqual(dataset.item_vocab_size, 3)


if __name__ == "__main__":
    unittest.main()
