import tempfile
import unittest
from pathlib import Path

import pandas as pd

from recommender.data_process.rank import HIST_COLUMNS, RankDataset, build_feature_maps
from recommender.data_process.recall import TwoTowerDataset


def sample_frame():
    rows = []
    for index in range(4):
        row = {
            "user_id": 10 + index // 2,
            "item_id": 100 + index,
            "video_category": index % 2,
            "gender": index % 2,
            "age": 20 + index,
            "click": index % 2,
            "follow": 0,
            "like": (index + 1) % 2,
            "share": 0,
        }
        row.update(
            {
                column: (100 + index - position if position <= index else "\\N")
                for position, column in enumerate(HIST_COLUMNS, start=1)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


class DataTest(unittest.TestCase):
    def test_rank_dataset_keeps_user_groups_and_padding(self):
        frame = sample_frame()
        maps = build_feature_maps(frame)
        dataset = RankDataset(frame, *maps)
        self.assertEqual(len(dataset), len(frame))
        self.assertEqual(dataset.user_ids[0].item(), dataset.user_ids[1].item())
        self.assertTrue(dataset.behavior_mask[0].all().item())
        self.assertFalse(dataset.behavior_mask[3, 0].item())

    def test_recall_dataset_contiguous_mapping_and_row_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            sample_frame().to_csv(path, index=False)
            dataset = TwoTowerDataset(
                str(path),
                max_seq_len=10,
                item_mapping_mode="contiguous",
                sample_rows=3,
            )
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.item_ids.min(), 1)
        self.assertEqual(dataset.behavior_sequences.shape, (3, 10))


if __name__ == "__main__":
    unittest.main()
