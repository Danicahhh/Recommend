import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from recommender.data_process.recall import TwoTowerDataset
from recommender.retrieval import _validate_mapping_strategy


def mapping_frame():
    rows = []
    for user_id, item_id, gender, age, category in (
        (1, 900, "M", "30", "sports"),
        (2, 100, "F", "20", "music"),
    ):
        row = {
            "user_id": user_id,
            "item_id": item_id,
            "video_category": category,
            "gender": gender,
            "age": age,
            "click": 1,
        }
        row.update({f"hist_{index}": 0 for index in range(1, 11)})
        rows.append(row)
    return pd.DataFrame(rows)


class RecallCheckpointMappingTest(unittest.TestCase):
    def test_legacy_raw_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no longer supported"):
            _validate_mapping_strategy({"item_mapping_mode": "raw"})

    def test_saved_mapping_survives_reordered_input(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            original_path = directory / "original.csv"
            reordered_path = directory / "reordered.csv"
            checkpoint_path = directory / "mapping.pt"
            frame = mapping_frame()
            frame.to_csv(original_path, index=False)
            frame.iloc[::-1].to_csv(reordered_path, index=False)

            original = TwoTowerDataset(str(original_path))
            torch.save(
                {"feature_mappings": original.export_feature_mappings()},
                checkpoint_path,
            )
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            restored = TwoTowerDataset(
                str(reordered_path),
                feature_mappings=checkpoint["feature_mappings"],
            )

        self.assertEqual(
            restored.item_id_to_contiguous,
            original.item_id_to_contiguous,
        )
        self.assertEqual(restored.gender_mapping, original.gender_mapping)
        self.assertEqual(restored.age_mapping, original.age_mapping)
        self.assertEqual(restored.category_mapping, original.category_mapping)
        for model_id in restored.candidate_item_ids:
            raw_id = restored.raw_item_id(int(model_id))
            self.assertEqual(
                restored.item_id_to_contiguous[raw_id],
                int(model_id),
            )


if __name__ == "__main__":
    unittest.main()
