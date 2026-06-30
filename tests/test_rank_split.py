import unittest

import pandas as pd

from recommender.training.rank import split_rank_frame


class RankSplitTest(unittest.TestCase):
    def test_official_eight_one_one_split(self):
        frame = pd.DataFrame({"row_id": range(100)})

        train, validation, test = split_rank_frame(
            frame,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=100,
        )

        self.assertEqual((len(train), len(validation), len(test)), (80, 10, 10))
        all_ids = set(train["row_id"]) | set(validation["row_id"]) | set(
            test["row_id"]
        )
        self.assertEqual(len(all_ids), 100)
        self.assertTrue(set(train["row_id"]).isdisjoint(validation["row_id"]))
        self.assertTrue(set(train["row_id"]).isdisjoint(test["row_id"]))
        self.assertTrue(set(validation["row_id"]).isdisjoint(test["row_id"]))

    def test_split_is_reproducible(self):
        frame = pd.DataFrame({"row_id": range(20)})
        first = split_rank_frame(frame, 0.1, 0.1, seed=100)
        second = split_rank_frame(frame, 0.1, 0.1, seed=100)

        for first_part, second_part in zip(first, second):
            self.assertListEqual(
                first_part["row_id"].tolist(),
                second_part["row_id"].tolist(),
            )


if __name__ == "__main__":
    unittest.main()
