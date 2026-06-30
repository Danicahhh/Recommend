import unittest

import torch

from recommender.models.two_tower import build_two_tower_model


class RecallSingleBatchTest(unittest.TestCase):
    def test_training_forward_accepts_one_sample(self):
        model = build_two_tower_model(
            user_vocab_size=3,
            item_vocab_size=4,
            gender_vocab_size=2,
            age_vocab_size=2,
            video_category_vocab_size=2,
            embedding_dim=8,
            transformer_heads=2,
            transformer_layers=1,
            user_tower_dims=[16, 8],
            item_tower_dims=[16, 8],
            output_dim=8,
            max_seq_len=10,
        )
        model.train()
        sequence = torch.tensor([[2] + [0] * 9], dtype=torch.long)
        mask = sequence.eq(0)

        scores, _ = model(
            user_ids=torch.tensor([1]),
            item_ids=torch.tensor([2]),
            behavior_sequence=sequence,
            behavior_mask=mask,
            gender_ids=torch.tensor([1]),
            age_ids=torch.tensor([1]),
            video_category_ids=torch.tensor([1]),
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            scores,
            torch.ones_like(scores),
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
    unittest.main()
