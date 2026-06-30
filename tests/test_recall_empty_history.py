import unittest

import torch

from recommender.models.two_tower import TransformerBehaviorEncoder


class RecallEmptyHistoryTest(unittest.TestCase):
    def test_single_empty_history_is_finite_in_evaluation_mode(self):
        encoder = TransformerBehaviorEncoder(
            item_vocab_size=4,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            max_seq_len=10,
        )
        encoder.eval()
        sequence = torch.zeros((1, 10), dtype=torch.long)
        mask = torch.ones((1, 10), dtype=torch.bool)

        with torch.no_grad():
            output = encoder(sequence, mask)

        self.assertEqual(tuple(output.shape), (1, 8))
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))


if __name__ == "__main__":
    unittest.main()
