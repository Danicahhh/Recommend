import unittest
from types import SimpleNamespace

import torch
from torch.utils.data import TensorDataset

from recommender.ablation import (
    ABLATIONS,
    build_ablation_model,
    build_ablation_train_loader,
)


def arguments(hidden_dim=12, num_layers=3):
    return SimpleNamespace(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        embedding_dim=8,
        num_experts=2,
        num_heads=2,
        dropout=0.0,
    )


class AblationConfigTest(unittest.TestCase):
    def test_hidden_dim_and_num_layers_reach_model(self):
        vocab_sizes = {
            "user": 4,
            "item": 6,
            "category": 3,
            "gender": 3,
            "age": 4,
        }
        model = build_ablation_model(arguments(), ABLATIONS[0], vocab_sizes)
        linear_layers = [
            layer
            for layer in model.mmoe.experts[0].mlp.net
            if isinstance(layer, torch.nn.Linear)
        ]

        self.assertEqual(len(linear_layers), 3)
        self.assertTrue(all(layer.out_features == 12 for layer in linear_layers))

    def test_each_experiment_gets_the_same_seeded_batch_order(self):
        dataset = TensorDataset(torch.arange(20))
        first = build_ablation_train_loader(dataset, batch_size=4, seed=17)
        second = build_ablation_train_loader(dataset, batch_size=4, seed=17)

        first_order = torch.cat([batch[0] for batch in first]).tolist()
        second_order = torch.cat([batch[0] for batch in second]).tolist()
        self.assertEqual(first_order, second_order)


if __name__ == "__main__":
    unittest.main()
