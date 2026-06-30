import unittest

from main import build_parser


class CliTest(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_rank_train(self):
        args = self.parser.parse_args(["rank", "train"])
        self.assertEqual((args.stage, args.command), ("rank", "train"))
        self.assertEqual((args.val_ratio, args.test_ratio), (0.1, 0.1))
        self.assertEqual(args.epochs, 20)
        self.assertEqual(args.batch_size, 4096)
        self.assertEqual(args.embedding_dim, 32)
        self.assertEqual((args.hidden_dim, args.num_layers), (128, 2))
        self.assertEqual(args.lr, 1e-4)

    def test_rank_ablation(self):
        args = self.parser.parse_args(["rank", "ablation", "--seeds", "1", "2"])
        self.assertEqual(args.seeds, [1, 2])
        self.assertEqual(args.num_experts, 3)

    def test_recall_train(self):
        args = self.parser.parse_args(["recall", "train"])
        self.assertEqual(args.loss_type, "infonce")
        self.assertEqual(args.item_mapping_mode, "contiguous")

    def test_recall_generate(self):
        args = self.parser.parse_args(
            ["recall", "generate", "--checkpoint", "model.pt", "--no-use-faiss"]
        )
        self.assertFalse(args.use_faiss)


if __name__ == "__main__":
    unittest.main()
