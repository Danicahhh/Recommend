import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from main import build_parser, main


class CliTest(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_rank_train(self):
        args = self.parser.parse_args(["rank", "train"])
        self.assertEqual((args.stage, args.command), ("rank", "train"))
        self.assertEqual((args.val_ratio, args.test_ratio), (0.1, 0.1))
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.batch_size, 1024)
        self.assertEqual(args.embedding_dim, 32)
        self.assertEqual((args.hidden_dim, args.num_layers), (128, 2))
        self.assertEqual(args.lr, 1e-4)
        self.assertEqual(args.early_stopping_patience, 2)
        self.assertEqual(args.early_stopping_min_delta, 0.001)
        self.assertTrue(args.use_personalized_gate)
        self.assertTrue(args.use_task_bias)

    def test_rank_train_personalized_features_can_be_disabled_independently(self):
        gate_args = self.parser.parse_args(
            ["rank", "train", "--no-use-personalized-gate"]
        )
        self.assertFalse(gate_args.use_personalized_gate)
        self.assertTrue(gate_args.use_task_bias)

        bias_args = self.parser.parse_args(["rank", "train", "--no-use-task-bias"])
        self.assertTrue(bias_args.use_personalized_gate)
        self.assertFalse(bias_args.use_task_bias)

    def test_rank_ablation(self):
        args = self.parser.parse_args(["rank", "ablation", "--seeds", "1", "2"])
        self.assertEqual(args.seeds, [1, 2])
        self.assertEqual(args.num_experts, 3)
        self.assertEqual(args.early_stopping_patience, 2)
        self.assertEqual(args.early_stopping_min_delta, 0.001)

    def test_rank_ablation_does_not_silently_retry_cuda_oom(self):
        handler = Mock(side_effect=RuntimeError("CUDA out of memory"))
        args = SimpleNamespace(
            stage="rank",
            command="ablation",
            seeds=[42],
            handler=handler,
        )
        parser = Mock()
        parser.parse_args.return_value = args

        with patch("main.build_parser", return_value=parser):
            with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
                main([])

        handler.assert_called_once_with(args)

    def test_recall_train(self):
        args = self.parser.parse_args(["recall", "train"])
        self.assertEqual(args.loss_type, "infonce")
        self.assertFalse(hasattr(args, "item_mapping_mode"))

    def test_recall_train_rejects_removed_raw_mapping_option(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(
                ["recall", "train", "--item-mapping-mode", "raw"]
            )

    def test_recall_generate(self):
        args = self.parser.parse_args(
            ["recall", "generate", "--checkpoint", "model.pt", "--no-use-faiss"]
        )
        self.assertFalse(args.use_faiss)


if __name__ == "__main__":
    unittest.main()
