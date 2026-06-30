import argparse
from pathlib import Path

import torch

from recommender.ablation import run_rank_ablation
from recommender.retrieval import run_recall_generation
from recommender.training.rank import run_rank_training
from recommender.training.recall import run_recall_training


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "dataset" / "ctr_data_1M.csv"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def add_rank_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--auxiliary-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", default=DEFAULT_DEVICE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="多任务推荐系统")
    stages = parser.add_subparsers(dest="stage", required=True)

    rank_parser = stages.add_parser("rank", help="MMoE 排序")
    rank_commands = rank_parser.add_subparsers(dest="command", required=True)

    rank_train = rank_commands.add_parser("train", help="训练排序模型")
    add_rank_model_arguments(rank_train)
    rank_train.add_argument("--baseline-mmoe", action="store_true")
    rank_train.add_argument("--mean-pooling", action="store_true")
    rank_train.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "rank",
    )
    rank_train.set_defaults(handler=run_rank_training)

    rank_ablation = rank_commands.add_parser("ablation", help="运行排序消融实验")
    add_rank_model_arguments(rank_ablation)
    rank_ablation.set_defaults(
        embedding_dim=32,
        num_experts=3,
        sample_rows=20000,
    )
    rank_ablation.add_argument("--seeds", type=int, nargs="+", default=None)
    rank_ablation.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "rank" / "ablation",
    )
    rank_ablation.set_defaults(handler=run_rank_ablation)

    recall_parser = stages.add_parser("recall", help="双塔召回")
    recall_commands = recall_parser.add_subparsers(dest="command", required=True)

    recall_train = recall_commands.add_parser("train", help="训练双塔模型")
    recall_train.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    recall_train.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "recall"
    )
    recall_train.add_argument("--sample-rows", type=int, default=None)
    recall_train.add_argument("--epochs", type=int, default=6)
    recall_train.add_argument("--batch-size", type=int, default=128)
    recall_train.add_argument("--max-seq-len", type=int, default=10)
    recall_train.add_argument(
        "--loss-type", choices=("bce", "infonce"), default="infonce"
    )
    recall_train.add_argument(
        "--item-mapping-mode", choices=("contiguous", "raw"), default="contiguous"
    )
    recall_train.add_argument("--infonce-temperature", type=float, default=0.07)
    recall_train.add_argument("--embedding-dim", type=int, default=32)
    recall_train.add_argument("--transformer-heads", type=int, default=2)
    recall_train.add_argument("--transformer-layers", type=int, default=1)
    recall_train.add_argument(
        "--user-tower-dims", type=int, nargs="+", default=[128, 64]
    )
    recall_train.add_argument(
        "--item-tower-dims", type=int, nargs="+", default=[128, 64]
    )
    recall_train.add_argument("--output-dim", type=int, default=32)
    recall_train.add_argument("--dropout", type=float, default=0.2)
    recall_train.add_argument("--temperature", type=float, default=0.05)
    recall_train.add_argument("--lr", type=float, default=1e-3)
    recall_train.add_argument("--weight-decay", type=float, default=1e-5)
    recall_train.add_argument("--val-ratio", type=float, default=0.2)
    recall_train.add_argument("--seed", type=int, default=42)
    recall_train.add_argument("--device", default=DEFAULT_DEVICE)
    recall_train.set_defaults(handler=run_recall_training)

    recall_generate = recall_commands.add_parser("generate", help="加载双塔模型生成候选集")
    recall_generate.add_argument("--checkpoint", type=Path, required=True)
    recall_generate.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    recall_generate.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "outputs" / "recall" / "recall_results.json",
    )
    recall_generate.add_argument("--sample-rows", type=int, default=None)
    recall_generate.add_argument("--top-k", type=int, default=20)
    recall_generate.add_argument("--num-users", type=int, default=20)
    recall_generate.add_argument(
        "--use-faiss", action=argparse.BooleanOptionalAction, default=True
    )
    recall_generate.add_argument("--rebuild-faiss", action="store_true")
    recall_generate.add_argument("--ann-nlist", type=int, default=4096)
    recall_generate.add_argument("--ann-nprobe", type=int, default=32)
    recall_generate.add_argument("--device", default=DEFAULT_DEVICE)
    recall_generate.set_defaults(handler=run_recall_generation)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "seeds", None) is None and args.stage == "rank":
        args.seeds = [args.seed]
    try:
        return args.handler(args)
    except RuntimeError as error:
        is_oom = "CUDA out of memory" in str(error)
        if (
            is_oom
            and args.stage == "rank"
            and args.command == "ablation"
            and args.device.startswith("cuda")
            and args.batch_size > 256
        ):
            print(f"CUDA OOM with batch_size={args.batch_size}; retrying with 256")
            torch.cuda.empty_cache()
            args.batch_size = 256
            return args.handler(args)
        raise


if __name__ == "__main__":
    main()
