import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.rank_mmoe import RankMMOEModel, RankMultiTaskLoss
from train.rank.train_rank_mmoe import CTRDataset, TASK_NAMES, build_maps, move_batch


ABLATIONS = (
    {
        "name": "A0_plain_mmoe_id_only",
        "use_personalized_mmoe": False,
        "use_attribute_expert_mask": False,
        "use_personalized_gate": False,
        "use_task_bias": False,
        "use_target_attention": False,
        "use_auxiliary_loss": False,
        "use_item_side_features": False,
        "use_profile_features": False,
    },
    {
        "name": "A1_item_side_features",
        "use_personalized_mmoe": False,
        "use_attribute_expert_mask": False,
        "use_personalized_gate": False,
        "use_task_bias": False,
        "use_target_attention": False,
        "use_auxiliary_loss": False,
        "use_item_side_features": True,
        "use_profile_features": False,
    },
    {
        "name": "A2_attribute_expert_mask",
        "use_personalized_mmoe": False,
        "use_attribute_expert_mask": True,
        "use_personalized_gate": False,
        "use_task_bias": False,
        "use_target_attention": False,
        "use_auxiliary_loss": False,
        "use_item_side_features": True,
        "use_profile_features": False,
    },
    {
        "name": "A3_personalized_gate",
        "use_personalized_mmoe": True,
        "use_attribute_expert_mask": True,
        "use_personalized_gate": True,
        "use_task_bias": False,
        "use_target_attention": False,
        "use_auxiliary_loss": False,
        "use_item_side_features": True,
        "use_profile_features": False,
    },
    {
        "name": "A4_task_bias",
        "use_personalized_mmoe": True,
        "use_attribute_expert_mask": True,
        "use_personalized_gate": True,
        "use_task_bias": True,
        "use_target_attention": False,
        "use_auxiliary_loss": False,
        "use_item_side_features": True,
        "use_profile_features": False,
    },
    {
        "name": "A5_target_attention",
        "use_personalized_mmoe": True,
        "use_attribute_expert_mask": True,
        "use_personalized_gate": True,
        "use_task_bias": True,
        "use_target_attention": True,
        "use_auxiliary_loss": False,
        "use_item_side_features": True,
        "use_profile_features": False,
    },
    {
        "name": "A6_two_tower_aux",
        "use_personalized_mmoe": True,
        "use_attribute_expert_mask": True,
        "use_personalized_gate": True,
        "use_task_bias": True,
        "use_target_attention": True,
        "use_auxiliary_loss": True,
        "use_item_side_features": True,
        "use_profile_features": False,
    },
    {
        "name": "A7_profile_features",
        "use_personalized_mmoe": True,
        "use_attribute_expert_mask": True,
        "use_personalized_gate": True,
        "use_task_bias": True,
        "use_target_attention": True,
        "use_auxiliary_loss": True,
        "use_item_side_features": True,
        "use_profile_features": True,
    },
    {
        "name": "B1_target_attention_only",
        "use_personalized_mmoe": False,
        "use_attribute_expert_mask": False,
        "use_personalized_gate": False,
        "use_task_bias": False,
        "use_target_attention": True,
        "use_auxiliary_loss": False,
        "use_item_side_features": True,
        "use_profile_features": False,
    },
    {
        "name": "B2_aux_only",
        "use_personalized_mmoe": False,
        "use_attribute_expert_mask": False,
        "use_personalized_gate": False,
        "use_task_bias": False,
        "use_target_attention": False,
        "use_auxiliary_loss": True,
        "use_item_side_features": True,
        "use_profile_features": False,
    },
)


def train_one_epoch(
    model: RankMMOEModel,
    criterion: RankMultiTaskLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_item_side_features: bool,
    use_profile_features: bool,
    use_auxiliary_loss: bool,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad()
        logits, auxiliary = forward_model(
            model,
            batch,
            use_item_side_features=use_item_side_features,
            use_profile_features=use_profile_features,
            return_auxiliary=True,
        )
        auxiliary_target = batch["targets"]["click"] if use_auxiliary_loss else None
        loss, _ = criterion(logits, batch["targets"], auxiliary, auxiliary_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: RankMMOEModel,
    criterion: RankMultiTaskLoss,
    loader: DataLoader,
    device: torch.device,
    use_item_side_features: bool,
    use_profile_features: bool,
    use_auxiliary_loss: bool,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    targets: Dict[str, List[float]] = {task: [] for task in TASK_NAMES}
    scores: Dict[str, List[float]] = {task: [] for task in TASK_NAMES}

    for batch in loader:
        batch = move_batch(batch, device)
        logits, auxiliary = forward_model(
            model,
            batch,
            use_item_side_features=use_item_side_features,
            use_profile_features=use_profile_features,
            return_auxiliary=True,
        )
        auxiliary_target = batch["targets"]["click"] if use_auxiliary_loss else None
        loss, _ = criterion(logits, batch["targets"], auxiliary, auxiliary_target)
        total_loss += loss.item()

        for task in TASK_NAMES:
            targets[task].extend(batch["targets"][task].detach().cpu().view(-1).tolist())
            scores[task].extend(torch.sigmoid(logits[task]).detach().cpu().view(-1).tolist())

    metrics: Dict[str, float] = {"val_loss": total_loss / max(len(loader), 1)}
    auc_values = []
    for task in TASK_NAMES:
        task_targets = targets[task]
        task_scores = scores[task]
        metrics[f"{task}_logloss"] = float(log_loss(task_targets, task_scores, labels=[0, 1]))
        if len(set(task_targets)) < 2:
            metrics[f"{task}_auc"] = float("nan")
        else:
            metrics[f"{task}_auc"] = float(roc_auc_score(task_targets, task_scores))
            auc_values.append(metrics[f"{task}_auc"])

    metrics["mean_auc"] = float(sum(auc_values) / len(auc_values)) if auc_values else float("nan")
    return metrics


def forward_model(
    model: RankMMOEModel,
    batch: Dict,
    use_item_side_features: bool,
    use_profile_features: bool,
    return_auxiliary: bool,
):
    return model(
        user_ids=batch["user_ids"],
        item_ids=batch["item_ids"],
        video_category_ids=batch["video_category_ids"] if use_item_side_features else None,
        gender_ids=batch["gender_ids"] if use_profile_features else None,
        age_ids=batch["age_ids"] if use_profile_features else None,
        behavior_sequence=batch["behavior_sequence"],
        behavior_mask=batch["behavior_mask"],
        return_auxiliary=return_auxiliary,
    )


def run_one_seed(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    seed: int,
    output_dir: Path,
) -> List[Dict]:
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    seed_frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_index = int(len(seed_frame) * (1.0 - args.val_ratio))
    train_frame = seed_frame.iloc[:split_index].reset_index(drop=True)
    val_frame = seed_frame.iloc[split_index:].reset_index(drop=True)
    user_map, item_map, category_map, gender_map, age_map = build_maps(frame)

    train_dataset = CTRDataset(train_frame, user_map, item_map, category_map, gender_map, age_map)
    val_dataset = CTRDataset(val_frame, user_map, item_map, category_map, gender_map, age_map)
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=train_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    seed_rows = []

    for exp in ABLATIONS:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = RankMMOEModel(
            user_vocab_size=len(user_map),
            item_vocab_size=len(item_map) + 1,
            category_vocab_size=len(category_map) + 1,
            gender_vocab_size=len(gender_map) + 1,
            age_vocab_size=len(age_map) + 1,
            task_names=TASK_NAMES,
            embedding_dim=args.embedding_dim,
            num_experts=args.num_experts,
            num_heads=args.num_heads,
            dropout=args.dropout,
            use_personalized_mmoe=exp["use_personalized_mmoe"],
            use_attribute_expert_mask=exp["use_attribute_expert_mask"],
            use_personalized_gate=exp["use_personalized_gate"],
            use_task_bias=exp["use_task_bias"],
            use_target_attention=exp["use_target_attention"],
        ).to(device)
        criterion = RankMultiTaskLoss(
            task_names=TASK_NAMES,
            auxiliary_weight=args.auxiliary_weight if exp["use_auxiliary_loss"] else 0.0,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        history = []
        print(f"seed={seed} running {exp['name']}")
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                criterion,
                train_loader,
                optimizer,
                device,
                use_item_side_features=exp["use_item_side_features"],
                use_profile_features=exp["use_profile_features"],
                use_auxiliary_loss=exp["use_auxiliary_loss"],
            )
            metrics = evaluate(
                model,
                criterion,
                val_loader,
                device,
                use_item_side_features=exp["use_item_side_features"],
                use_profile_features=exp["use_profile_features"],
                use_auxiliary_loss=exp["use_auxiliary_loss"],
            )
            epoch_row = {"epoch": epoch, "train_loss": train_loss, **metrics}
            history.append(epoch_row)
            print(
                f"seed={seed} {exp['name']} epoch={epoch} train_loss={train_loss:.6f} "
                f"val_loss={metrics['val_loss']:.6f} mean_auc={metrics['mean_auc']:.6f}"
            )

        best = max(history, key=lambda row: row["mean_auc"])
        row = {"seed": seed, **exp, **best}
        seed_rows.append(row)
        with (seed_dir / f"{exp['name']}_history.json").open("w", encoding="utf-8") as f:
            json.dump({"seed": seed, "config": exp, "history": history}, f, ensure_ascii=False, indent=2)

    pd.DataFrame(seed_rows).to_csv(seed_dir / "summary.csv", index=False, encoding="utf-8-sig")
    return seed_rows


def summarize_results(per_seed: pd.DataFrame) -> pd.DataFrame:
    config_columns = [
        "name",
        "use_personalized_mmoe",
        "use_attribute_expert_mask",
        "use_personalized_gate",
        "use_task_bias",
        "use_target_attention",
        "use_auxiliary_loss",
        "use_item_side_features",
        "use_profile_features",
    ]
    metric_columns = [
        "epoch",
        "train_loss",
        "val_loss",
        "click_logloss",
        "click_auc",
        "follow_logloss",
        "follow_auc",
        "like_logloss",
        "like_auc",
        "share_logloss",
        "share_auc",
        "mean_auc",
    ]

    rows = []
    for name, group in per_seed.groupby("name", sort=False):
        row = {column: group.iloc[0][column] for column in config_columns}
        for metric in metric_columns:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def run_ablation(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.data_path, nrows=args.sample_rows)

    all_rows = []
    for seed in args.seeds:
        all_rows.extend(run_one_seed(args, frame, seed, args.output_dir))

    per_seed = pd.DataFrame(all_rows)
    per_seed.to_csv(args.output_dir / "per_seed_results.csv", index=False, encoding="utf-8-sig")
    summary = summarize_results(per_seed)
    summary.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")

    print(f"saved per-seed results to {args.output_dir / 'per_seed_results.csv'}")
    print(f"saved summary to {args.output_dir / 'summary.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "ctr_data_500k.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "train" / "rank" / "result" / "rank_mmoe_ablation")
    parser.add_argument("--sample-rows", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--num-experts", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--auxiliary-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.seeds is None:
        args.seeds = [args.seed]
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    try:
        run_ablation(parsed_args)
    except RuntimeError as exc:
        is_cuda_oom = "CUDA out of memory" in str(exc)
        if is_cuda_oom and parsed_args.device.startswith("cuda") and parsed_args.batch_size > 256:
            print("CUDA OOM with batch_size=", parsed_args.batch_size, "; retrying with batch_size=256")
            torch.cuda.empty_cache()
            parsed_args.batch_size = 256
            run_ablation(parsed_args)
        else:
            raise
