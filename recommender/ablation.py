import json
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from recommender.data_process.rank import TASK_NAMES, RankDataset, build_feature_maps
from recommender.evaluation import compute_multitask_metrics
from recommender.models.mmoe import RankMMOEModel, RankMultiTaskLoss
from recommender.training.rank import move_batch, sample_click_negatives


ABLATIONS = (
    {
        "name": "A0_plain_mmoe_id_only",
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
    progress_desc: str = "ablation train",
) -> float:
    model.train()
    total_loss = 0.0
    progress = tqdm(
        loader,
        desc=progress_desc,
        unit="batch",
        dynamic_ncols=True,
    )
    for batch_index, batch in enumerate(progress, start=1):
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
        progress.set_postfix(loss=f"{total_loss / batch_index:.6f}")
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
    progress_desc: str = "ablation validate",
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    targets: Dict[str, List[float]] = {task: [] for task in TASK_NAMES}
    logits_by_task: Dict[str, List[float]] = {task: [] for task in TASK_NAMES}
    user_ids: List[int] = []

    progress = tqdm(
        loader,
        desc=progress_desc,
        unit="batch",
        dynamic_ncols=True,
    )
    for batch_index, batch in enumerate(progress, start=1):
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
        progress.set_postfix(loss=f"{total_loss / batch_index:.6f}")
        user_ids.extend(batch["user_ids"].detach().cpu().view(-1).tolist())

        for task in TASK_NAMES:
            targets[task].extend(
                batch["targets"][task].detach().cpu().view(-1).tolist()
            )
            logits_by_task[task].extend(logits[task].detach().cpu().view(-1).tolist())

    metrics: Dict[str, float] = {"val_loss": total_loss / max(len(loader), 1)}
    metrics.update(
        compute_multitask_metrics(
            targets=targets,
            logits=logits_by_task,
            user_ids=user_ids,
            task_names=TASK_NAMES,
        )
    )
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
        video_category_ids=batch["video_category_ids"]
        if use_item_side_features
        else None,
        gender_ids=batch["gender_ids"] if use_profile_features else None,
        age_ids=batch["age_ids"] if use_profile_features else None,
        behavior_sequence=batch["behavior_sequence"],
        behavior_mask=batch["behavior_mask"],
        return_auxiliary=return_auxiliary,
    )


def build_ablation_model(args, exp: Dict, vocab_sizes: Dict[str, int]):
    """按 CLI 参数构建消融模型，便于保证训练和测试使用同一套配置。"""
    hidden_units = [args.hidden_dim] * args.num_layers
    return RankMMOEModel(
        user_vocab_size=vocab_sizes["user"],
        item_vocab_size=vocab_sizes["item"],
        category_vocab_size=vocab_sizes["category"],
        gender_vocab_size=vocab_sizes["gender"],
        age_vocab_size=vocab_sizes["age"],
        task_names=TASK_NAMES,
        embedding_dim=args.embedding_dim,
        num_experts=args.num_experts,
        num_heads=args.num_heads,
        expert_units=hidden_units,
        gate_units=hidden_units,
        tower_units=hidden_units,
        dropout=args.dropout,
        use_attribute_expert_mask=exp["use_attribute_expert_mask"],
        use_personalized_gate=exp["use_personalized_gate"],
        use_task_bias=exp["use_task_bias"],
        use_target_attention=exp["use_target_attention"],
    )


def build_ablation_train_loader(
    dataset: RankDataset,
    batch_size: int,
    seed: int,
) -> DataLoader:
    """每个消融项使用相同 seed 的独立 generator，确保 batch 顺序一致。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )


def run_one_seed(
    args,
    frame: pd.DataFrame,
    seed: int,
    output_dir: Path,
) -> List[Dict]:
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    sampled_frame = sample_click_negatives(frame, seed=seed)
    seed_frame = sampled_frame.sample(frac=1.0, random_state=seed).reset_index(
        drop=True
    )
    split_index = int(len(seed_frame) * (1.0 - args.val_ratio))
    train_frame = seed_frame.iloc[:split_index].reset_index(drop=True)
    val_frame = seed_frame.iloc[split_index:].reset_index(drop=True)
    user_map, item_map, category_map, gender_map, age_map = build_feature_maps(
        sampled_frame
    )

    train_dataset = RankDataset(
        train_frame, user_map, item_map, category_map, gender_map, age_map
    )
    val_dataset = RankDataset(
        val_frame, user_map, item_map, category_map, gender_map, age_map
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    device = torch.device(args.device)
    seed_rows = []
    vocab_sizes = {
        "user": len(user_map),
        "item": len(item_map) + 1,
        "category": len(category_map) + 1,
        "gender": len(gender_map) + 1,
        "age": len(age_map) + 1,
    }

    for exp in ABLATIONS:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = build_ablation_model(args, exp, vocab_sizes).to(device)
        train_loader = build_ablation_train_loader(
            train_dataset,
            batch_size=args.batch_size,
            seed=seed,
        )
        criterion = RankMultiTaskLoss(
            task_names=TASK_NAMES,
            auxiliary_weight=args.auxiliary_weight
            if exp["use_auxiliary_loss"]
            else 0.0,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

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
                progress_desc=(
                    f"ablation train {exp['name']} {epoch}/{args.epochs}"
                ),
            )
            metrics = evaluate(
                model,
                criterion,
                val_loader,
                device,
                use_item_side_features=exp["use_item_side_features"],
                use_profile_features=exp["use_profile_features"],
                use_auxiliary_loss=exp["use_auxiliary_loss"],
                progress_desc=(
                    f"ablation validate {exp['name']} {epoch}/{args.epochs}"
                ),
            )
            epoch_row = {"epoch": epoch, "train_loss": train_loss, **metrics}
            history.append(epoch_row)
            print(
                f"seed={seed} {exp['name']} epoch={epoch} train_loss={train_loss:.6f} "
                f"val_loss={metrics['val_loss']:.6f} "
                f"mean_auc={metrics['mean_auc']:.6f} "
                f"mean_gauc={metrics['mean_gauc']:.6f} "
                f"mean_logloss={metrics['mean_logloss']:.6f}"
            )

        best = max(
            history,
            key=lambda row: (
                row["mean_gauc"] if math.isfinite(row["mean_gauc"]) else float("-inf")
            ),
        )
        if not any(math.isfinite(row["mean_gauc"]) for row in history):
            print(
                f"warning: seed={seed} {exp['name']} has no computable mean_gauc; "
                "retaining the first epoch as best"
            )
        row = {"seed": seed, **exp, **best}
        seed_rows.append(row)
        with (seed_dir / f"{exp['name']}_history.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "seed": seed,
                    "config": exp,
                    "model_hyperparameters": {
                        "embedding_dim": args.embedding_dim,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "num_experts": args.num_experts,
                        "num_heads": args.num_heads,
                        "dropout": args.dropout,
                    },
                    "history": history,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    pd.DataFrame(seed_rows).to_csv(
        seed_dir / "summary.csv", index=False, encoding="utf-8-sig"
    )
    return seed_rows


def summarize_results(per_seed: pd.DataFrame) -> pd.DataFrame:
    config_columns = [
        "name",
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
        "click_auc",
        "click_gauc",
        "click_logloss",
        "follow_auc",
        "follow_gauc",
        "follow_logloss",
        "like_auc",
        "like_gauc",
        "like_logloss",
        "share_auc",
        "share_gauc",
        "share_logloss",
        "mean_auc",
        "mean_gauc",
        "mean_logloss",
    ]

    rows = []
    for name, group in per_seed.groupby("name", sort=False):
        row = {column: group.iloc[0][column] for column in config_columns}
        for metric in metric_columns:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def run_rank_ablation(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.data_path, nrows=args.sample_rows)

    all_rows = []
    for seed in args.seeds:
        all_rows.extend(run_one_seed(args, frame, seed, args.output_dir))

    per_seed = pd.DataFrame(all_rows)
    per_seed.to_csv(
        args.output_dir / "per_seed_results.csv", index=False, encoding="utf-8-sig"
    )
    summary = summarize_results(per_seed)
    summary.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")

    print(f"saved per-seed results to {args.output_dir / 'per_seed_results.csv'}")
    print(f"saved summary to {args.output_dir / 'summary.csv'}")
