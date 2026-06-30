import math
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from recommender.data_process.rank import TASK_NAMES, RankDataset, build_feature_maps
from recommender.evaluation import compute_multitask_metrics
from recommender.models.mmoe import RankMMOEModel, RankMultiTaskLoss


def run_rank_training(args) -> None:
    frame = pd.read_csv(args.data_path, nrows=args.sample_rows)
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    if len(frame) < 2:
        raise ValueError("at least two rows are required for a train/validation split")

    shuffled_frame = frame.sample(frac=1.0, random_state=args.seed).reset_index(
        drop=True
    )
    split_index = int(len(shuffled_frame) * (1.0 - args.val_ratio))
    split_index = min(max(split_index, 1), len(shuffled_frame) - 1)
    train_frame = shuffled_frame.iloc[:split_index].reset_index(drop=True)
    val_frame = shuffled_frame.iloc[split_index:].reset_index(drop=True)

    user_map, item_map, category_map, gender_map, age_map = build_feature_maps(frame)
    train_dataset = RankDataset(
        train_frame, user_map, item_map, category_map, gender_map, age_map
    )
    val_dataset = RankDataset(
        val_frame, user_map, item_map, category_map, gender_map, age_map
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
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
        use_personalized_mmoe=not args.baseline_mmoe,
        use_target_attention=not args.mean_pooling,
    ).to(device)
    criterion = RankMultiTaskLoss(
        task_names=TASK_NAMES, auxiliary_weight=args.auxiliary_weight
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_epoch = 1
    best_metrics = None
    best_gauc = float("-inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad()
            logits, auxiliary = model(
                user_ids=batch["user_ids"],
                item_ids=batch["item_ids"],
                video_category_ids=batch["video_category_ids"],
                gender_ids=batch["gender_ids"],
                age_ids=batch["age_ids"],
                behavior_sequence=batch["behavior_sequence"],
                behavior_mask=batch["behavior_mask"],
                return_auxiliary=True,
            )
            loss, _ = criterion(
                logits, batch["targets"], auxiliary, batch["targets"]["click"]
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        metrics = evaluate(model, criterion, val_loader, device)
        if best_metrics is None:
            best_epoch = epoch
            best_metrics = metrics
        if math.isfinite(metrics["mean_gauc"]) and metrics["mean_gauc"] > best_gauc:
            best_epoch = epoch
            best_metrics = metrics
            best_gauc = metrics["mean_gauc"]

        print(
            f"epoch={epoch} train_loss={avg_loss:.6f} "
            f"val_loss={metrics['val_loss']:.6f} "
            f"mean_auc={metrics['mean_auc']:.6f} "
            f"mean_gauc={metrics['mean_gauc']:.6f} "
            f"mean_logloss={metrics['mean_logloss']:.6f}"
        )
        for task in TASK_NAMES:
            print(
                f"  {task}: AUC={metrics[f'{task}_auc']:.6f} "
                f"GAUC={metrics[f'{task}_gauc']:.6f} "
                f"LogLoss={metrics[f'{task}_logloss']:.6f}"
            )

    if best_gauc == float("-inf"):
        print(
            "warning: mean_gauc could not be computed for any epoch; "
            "retaining the first epoch as best"
        )
    print(f"best_epoch={best_epoch} " f"mean_gauc={best_metrics['mean_gauc']:.6f}")


@torch.no_grad()
def evaluate(
    model: RankMMOEModel,
    criterion: RankMultiTaskLoss,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    targets: Dict[str, List[float]] = {task: [] for task in TASK_NAMES}
    logits_by_task: Dict[str, List[float]] = {task: [] for task in TASK_NAMES}
    user_ids: List[int] = []

    for batch in loader:
        batch = move_batch(batch, device)
        logits, auxiliary = model(
            user_ids=batch["user_ids"],
            item_ids=batch["item_ids"],
            video_category_ids=batch["video_category_ids"],
            gender_ids=batch["gender_ids"],
            age_ids=batch["age_ids"],
            behavior_sequence=batch["behavior_sequence"],
            behavior_mask=batch["behavior_mask"],
            return_auxiliary=True,
        )
        loss, _ = criterion(
            logits, batch["targets"], auxiliary, batch["targets"]["click"]
        )
        total_loss += loss.item()
        user_ids.extend(batch["user_ids"].detach().cpu().view(-1).tolist())
        for task in TASK_NAMES:
            targets[task].extend(
                batch["targets"][task].detach().cpu().view(-1).tolist()
            )
            logits_by_task[task].extend(logits[task].detach().cpu().view(-1).tolist())

    metrics = compute_multitask_metrics(
        targets=targets,
        logits=logits_by_task,
        user_ids=user_ids,
        task_names=TASK_NAMES,
    )
    return {"val_loss": total_loss / max(len(loader), 1), **metrics}


def move_batch(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = {
                sub_key: sub_value.to(device) for sub_key, sub_value in value.items()
            }
        else:
            moved[key] = value.to(device)
    return moved
