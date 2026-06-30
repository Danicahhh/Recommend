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


def split_rank_frame(
    frame: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    if val_ratio <= 0.0 or test_ratio <= 0.0:
        raise ValueError("--val-ratio and --test-ratio must both be positive")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("--val-ratio + --test-ratio must be less than 1")
    if len(frame) < 3:
        raise ValueError(
            "at least three rows are required for train/validation/test splits"
        )

    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    held_out_size = round(len(shuffled) * (val_ratio + test_ratio))
    held_out_size = min(max(held_out_size, 2), len(shuffled) - 1)
    val_share = val_ratio / (val_ratio + test_ratio)
    val_size = round(held_out_size * val_share)
    val_size = min(max(val_size, 1), held_out_size - 1)
    train_size = len(shuffled) - held_out_size

    train_frame = shuffled.iloc[:train_size].reset_index(drop=True)
    val_frame = shuffled.iloc[train_size : train_size + val_size].reset_index(
        drop=True
    )
    test_frame = shuffled.iloc[train_size + val_size :].reset_index(drop=True)
    return train_frame, val_frame, test_frame


def run_rank_training(args) -> None:
    frame = pd.read_csv(args.data_path, nrows=args.sample_rows)
    train_frame, val_frame, test_frame = split_rank_frame(
        frame,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(
        f"split train={len(train_frame)} ({len(train_frame) / len(frame):.1%}) "
        f"validate={len(val_frame)} ({len(val_frame) / len(frame):.1%}) "
        f"test={len(test_frame)} ({len(test_frame) / len(frame):.1%})"
    )

    user_map, item_map, category_map, gender_map, age_map = build_feature_maps(frame)
    train_dataset = RankDataset(
        train_frame, user_map, item_map, category_map, gender_map, age_map
    )
    val_dataset = RankDataset(
        val_frame, user_map, item_map, category_map, gender_map, age_map
    )
    test_dataset = RankDataset(
        test_frame, user_map, item_map, category_map, gender_map, age_map
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
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    hidden_units = [args.hidden_dim] * args.num_layers
    model_config = {
        "user_vocab_size": len(user_map),
        "item_vocab_size": len(item_map) + 1,
        "category_vocab_size": len(category_map) + 1,
        "gender_vocab_size": len(gender_map) + 1,
        "age_vocab_size": len(age_map) + 1,
        "task_names": TASK_NAMES,
        "embedding_dim": args.embedding_dim,
        "num_experts": args.num_experts,
        "num_heads": args.num_heads,
        "expert_units": hidden_units,
        "gate_units": hidden_units,
        "tower_units": hidden_units,
        "dropout": args.dropout,
        "use_personalized_mmoe": not args.baseline_mmoe,
        "use_target_attention": not args.mean_pooling,
    }
    model = RankMMOEModel(
        **model_config
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "rank_mmoe_best.pt"
    results_path = output_dir / "rank_mmoe_results.json"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(
            train_loader,
            desc=f"rank train {epoch}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch_index, batch in enumerate(progress, start=1):
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
            progress.set_postfix(loss=f"{total_loss / batch_index:.6f}")

        avg_loss = total_loss / max(len(train_loader), 1)
        metrics = evaluate(
            model,
            criterion,
            val_loader,
            device,
            progress_desc=f"rank validate {epoch}/{args.epochs}",
        )
        is_best = best_metrics is None
        if is_best:
            best_epoch = epoch
            best_metrics = metrics
        if math.isfinite(metrics["mean_gauc"]) and metrics["mean_gauc"] > best_gauc:
            best_epoch = epoch
            best_metrics = metrics
            best_gauc = metrics["mean_gauc"]
            is_best = True
        if is_best:
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "validation_metrics": best_metrics,
                },
                checkpoint_path,
            )

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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(
        model,
        criterion,
        test_loader,
        device,
        progress_desc="rank test",
    )
    print(f"best_epoch={best_epoch} mean_gauc={best_metrics['mean_gauc']:.6f}")
    print(
        f"test_loss={test_metrics['val_loss']:.6f} "
        f"mean_auc={test_metrics['mean_auc']:.6f} "
        f"mean_gauc={test_metrics['mean_gauc']:.6f} "
        f"mean_logloss={test_metrics['mean_logloss']:.6f}"
    )
    for task in TASK_NAMES:
        print(
            f"  test {task}: AUC={test_metrics[f'{task}_auc']:.6f} "
            f"GAUC={test_metrics[f'{task}_gauc']:.6f} "
            f"LogLoss={test_metrics[f'{task}_logloss']:.6f}"
        )

    results = {
        "split_sizes": {
            "train": len(train_frame),
            "validation": len(val_frame),
            "test": len(test_frame),
        },
        "best_epoch": best_epoch,
        "validation_metrics": best_metrics,
        "test_metrics": test_metrics,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
    }
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(f"checkpoint={checkpoint_path}")
    print(f"results={results_path}")


@torch.no_grad()
def evaluate(
    model: RankMMOEModel,
    criterion: RankMultiTaskLoss,
    loader: DataLoader,
    device: torch.device,
    progress_desc: str = "rank validate",
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
        progress.set_postfix(loss=f"{total_loss / batch_index:.6f}")
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
