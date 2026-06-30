import json
import uuid
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from recommender.data_process.recall import TwoTowerDataset, collate_fn
from recommender.models.two_tower import build_two_tower_model


class TwoTowerTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate: float,
        weight_decay: float,
        loss_type: str,
        infonce_temperature: float,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.loss_type = loss_type.lower()
        if self.loss_type not in {"bce", "infonce"}:
            raise ValueError("loss_type must be 'bce' or 'infonce'")
        self.use_infonce = self.loss_type == "infonce"
        self.infonce_temperature = infonce_temperature
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "val_precision": [],
            "val_recall": [],
        }

    def _compute_infonce_loss(self, user_repr, item_repr, labels):
        positive_mask = labels > 0.5
        if positive_mask.sum().item() < 2:
            return None
        positive_users = user_repr[positive_mask]
        positive_items = item_repr[positive_mask]
        logits = (
            torch.matmul(positive_users, positive_items.t()) / self.infonce_temperature
        )
        targets = torch.arange(positive_users.size(0), device=logits.device)
        return 0.5 * (
            nn.functional.cross_entropy(logits, targets)
            + nn.functional.cross_entropy(logits.t(), targets)
        )

    def _forward_batch(self, batch):
        (
            user_ids,
            item_ids,
            behavior_sequences,
            behavior_masks,
            gender_ids,
            age_ids,
            video_category_ids,
            labels,
        ) = [value.to(self.device) for value in batch]
        scores, embeddings = self.model(
            user_ids,
            item_ids,
            behavior_sequences,
            behavior_masks,
            gender_ids=gender_ids,
            age_ids=age_ids,
            video_category_ids=video_category_ids,
            return_embeddings=self.use_infonce,
        )
        return scores, embeddings, labels

    def _compute_loss(self, scores, labels, embeddings=None):
        if not self.use_infonce:
            return self.criterion(scores, labels)
        if embeddings is None:
            raise ValueError("InfoNCE training requires model embeddings")
        return self._compute_infonce_loss(
            embeddings["user_repr"], embeddings["item_repr"], labels
        )

    def train_epoch(self, epoch: int, epochs: int) -> float:
        self.model.train()
        total_loss = 0.0
        batches = 0
        progress = tqdm(
            self.train_loader,
            desc=f"recall train {epoch}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in progress:
            scores, embeddings, labels = self._forward_batch(batch)
            loss = self._compute_loss(scores, labels, embeddings)
            if loss is None:
                continue
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            total_loss += loss.item()
            batches += 1
            progress.set_postfix(loss=f"{total_loss / batches:.6f}")
        return total_loss / max(batches, 1)

    @torch.no_grad()
    def validate(self, epoch: int, epochs: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        batches = 0
        scores_all = []
        labels_all = []
        progress = tqdm(
            self.val_loader,
            desc=f"recall validate {epoch}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in progress:
            scores, embeddings, labels = self._forward_batch(batch)
            loss = self._compute_loss(scores, labels, embeddings)
            if loss is not None:
                total_loss += loss.item()
                batches += 1
                progress.set_postfix(loss=f"{total_loss / batches:.6f}")
            scores_all.extend(torch.sigmoid(scores).cpu().view(-1).tolist())
            labels_all.extend(labels.cpu().view(-1).tolist())

        labels_array = np.asarray(labels_all)
        scores_array = np.asarray(scores_all)
        predictions = (scores_array >= 0.5).astype(np.int64)
        auc = (
            float(roc_auc_score(labels_array, scores_array))
            if np.unique(labels_array).size >= 2
            else float("nan")
        )
        return {
            "val_loss": total_loss / max(batches, 1),
            "val_auc": auc,
            "val_precision": float(
                precision_score(labels_array, predictions, zero_division=0)
            ),
            "val_recall": float(
                recall_score(labels_array, predictions, zero_division=0)
            ),
        }

    def train(
        self,
        epochs: int,
        checkpoint_path: Path,
        checkpoint_metadata: Dict,
    ) -> Dict[str, list]:
        best_loss = float("inf")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch, epochs)
            metrics = self.validate(epoch, epochs)
            self.scheduler.step(metrics["val_loss"])
            self.history["train_loss"].append(train_loss)
            for key in ("val_loss", "val_auc", "val_precision", "val_recall"):
                self.history[key].append(metrics[key])
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} "
                f"val_loss={metrics['val_loss']:.6f} "
                f"val_auc={metrics['val_auc']:.6f} "
                f"precision={metrics['val_precision']:.6f} "
                f"recall={metrics['val_recall']:.6f}"
            )
            if metrics["val_loss"] < best_loss:
                best_loss = metrics["val_loss"]
                torch.save(
                    {
                        **checkpoint_metadata,
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        **metrics,
                    },
                    checkpoint_path,
                )
        return self.history


def _model_config(dataset: TwoTowerDataset, args) -> Dict:
    return {
        "user_vocab_size": dataset.user_vocab_size,
        "item_vocab_size": dataset.item_vocab_size,
        "gender_vocab_size": dataset.gender_vocab_size,
        "age_vocab_size": dataset.age_vocab_size,
        "video_category_vocab_size": dataset.video_category_vocab_size,
        "embedding_dim": args.embedding_dim,
        "transformer_heads": args.transformer_heads,
        "transformer_layers": args.transformer_layers,
        "user_tower_dims": list(args.user_tower_dims),
        "item_tower_dims": list(args.item_tower_dims),
        "output_dim": args.output_dim,
        "dropout": args.dropout,
        "max_seq_len": args.max_seq_len,
        "temperature": args.temperature,
    }


def run_recall_training(args) -> Dict[str, Path]:
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = TwoTowerDataset(
        str(args.data_path),
        max_seq_len=args.max_seq_len,
        sample_rows=args.sample_rows,
    )
    if len(dataset) < 2:
        raise ValueError("at least two rows are required for a train/validation split")
    train_size = int(len(dataset) * (1.0 - args.val_ratio))
    train_size = min(max(train_size, 1), len(dataset) - 1)
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, len(dataset) - train_size], generator=generator
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model_config = _model_config(dataset, args)
    model = build_two_tower_model(**model_config)
    device = torch.device(args.device)
    trainer = TwoTowerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type,
        infonce_temperature=args.infonce_temperature,
    )
    experiment_name = f"two_tower_contiguous_{args.loss_type}"
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / f"{experiment_name}_best.pt"
    history_path = output_dir / f"{experiment_name}_history.json"
    metadata = {
        "checkpoint_version": 2,
        "checkpoint_id": uuid.uuid4().hex,
        "model_config": model_config,
        "data_config": {
            "mapping_strategy": "contiguous",
            "max_seq_len": args.max_seq_len,
            "sample_rows": args.sample_rows,
        },
        "feature_mappings": dataset.export_feature_mappings(),
        "vocab_sizes": {
            key: model_config[key]
            for key in (
                "user_vocab_size",
                "item_vocab_size",
                "gender_vocab_size",
                "age_vocab_size",
                "video_category_vocab_size",
            )
        },
    }
    history = trainer.train(args.epochs, checkpoint_path, metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
    print(f"checkpoint={checkpoint_path}")
    print(f"history={history_path}")
    return {"checkpoint": checkpoint_path, "history": history_path}
