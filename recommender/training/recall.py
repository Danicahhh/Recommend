import json
import math
import uuid
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from recommender.data_process.recall import TwoTowerDataset, collate_fn
from recommender.evaluation import compute_topk_retrieval_metrics
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
        evaluation_dataset: TwoTowerDataset,
        eval_ks: Sequence[int],
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
        self.evaluation_dataset = evaluation_dataset
        self.eval_ks = tuple(sorted({int(k) for k in eval_ks}))
        if not self.eval_ks or any(k <= 0 for k in self.eval_ks):
            raise ValueError("eval_ks must contain at least one positive integer")
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
            "val_evaluated_users": [],
        }
        for k in self.eval_ks:
            for metric_name in ("recall", "hit_rate", "ndcg"):
                self.history[f"val_{metric_name}@{k}"] = []

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

    def _forward_batch(self, batch, return_embeddings=None):
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
        if return_embeddings is None:
            return_embeddings = self.use_infonce
        scores, embeddings = self.model(
            user_ids,
            item_ids,
            behavior_sequences,
            behavior_masks,
            gender_ids=gender_ids,
            age_ids=age_ids,
            video_category_ids=video_category_ids,
            return_embeddings=return_embeddings,
        )
        return scores, embeddings, labels, user_ids, item_ids

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
            scores, embeddings, labels, _, _ = self._forward_batch(batch)
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
    def _candidate_embeddings(self) -> torch.Tensor:
        candidate_ids = self.evaluation_dataset.candidate_item_ids
        embeddings = []
        for start in range(0, len(candidate_ids), 1024):
            ids = torch.as_tensor(
                candidate_ids[start : start + 1024],
                device=self.device,
                dtype=torch.long,
            )
            categories = torch.as_tensor(
                self.evaluation_dataset.item_category_by_id[ids.cpu().numpy()],
                device=self.device,
                dtype=torch.long,
            )
            embeddings.append(
                self.model.get_item_embedding(
                    ids,
                    video_category_ids=categories,
                ).cpu()
            )
        return torch.cat(embeddings, dim=0)

    @torch.no_grad()
    def _retrieve_topk(
        self,
        user_representations: Mapping[int, torch.Tensor],
    ) -> Dict[int, list]:
        if not user_representations:
            return {}

        candidate_ids = np.asarray(
            self.evaluation_dataset.candidate_item_ids,
            dtype=np.int64,
        )
        candidate_embeddings = self._candidate_embeddings()
        limit = min(max(self.eval_ks), len(candidate_ids))
        user_ids = list(user_representations)
        recommendations: Dict[int, list] = {}

        for user_start in range(0, len(user_ids), 256):
            batch_user_ids = user_ids[user_start : user_start + 256]
            user_batch = torch.stack(
                [user_representations[user_id] for user_id in batch_user_ids]
            ).to(self.device)
            best_scores = user_batch.new_empty((len(batch_user_ids), 0))
            best_item_ids = torch.empty(
                (len(batch_user_ids), 0),
                device=self.device,
                dtype=torch.long,
            )

            for item_start in range(0, len(candidate_ids), 4096):
                item_batch = candidate_embeddings[
                    item_start : item_start + 4096
                ].to(self.device)
                scores = torch.matmul(user_batch, item_batch.t())
                item_ids = torch.as_tensor(
                    candidate_ids[item_start : item_start + len(item_batch)],
                    device=self.device,
                    dtype=torch.long,
                ).expand(len(batch_user_ids), -1)

                merged_scores = torch.cat((best_scores, scores), dim=1)
                merged_item_ids = torch.cat((best_item_ids, item_ids), dim=1)
                keep = min(limit, merged_scores.size(1))
                best_scores, indices = torch.topk(
                    merged_scores,
                    k=keep,
                    dim=1,
                )
                best_item_ids = torch.gather(merged_item_ids, 1, indices)

            for row, user_id in enumerate(batch_user_ids):
                recommendations[user_id] = best_item_ids[row].cpu().tolist()
        return recommendations

    @torch.no_grad()
    def validate(self, epoch: int, epochs: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        batches = 0
        user_representations: Dict[int, torch.Tensor] = {}
        relevant_items: Dict[int, set] = {}
        progress = tqdm(
            self.val_loader,
            desc=f"recall validate {epoch}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in progress:
            scores, embeddings, labels, user_ids, item_ids = self._forward_batch(
                batch,
                return_embeddings=True,
            )
            loss = self._compute_loss(scores, labels, embeddings)
            if loss is not None:
                total_loss += loss.item()
                batches += 1
                progress.set_postfix(loss=f"{total_loss / batches:.6f}")
            for user_id, item_id, label, user_repr in zip(
                user_ids.cpu().tolist(),
                item_ids.cpu().tolist(),
                labels.cpu().tolist(),
                embeddings["user_repr"].detach().cpu(),
            ):
                user_representations.setdefault(int(user_id), user_repr)
                if label > 0.5:
                    relevant_items.setdefault(int(user_id), set()).add(int(item_id))

        candidate_set = set(
            np.asarray(
                self.evaluation_dataset.candidate_item_ids,
                dtype=np.int64,
            ).tolist()
        )
        relevant_items = {
            user_id: items & candidate_set
            for user_id, items in relevant_items.items()
            if items & candidate_set
        }
        eligible_representations = {
            user_id: user_representations[user_id] for user_id in relevant_items
        }
        recommendations = self._retrieve_topk(eligible_representations)
        retrieval_metrics = compute_topk_retrieval_metrics(
            recommendations,
            relevant_items,
            self.eval_ks,
        )
        return {
            "val_loss": total_loss / max(batches, 1),
            **{
                f"val_{name}": value
                for name, value in retrieval_metrics.items()
            },
        }

    def train(
        self,
        epochs: int,
        checkpoint_path: Path,
        checkpoint_metadata: Dict,
    ) -> Dict[str, list]:
        monitor_key = f"val_ndcg@{max(self.eval_ks)}"
        best_metric = float("-inf")
        saved_checkpoint = False
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch, epochs)
            metrics = self.validate(epoch, epochs)
            self.scheduler.step(metrics["val_loss"])
            self.history["train_loss"].append(train_loss)
            for key in self.history:
                if key == "train_loss":
                    continue
                self.history[key].append(metrics[key])
            retrieval_summary = " ".join(
                f"recall@{k}={metrics[f'val_recall@{k}']:.6f} "
                f"hit_rate@{k}={metrics[f'val_hit_rate@{k}']:.6f} "
                f"ndcg@{k}={metrics[f'val_ndcg@{k}']:.6f}"
                for k in self.eval_ks
            )
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} "
                f"val_loss={metrics['val_loss']:.6f} "
                f"evaluated_users={int(metrics['val_evaluated_users'])} "
                f"{retrieval_summary}"
            )
            monitor_value = metrics[monitor_key]
            is_best = math.isfinite(monitor_value) and monitor_value > best_metric
            if is_best or not saved_checkpoint:
                if is_best:
                    best_metric = monitor_value
                torch.save(
                    {
                        **checkpoint_metadata,
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "monitor": monitor_key,
                        **metrics,
                    },
                    checkpoint_path,
                )
                saved_checkpoint = True
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
    if not args.eval_k or any(k <= 0 for k in args.eval_k):
        raise ValueError("--eval-k must contain at least one positive integer")
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
        evaluation_dataset=dataset,
        eval_ks=args.eval_k,
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
            "eval_k": list(trainer.eval_ks),
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
