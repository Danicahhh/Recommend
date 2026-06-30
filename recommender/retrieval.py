import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from recommender.data_process.recall import (
    TwoTowerDataset,
    UserSample,
    build_user_samples,
)
from recommender.models.two_tower import build_two_tower_model

try:
    import faiss
except ImportError:
    faiss = None


class RecallEngine:
    def __init__(
        self,
        model,
        device,
        dataset,
        use_faiss: bool,
        ann_nlist: int,
        ann_nprobe: int,
        faiss_index_path: Optional[Path] = None,
        rebuild_faiss: bool = False,
        index_namespace: str = "",
    ):
        self.model = model.to(device)
        self.device = device
        self.dataset = dataset
        self.use_faiss = use_faiss and faiss is not None
        self.ann_nlist = ann_nlist
        self.ann_nprobe = ann_nprobe
        self.faiss_index_path = faiss_index_path
        self.faiss_index = None
        self.candidate_item_ids = np.asarray(
            dataset.candidate_item_ids, dtype=np.int64
        )
        fingerprint = hashlib.sha256(self.candidate_item_ids.tobytes())
        candidate_raw_ids = np.asarray(
            [
                dataset.raw_item_id(int(model_id))
                for model_id in self.candidate_item_ids
            ],
            dtype=np.int64,
        )
        fingerprint.update(candidate_raw_ids.tobytes())
        fingerprint.update(index_namespace.encode("utf-8"))
        self.index_fingerprint = fingerprint.hexdigest()
        if use_faiss and faiss is None:
            print("FAISS is unavailable; falling back to exact retrieval")

        loaded = False
        if self.use_faiss and faiss_index_path and not rebuild_faiss:
            loaded = self._load_index()
        self.item_embeddings = None
        if not loaded:
            self.item_embeddings = self._compute_item_embeddings()
            if self.use_faiss:
                self._build_index()
                self._save_index()

    @torch.no_grad()
    def _compute_item_embeddings(self):
        self.model.eval()
        embeddings = []
        item_ids = torch.tensor(
            self.candidate_item_ids, device=self.device, dtype=torch.long
        )
        for start in tqdm(
            range(0, len(item_ids), 1024),
            desc="test item embeddings",
            unit="batch",
            dynamic_ncols=True,
        ):
            batch_ids = item_ids[start : start + 1024]
            categories = torch.tensor(
                self.dataset.item_category_by_id[batch_ids.cpu().numpy()],
                device=self.device,
                dtype=torch.long,
            )
            embeddings.append(
                self.model.get_item_embedding(batch_ids, video_category_ids=categories)
            )
        return torch.cat(embeddings, dim=0)

    def _build_index(self):
        values = self.item_embeddings.detach().cpu().numpy().astype(np.float32)
        faiss.normalize_L2(values)
        dimension = values.shape[1]
        nlist = min(self.ann_nlist, max(1, values.shape[0] // 100))
        quantizer = faiss.IndexFlatIP(dimension)
        self.faiss_index = faiss.IndexIVFFlat(
            quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT
        )
        self.faiss_index.train(values)
        self.faiss_index.add(values)
        self.faiss_index.nprobe = min(self.ann_nprobe, nlist)

    def _load_index(self) -> bool:
        if not self.faiss_index_path or not self.faiss_index_path.exists():
            return False
        metadata_path = self.faiss_index_path.with_suffix(
            self.faiss_index_path.suffix + ".json"
        )
        if not metadata_path.exists():
            return False
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("candidate_fingerprint") != self.index_fingerprint:
            return False
        index = faiss.read_index(str(self.faiss_index_path))
        if index.ntotal != len(self.candidate_item_ids):
            return False
        index.nprobe = min(self.ann_nprobe, index.nlist)
        self.faiss_index = index
        return True

    def _save_index(self):
        if self.faiss_index_path is None:
            return
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(self.faiss_index_path))
        metadata_path = self.faiss_index_path.with_suffix(
            self.faiss_index_path.suffix + ".json"
        )
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(
                {"candidate_fingerprint": self.index_fingerprint},
                file,
                ensure_ascii=False,
                indent=2,
            )

    @torch.no_grad()
    def recall(self, user_samples: Sequence[UserSample], top_k: int):
        self.model.eval()
        results = {}
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        limit = min(top_k, len(self.candidate_item_ids))
        for user_id, sequence, mask, gender_id, age_id in tqdm(
            user_samples,
            desc="test recall",
            unit="user",
            dynamic_ncols=True,
        ):
            user_embedding = self.model.get_user_embedding(
                torch.tensor([user_id], device=self.device),
                torch.tensor(np.asarray([sequence]), device=self.device),
                torch.tensor(np.asarray([mask]), device=self.device),
                gender_ids=torch.tensor([gender_id], device=self.device),
                age_ids=torch.tensor([age_id], device=self.device),
            )
            if self.faiss_index is not None:
                query = user_embedding.cpu().numpy().astype(np.float32)
                faiss.normalize_L2(query)
                scores, item_ids = self.faiss_index.search(query, limit)
                results[user_id] = [
                    (
                        self.dataset.raw_item_id(
                            self.candidate_item_ids[int(candidate_index)]
                        ),
                        float(score),
                    )
                    for candidate_index, score in zip(item_ids[0], scores[0])
                    if candidate_index >= 0
                ]
            else:
                scores = self.model.predict_batch(user_embedding, self.item_embeddings)
                top_scores, top_indices = torch.topk(scores, k=limit)
                results[user_id] = [
                    (
                        self.dataset.raw_item_id(
                            self.candidate_item_ids[int(candidate_index)]
                        ),
                        float(score),
                    )
                    for candidate_index, score in zip(
                        top_indices.cpu().numpy(), top_scores.cpu().numpy()
                    )
                ]
        return results


def _validate_vocab(dataset, expected):
    actual = {
        "user_vocab_size": dataset.user_vocab_size,
        "item_vocab_size": dataset.item_vocab_size,
        "gender_vocab_size": dataset.gender_vocab_size,
        "age_vocab_size": dataset.age_vocab_size,
        "video_category_vocab_size": dataset.video_category_vocab_size,
    }
    if actual != expected:
        raise ValueError(
            "checkpoint and dataset vocabularies do not match: "
            f"expected={expected}, actual={actual}"
        )


def _validate_mapping_strategy(data_config):
    strategy = data_config.get(
        "mapping_strategy",
        data_config.get("item_mapping_mode", "contiguous"),
    )
    if strategy != "contiguous":
        raise ValueError(
            "raw item mapping checkpoints are no longer supported; "
            "retrain the recall model with contiguous mapping"
        )


def run_recall_generation(args) -> Path:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    data_config = checkpoint["data_config"]
    _validate_mapping_strategy(data_config)
    sample_rows = (
        args.sample_rows
        if args.sample_rows is not None
        else data_config.get("sample_rows")
    )
    dataset = TwoTowerDataset(
        str(args.data_path),
        max_seq_len=data_config["max_seq_len"],
        sample_rows=sample_rows,
        feature_mappings=checkpoint.get("feature_mappings"),
        legacy_encoding="feature_mappings" not in checkpoint,
    )
    _validate_vocab(dataset, checkpoint["vocab_sizes"])
    model = build_two_tower_model(**checkpoint["model_config"])
    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=checkpoint.get("checkpoint_version", 1) >= 2,
    )
    if checkpoint.get("checkpoint_version", 1) < 2 and incompatible.unexpected_keys:
        print(
            "Loaded a legacy checkpoint; obsolete BatchNorm state was ignored: "
            f"{incompatible.unexpected_keys}"
        )
    output_path = Path(args.output_path)
    index_path = output_path.with_suffix(".faiss.index")
    checkpoint_stat = Path(args.checkpoint).stat()
    index_namespace = checkpoint.get(
        "checkpoint_id",
        (
            f"{Path(args.checkpoint).resolve()}:"
            f"{checkpoint_stat.st_size}:{checkpoint_stat.st_mtime_ns}"
        ),
    )
    engine = RecallEngine(
        model=model,
        device=device,
        dataset=dataset,
        use_faiss=args.use_faiss,
        ann_nlist=args.ann_nlist,
        ann_nprobe=args.ann_nprobe,
        faiss_index_path=index_path,
        rebuild_faiss=args.rebuild_faiss,
        index_namespace=index_namespace,
    )
    unique_users = np.unique(dataset.user_ids)[: args.num_users]
    user_samples = build_user_samples(
        dataset, user_ids=unique_users, max_seq_len=data_config["max_seq_len"]
    )
    recalls = engine.recall(user_samples, args.top_k)
    payload = {
        str(user_id): {
            "recalled_items": [item_id for item_id, _ in items],
            "scores": [score for _, score in items],
        }
        for user_id, items in recalls.items()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(f"recall_results={output_path}")
    return output_path
