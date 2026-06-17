"""双塔召回的数据处理与数据集封装。"""

from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

UserSample = Tuple[int, np.ndarray, np.ndarray, int, int]


def build_padded_behavior_sequence(raw_sequence, max_seq_len: int):
    """把原始行为序列统一整理成定长序列和 mask。"""
    behavior_seq = np.asarray(raw_sequence, dtype=np.int64)
    valid_behavior_seq = behavior_seq[behavior_seq > 0][:max_seq_len]

    padded_seq = np.zeros(max_seq_len, dtype=np.int64)
    seq_len = len(valid_behavior_seq)
    padded_seq[:seq_len] = valid_behavior_seq

    mask = np.ones(max_seq_len, dtype=bool)
    mask[:seq_len] = False
    return padded_seq, mask


def build_first_value_mapping(keys, values, size: int):
    """按 key 取每个 ID 第一次出现的 value，生成稀疏映射数组。"""
    mapping = np.zeros(size, dtype=np.int64)
    seen = np.zeros(size, dtype=bool)
    for key, value in zip(keys, values):
        if not seen[key]:
            mapping[key] = value
            seen[key] = True
    return mapping


class TwoTowerDataset(Dataset):
    """双塔召回模型数据集。"""

    def __init__(self, data_path: str, max_seq_len: int = 10, item_mapping_mode: str = "contiguous"):
        self.max_seq_len = max_seq_len
        self.item_mapping_mode = item_mapping_mode.lower()
        if self.item_mapping_mode not in {"contiguous", "raw"}:
            raise ValueError("item_mapping_mode must be 'contiguous' or 'raw'")

        print(f"正在加载数据: {data_path}")
        df = pd.read_csv(data_path, na_values=["\\N", "NULL", "null", "None", ""])
        print(f"数据 shape: {df.shape}")

        self.hist_cols = [f"hist_{i}" for i in range(1, 11)]

        # ID 列清洗
        id_cols = ["user_id", "item_id"] + self.hist_cols
        for col in id_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64)
            df[col] = df[col].clip(lower=0)

        # 用户侧/物品侧离散特征编码
        df["gender"] = df["gender"].fillna("UNK").astype(str)
        df["age"] = df["age"].fillna("UNK").astype(str)
        df["video_category"] = df["video_category"].fillna("UNK").astype(str)

        df["gender_id"] = pd.Categorical(df["gender"]).codes.astype(np.int64)
        df["age_id"] = pd.Categorical(df["age"]).codes.astype(np.int64)
        df["video_category_id"] = pd.Categorical(df["video_category"]).codes.astype(np.int64)

        # 标签清洗
        df["click"] = pd.to_numeric(df["click"], errors="coerce").fillna(0).astype(np.float32)

        self.user_ids = df["user_id"].values

        all_item_ids = np.concatenate([
            df["item_id"].to_numpy(dtype=np.int64),
            df[self.hist_cols].to_numpy(dtype=np.int64).reshape(-1),
        ])
        unique_item_ids = np.unique(all_item_ids[all_item_ids > 0])

        if self.item_mapping_mode == "contiguous":
            # 将 item_id 和历史行为里的 item 编号统一映射成连续 ID，0 预留给 padding/未知值
            self.item_id_to_contiguous = {int(raw_id): int(idx + 1) for idx, raw_id in enumerate(unique_item_ids)}
            self.item_ids = np.array(
                [self.item_id_to_contiguous.get(int(raw_id), 0) for raw_id in df["item_id"].values],
                dtype=np.int64,
            )
            self.behavior_sequences = np.array(
                [
                    [self.item_id_to_contiguous.get(int(raw_id), 0) for raw_id in row]
                    for row in df[self.hist_cols].values
                ],
                dtype=np.int64,
            )
            self.item_vocab_size = int(unique_item_ids.size) + 1
            self.item_id_space_size = self.item_vocab_size
            self.item_id_offset = 0
        else:
            # 保留原始 item 编号，仅做正整数清洗；0 仍然作为 padding/unknown
            self.item_id_to_contiguous = None
            self.item_ids = df["item_id"].values.astype(np.int64)
            self.behavior_sequences = df[self.hist_cols].values.astype(np.int64)
            self.item_vocab_size = int(max(df["item_id"].max(), df[self.hist_cols].max().max())) + 1
            self.item_id_space_size = self.item_vocab_size
            self.item_id_offset = 0

        self.unique_item_ids = unique_item_ids
        self.gender_ids = df["gender_id"].values
        self.age_ids = df["age_id"].values
        self.video_category_ids = df["video_category_id"].values
        self.labels = df["click"].values

        self.user_vocab_size = int(df["user_id"].max()) + 1
        self.gender_vocab_size = int(df["gender_id"].max()) + 1
        self.age_vocab_size = int(df["age_id"].max()) + 1
        self.video_category_vocab_size = int(df["video_category_id"].max()) + 1

        # user_id -> 用户侧特征映射（召回时构造 user embedding 需要）
        self.user_gender_by_id = build_first_value_mapping(
            self.user_ids,
            self.gender_ids,
            self.user_vocab_size,
        )
        self.user_age_by_id = build_first_value_mapping(
            self.user_ids,
            self.age_ids,
            self.user_vocab_size,
        )

        # item_id -> 视频类目映射（批量预计算 item embedding 需要）
        self.item_category_by_id = build_first_value_mapping(
            self.item_ids,
            self.video_category_ids,
            self.item_id_space_size,
        )

        print(f"用户数: {df['user_id'].nunique()}")
        print(f"物品数(去重后): {len(unique_item_ids)}")
        print(f"item 映射模式: {self.item_mapping_mode}")
        print(f"样本数: {len(df)}")
        print(f"User vocab size: {self.user_vocab_size}")
        print(f"Item vocab size: {self.item_vocab_size}")
        print(f"Gender vocab size: {self.gender_vocab_size}")
        print(f"Age vocab size: {self.age_vocab_size}")
        print(f"Video-category vocab size: {self.video_category_vocab_size}")
        print(f"正样本比例: {self.labels.mean():.4f}")

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int):
        user_id = torch.tensor(self.user_ids[idx], dtype=torch.long)
        item_id = torch.tensor(self.item_ids[idx], dtype=torch.long)
        gender_id = torch.tensor(self.gender_ids[idx], dtype=torch.long)
        age_id = torch.tensor(self.age_ids[idx], dtype=torch.long)
        video_category_id = torch.tensor(self.video_category_ids[idx], dtype=torch.long)

        padded_seq, mask = build_padded_behavior_sequence(
            self.behavior_sequences[idx],
            self.max_seq_len,
        )

        behavior_sequence = torch.tensor(padded_seq, dtype=torch.long)
        behavior_mask = torch.tensor(mask, dtype=torch.bool)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return (
            user_id,
            item_id,
            behavior_sequence,
            behavior_mask,
            gender_id,
            age_id,
            video_category_id,
            label,
        )


def collate_fn(batch):
    user_ids = torch.stack([item[0] for item in batch])
    item_ids = torch.stack([item[1] for item in batch])
    behavior_sequences = torch.stack([item[2] for item in batch])
    behavior_masks = torch.stack([item[3] for item in batch])
    gender_ids = torch.stack([item[4] for item in batch])
    age_ids = torch.stack([item[5] for item in batch])
    video_category_ids = torch.stack([item[6] for item in batch])
    labels = torch.stack([item[7] for item in batch])
    return (
        user_ids,
        item_ids,
        behavior_sequences,
        behavior_masks,
        gender_ids,
        age_ids,
        video_category_ids,
        labels,
    )


def build_user_samples(
    dataset,
    user_ids: Optional[Iterable[int]] = None,
    max_seq_len: int = 10,
) -> List[UserSample]:
    """从数据集中构建用户召回输入。"""
    if user_ids is None:
        user_ids = np.unique(dataset.user_ids)

    user_samples: List[UserSample] = []
    for user_id in user_ids:
        user_indices = np.where(dataset.user_ids == user_id)[0]
        if len(user_indices) == 0:
            continue

        padded_seq, mask = build_padded_behavior_sequence(
            dataset.behavior_sequences[user_indices[0]],
            max_seq_len,
        )

        gender_id = int(dataset.user_gender_by_id[user_id])
        age_id = int(dataset.user_age_by_id[user_id])
        user_samples.append((int(user_id), padded_seq, mask, gender_id, age_id))

    return user_samples
