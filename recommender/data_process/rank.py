from typing import Dict, Iterable, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


TASK_NAMES = ("click", "follow", "like", "share")
HIST_COLUMNS = tuple(f"hist_{idx}" for idx in range(1, 11))


class RankDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        user_map: Dict[int, int],
        item_map: Dict[int, int],
        category_map: Dict[int, int],
        gender_map: Dict[int, int],
        age_map: Dict[int, int],
    ):
        self.user_ids = torch.tensor(
            frame["user_id"].map(user_map).fillna(0).astype("int64").values,
            dtype=torch.long,
        )
        self.item_ids = torch.tensor(
            frame["item_id"].map(item_map).fillna(0).astype("int64").values,
            dtype=torch.long,
        )
        self.video_category_ids = torch.tensor(
            pd.to_numeric(frame["video_category"], errors="coerce")
            .map(category_map)
            .fillna(0)
            .astype("int64")
            .values,
            dtype=torch.long,
        )
        self.gender_ids = torch.tensor(
            pd.to_numeric(frame["gender"], errors="coerce")
            .map(gender_map)
            .fillna(0)
            .astype("int64")
            .values,
            dtype=torch.long,
        )
        self.age_ids = torch.tensor(
            pd.to_numeric(frame["age"], errors="coerce")
            .map(age_map)
            .fillna(0)
            .astype("int64")
            .values,
            dtype=torch.long,
        )
        history = frame.loc[:, HIST_COLUMNS].copy()
        history = history.replace({"\\N": None, "": None}).apply(
            pd.to_numeric, errors="coerce"
        )
        for column in HIST_COLUMNS:
            history[column] = history[column].map(item_map).fillna(0).astype("int64")
        self.behavior_sequence = torch.tensor(history.values, dtype=torch.long)
        self.behavior_mask = self.behavior_sequence.eq(0)
        self.targets = {
            task: torch.tensor(
                frame[task].astype("float32").values, dtype=torch.float32
            ).view(-1, 1)
            for task in TASK_NAMES
        }

    def __len__(self) -> int:
        return self.user_ids.size(0)

    def __getitem__(self, index: int) -> Dict:
        return {
            "user_ids": self.user_ids[index],
            "item_ids": self.item_ids[index],
            "video_category_ids": self.video_category_ids[index],
            "gender_ids": self.gender_ids[index],
            "age_ids": self.age_ids[index],
            "behavior_sequence": self.behavior_sequence[index],
            "behavior_mask": self.behavior_mask[index],
            "targets": {
                task: target[index] for task, target in self.targets.items()
            },
        }


def make_index(values: Iterable, reserve_padding: bool) -> Dict[int, int]:
    offset = 1 if reserve_padding else 0
    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce")
    uniques = numeric_values.dropna().astype("int64").unique()
    return {int(value): idx + offset for idx, value in enumerate(uniques)}


def build_feature_maps(
    frame: pd.DataFrame,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    user_map = make_index(frame["user_id"], reserve_padding=False)
    item_values = [frame["item_id"]]
    for column in HIST_COLUMNS:
        item_values.append(pd.to_numeric(frame[column], errors="coerce"))
    item_map = make_index(pd.concat(item_values, ignore_index=True), reserve_padding=True)
    category_map = make_index(frame["video_category"], reserve_padding=True)
    gender_map = make_index(frame["gender"], reserve_padding=True)
    age_map = make_index(frame["age"], reserve_padding=True)
    return user_map, item_map, category_map, gender_map, age_map
