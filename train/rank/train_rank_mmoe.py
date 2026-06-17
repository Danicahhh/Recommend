import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.rank_mmoe import RankMMOEModel, RankMultiTaskLoss


TASK_NAMES = ("click", "follow", "like", "share")
HIST_COLUMNS = tuple(f"hist_{idx}" for idx in range(1, 11))


class CTRDataset(Dataset):
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
        self.watching_times = torch.tensor(
            frame["watching_times"].fillna(0).astype("float32").values,
            dtype=torch.float32,
        )

        history = frame.loc[:, HIST_COLUMNS].copy()
        history = history.replace({"\\N": None, "": None}).apply(pd.to_numeric, errors="coerce")
        for column in HIST_COLUMNS:
            history[column] = history[column].map(item_map).fillna(0).astype("int64")
        self.behavior_sequence = torch.tensor(history.values, dtype=torch.long)
        self.behavior_mask = self.behavior_sequence.eq(0)

        self.targets = {
            task: torch.tensor(frame[task].astype("float32").values, dtype=torch.float32).view(-1, 1)
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
            "watching_times": self.watching_times[index],
            "behavior_sequence": self.behavior_sequence[index],
            "behavior_mask": self.behavior_mask[index],
            "targets": {task: target[index] for task, target in self.targets.items()},
        }


def make_index(values: Iterable, reserve_padding: bool) -> Dict[int, int]:
    offset = 1 if reserve_padding else 0
    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce")
    uniques = numeric_values.dropna().astype("int64").unique()
    return {int(value): idx + offset for idx, value in enumerate(uniques)}


def build_maps(
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


def train(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.data_path, nrows=args.sample_rows)
    user_map, item_map, category_map, gender_map, age_map = build_maps(frame)
    dataset = CTRDataset(frame, user_map, item_map, category_map, gender_map, age_map)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = torch.device(args.device)
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
    criterion = RankMultiTaskLoss(task_names=TASK_NAMES, auxiliary_weight=args.auxiliary_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad()
            logits, auxiliary = model(
                user_ids=batch["user_ids"],
                item_ids=batch["item_ids"],
                video_category_ids=batch["video_category_ids"],
                watching_times=batch["watching_times"],
                gender_ids=batch["gender_ids"],
                age_ids=batch["age_ids"],
                behavior_sequence=batch["behavior_sequence"],
                behavior_mask=batch["behavior_mask"],
                return_auxiliary=True,
            )
            loss, _ = criterion(logits, batch["targets"], auxiliary, batch["targets"]["click"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        print(f"epoch={epoch} loss={avg_loss:.6f}")


def move_batch(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = {sub_key: sub_value.to(device) for sub_key, sub_value in value.items()}
        else:
            moved[key] = value.to(device)
    return moved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "ctr_data_500k.csv")
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--auxiliary-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--baseline-mmoe", action="store_true")
    parser.add_argument("--mean-pooling", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
