from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: Sequence[int],
        dropout: float = 0.0,
        use_output_activation: bool = True,
    ):
        super().__init__()
        if len(hidden_units) == 0:
            raise ValueError("hidden_units must not be empty")

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for i, unit in enumerate(hidden_units):
            layers.append(nn.Linear(prev_dim, unit))
            if i < len(hidden_units) - 1 or use_output_activation:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            prev_dim = unit

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MMOE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_name_list: Sequence[str],
        expert_nums: int = 4,
        expert_dnn_units: Sequence[int] = (128, 64),
        gate_dnn_units: Sequence[int] = (128, 64),
        task_tower_dnn_units: Sequence[int] = (128, 64),
        dropout: float = 0.0,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if expert_nums <= 0:
            raise ValueError("expert_nums must be positive")
        if len(task_name_list) == 0:
            raise ValueError("task_name_list must not be empty")

        self.task_name_list = list(task_name_list)
        self.num_tasks = len(task_name_list)

        self.experts = nn.ModuleList(
            [DNN(input_dim, expert_dnn_units, dropout=dropout) for _ in range(expert_nums)]
        )
        expert_out_dim = expert_dnn_units[-1]

        self.gate_mlps = nn.ModuleList(
            [DNN(input_dim, gate_dnn_units, dropout=dropout) for _ in range(self.num_tasks)]
        )
        self.gate_logits = nn.ModuleList(
            [nn.Linear(gate_dnn_units[-1], expert_nums, bias=False) for _ in range(self.num_tasks)]
        )

        self.task_towers = nn.ModuleList(
            [
                DNN(
                    expert_out_dim,
                    list(task_tower_dnn_units) + [1],
                    dropout=dropout,
                    use_output_activation=False,
                )
                for _ in range(self.num_tasks)
            ]
        )

    def forward(self, x: torch.Tensor, return_prob: bool = False):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        task_outputs = []
        gate_weights_list = []
        for task_idx in range(self.num_tasks):
            gate_hidden = self.gate_mlps[task_idx](x)
            gate_weight = F.softmax(self.gate_logits[task_idx](gate_hidden), dim=-1)
            gate_weights_list.append(gate_weight)

            task_feature = torch.einsum("be,bed->bd", gate_weight, expert_outputs)
            task_logit = self.task_towers[task_idx](task_feature)
            task_outputs.append(torch.sigmoid(task_logit) if return_prob else task_logit)

        return task_outputs, gate_weights_list


def build_mmoe_model(
    feature_columns: int,
    task_name_list: Sequence[str],
    expert_nums: int = 4,
    expert_dnn_units: Sequence[int] = (128, 64),
    gate_dnn_units: Sequence[int] = (128, 64),
    task_tower_dnn_units: Sequence[int] = (128, 64),
    dropout: float = 0.0,
):
    return MMOE(
        input_dim=int(feature_columns),
        task_name_list=task_name_list,
        expert_nums=expert_nums,
        expert_dnn_units=expert_dnn_units,
        gate_dnn_units=gate_dnn_units,
        task_tower_dnn_units=task_tower_dnn_units,
        dropout=dropout,
    )
