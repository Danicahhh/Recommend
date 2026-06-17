from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: Sequence[int],
        dropout: float = 0.0,
        output_activation: bool = True,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not hidden_units:
            raise ValueError("hidden_units must not be empty")

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for idx, hidden_dim in enumerate(hidden_units):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if idx < len(hidden_units) - 1 or output_activation:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TargetAttention(nn.Module):
    """Use the target item as the only query over behavior sequence."""

    def __init__(self, embedding_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim**0.5

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        target_embedding: torch.Tensor,
        behavior_embeddings: torch.Tensor,
        behavior_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = behavior_embeddings.shape

        query = self.q_proj(target_embedding).view(
            batch_size, 1, self.num_heads, self.head_dim
        )
        key = self.k_proj(behavior_embeddings).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        value = self.v_proj(behavior_embeddings).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        all_masked = None
        if behavior_mask is not None:
            mask = behavior_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
            all_masked = behavior_mask.all(dim=1, keepdim=True).unsqueeze(1).unsqueeze(2)
            scores = torch.where(all_masked, torch.zeros_like(scores), scores)

        weights = F.softmax(scores, dim=-1)
        if all_masked is not None:
            weights = torch.where(all_masked, torch.zeros_like(weights), weights)
        weights = self.dropout(weights)

        output = torch.matmul(weights, value)
        output = output.transpose(1, 2).reshape(batch_size, self.embedding_dim)
        return self.out_proj(output)


class AttributeExpert(nn.Module):
    """Expert with a learnable feature mask, so experts specialize by attributes."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: Sequence[int],
        dropout: float = 0.0,
        use_attribute_mask: bool = True,
    ):
        super().__init__()
        self.use_attribute_mask = use_attribute_mask
        if use_attribute_mask:
            self.attribute_mask = nn.Parameter(torch.zeros(input_dim))
        else:
            self.register_parameter("attribute_mask", None)
        self.mlp = MLP(input_dim, hidden_units, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_x = x
        if self.use_attribute_mask:
            masked_x = x * torch.sigmoid(self.attribute_mask)
        return self.mlp(masked_x)


class PersonalizedMMOE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_names: Sequence[str],
        num_experts: int = 4,
        expert_units: Sequence[int] = (256, 128),
        gate_units: Sequence[int] = (128, 64),
        tower_units: Sequence[int] = (128, 64),
        dropout: float = 0.0,
        personalized: bool = True,
        use_attribute_expert_mask: bool = True,
        use_personalized_gate: Optional[bool] = None,
        use_task_bias: Optional[bool] = None,
    ):
        super().__init__()
        if not task_names:
            raise ValueError("task_names must not be empty")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")

        self.task_names = list(task_names)
        self.use_personalized_gate = personalized if use_personalized_gate is None else use_personalized_gate
        self.use_task_bias = personalized if use_task_bias is None else use_task_bias
        self.num_tasks = len(task_names)
        self.num_experts = num_experts

        self.experts = nn.ModuleList(
            [
                AttributeExpert(
                    input_dim,
                    expert_units,
                    dropout,
                    use_attribute_mask=use_attribute_expert_mask,
                )
                for _ in range(num_experts)
            ]
        )
        expert_out_dim = expert_units[-1]

        self.gate_mlps = nn.ModuleList(
            [MLP(input_dim, gate_units, dropout=dropout) for _ in range(self.num_tasks)]
        )
        self.gate_logits = nn.ModuleList(
            [nn.Linear(gate_units[-1], num_experts, bias=False) for _ in range(self.num_tasks)]
        )
        self.personalized_gate_bias = nn.ModuleList(
            [nn.Linear(input_dim, num_experts, bias=False) for _ in range(self.num_tasks)]
        )
        self.task_bias = nn.ModuleList(
            [
                MLP(input_dim, list(tower_units) + [1], dropout=dropout, output_activation=False)
                for _ in range(self.num_tasks)
            ]
        )
        self.task_towers = nn.ModuleList(
            [
                MLP(
                    expert_out_dim,
                    list(tower_units) + [1],
                    dropout=dropout,
                    output_activation=False,
                )
                for _ in range(self.num_tasks)
            ]
        )

    def forward(
        self, x: torch.Tensor, return_prob: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        logits: Dict[str, torch.Tensor] = {}
        gate_weights: Dict[str, torch.Tensor] = {}
        for task_idx, task_name in enumerate(self.task_names):
            gate_hidden = self.gate_mlps[task_idx](x)
            gate_logit = self.gate_logits[task_idx](gate_hidden)
            if self.use_personalized_gate:
                gate_logit = gate_logit + self.personalized_gate_bias[task_idx](x)

            gate_weight = F.softmax(gate_logit, dim=-1)
            task_feature = torch.einsum("be,bed->bd", gate_weight, expert_outputs)
            task_logit = self.task_towers[task_idx](task_feature)
            if self.use_task_bias:
                task_logit = task_logit + self.task_bias[task_idx](x)

            logits[task_name] = torch.sigmoid(task_logit) if return_prob else task_logit
            gate_weights[task_name] = gate_weight

        return logits, gate_weights


class RankMMOEModel(nn.Module):
    """
    Rank model with MMOE baseline, personalized MMOE, two-tower auxiliary output,
    and target attention over behavior sequence.
    """

    def __init__(
        self,
        user_vocab_size: int,
        item_vocab_size: int,
        task_names: Sequence[str] = ("click", "follow", "like", "share"),
        category_vocab_size: Optional[int] = None,
        gender_vocab_size: Optional[int] = None,
        age_vocab_size: Optional[int] = None,
        embedding_dim: int = 64,
        num_experts: int = 4,
        num_heads: int = 4,
        expert_units: Sequence[int] = (256, 128),
        gate_units: Sequence[int] = (128, 64),
        tower_units: Sequence[int] = (128, 64),
        dropout: float = 0.0,
        use_personalized_mmoe: bool = True,
        use_attribute_expert_mask: bool = True,
        use_personalized_gate: Optional[bool] = None,
        use_task_bias: Optional[bool] = None,
        use_target_attention: bool = True,
    ):
        super().__init__()
        self.task_names = list(task_names)
        self.embedding_dim = embedding_dim
        self.use_target_attention = use_target_attention

        self.user_embedding = nn.Embedding(user_vocab_size, embedding_dim)
        self.item_embedding = nn.Embedding(item_vocab_size, embedding_dim)
        self.category_embedding = (
            nn.Embedding(category_vocab_size, embedding_dim)
            if category_vocab_size is not None and category_vocab_size > 0
            else None
        )
        self.gender_embedding = (
            nn.Embedding(gender_vocab_size, embedding_dim)
            if gender_vocab_size is not None and gender_vocab_size > 0
            else None
        )
        self.age_embedding = (
            nn.Embedding(age_vocab_size, embedding_dim)
            if age_vocab_size is not None and age_vocab_size > 0
            else None
        )
        self.user_profile_projection = nn.Linear(embedding_dim * 3, embedding_dim)
        self.watch_projection = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        target_input_dim = embedding_dim * 3
        self.target_projection = nn.Linear(target_input_dim, embedding_dim)
        self.target_attention = TargetAttention(embedding_dim, num_heads, dropout)

        self.user_tower = MLP(embedding_dim * 2, tower_units, dropout=dropout)
        self.item_tower = MLP(embedding_dim, tower_units, dropout=dropout)
        tower_out_dim = tower_units[-1]

        rank_input_dim = tower_out_dim * 2 + embedding_dim * 2
        self.mmoe = PersonalizedMMOE(
            input_dim=rank_input_dim,
            task_names=task_names,
            num_experts=num_experts,
            expert_units=expert_units,
            gate_units=gate_units,
            tower_units=tower_units,
            dropout=dropout,
            personalized=use_personalized_mmoe,
            use_attribute_expert_mask=use_attribute_expert_mask,
            use_personalized_gate=use_personalized_gate,
            use_task_bias=use_task_bias,
        )

        self.user_aux_projection = nn.Linear(tower_out_dim, embedding_dim)
        self.item_aux_projection = nn.Linear(tower_out_dim, embedding_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def _zero_embedding_like(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(ids.size(0), self.embedding_dim, device=ids.device)

    def _make_target_embedding(
        self,
        item_ids: torch.Tensor,
        video_category_ids: Optional[torch.Tensor],
        watching_times: Optional[torch.Tensor],
    ) -> torch.Tensor:
        item_emb = self.item_embedding(item_ids)
        if self.category_embedding is not None and video_category_ids is not None:
            category_emb = self.category_embedding(video_category_ids)
        else:
            category_emb = self._zero_embedding_like(item_ids)

        if watching_times is None:
            watch_emb = self._zero_embedding_like(item_ids)
        else:
            watch_value = torch.log1p(watching_times.float()).view(-1, 1)
            watch_emb = self.watch_projection(watch_value)

        return self.target_projection(torch.cat([item_emb, category_emb, watch_emb], dim=1))

    def _make_user_embedding(
        self,
        user_ids: torch.Tensor,
        gender_ids: Optional[torch.Tensor],
        age_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        if self.gender_embedding is not None and gender_ids is not None:
            gender_emb = self.gender_embedding(gender_ids)
        else:
            gender_emb = self._zero_embedding_like(user_ids)

        if self.age_embedding is not None and age_ids is not None:
            age_emb = self.age_embedding(age_ids)
        else:
            age_emb = self._zero_embedding_like(user_ids)

        return self.user_profile_projection(torch.cat([user_emb, gender_emb, age_emb], dim=1))

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        behavior_sequence: Optional[torch.Tensor] = None,
        video_category_ids: Optional[torch.Tensor] = None,
        watching_times: Optional[torch.Tensor] = None,
        gender_ids: Optional[torch.Tensor] = None,
        age_ids: Optional[torch.Tensor] = None,
        behavior_mask: Optional[torch.Tensor] = None,
        return_prob: bool = False,
        return_auxiliary: bool = False,
    ):
        user_emb = self._make_user_embedding(user_ids, gender_ids, age_ids)
        target_emb = self._make_target_embedding(item_ids, video_category_ids, watching_times)

        if behavior_sequence is None:
            behavior_repr = torch.zeros_like(target_emb)
        else:
            behavior_emb = self.item_embedding(behavior_sequence)
            if self.use_target_attention:
                behavior_repr = self.target_attention(target_emb, behavior_emb, behavior_mask)
            elif behavior_mask is None:
                behavior_repr = behavior_emb.mean(dim=1)
            else:
                valid = (~behavior_mask).unsqueeze(-1).float()
                behavior_repr = (behavior_emb * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        user_vec = self.user_tower(torch.cat([user_emb, behavior_repr], dim=1))
        item_vec = self.item_tower(target_emb)
        rank_features = torch.cat([user_vec, item_vec, target_emb, behavior_repr], dim=1)
        logits, gate_weights = self.mmoe(rank_features, return_prob=return_prob)

        if not return_auxiliary:
            return logits

        user_aux = self.user_aux_projection(user_vec)
        item_aux = self.item_aux_projection(item_vec)
        auxiliary = {
            "user_embedding": user_aux,
            "item_embedding": item_aux,
            "two_tower_logit": (user_aux * item_aux).sum(dim=1, keepdim=True),
            "gate_weights": gate_weights,
        }
        return logits, auxiliary


class RankMultiTaskLoss(nn.Module):
    def __init__(
        self,
        task_names: Sequence[str] = ("click", "follow", "like", "share"),
        task_weights: Optional[Dict[str, float]] = None,
        pos_weights: Optional[Dict[str, float]] = None,
        auxiliary_weight: float = 0.1,
    ):
        super().__init__()
        self.task_names = list(task_names)
        self.task_weights = task_weights or {}
        self.auxiliary_weight = auxiliary_weight

        if pos_weights is None:
            self.pos_weights = None
        else:
            weights = [float(pos_weights.get(task, 1.0)) for task in self.task_names]
            self.register_buffer("pos_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        auxiliary_outputs: Optional[Dict[str, torch.Tensor]] = None,
        auxiliary_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses: Dict[str, torch.Tensor] = {}
        total = next(iter(predictions.values())).new_tensor(0.0)

        for idx, task in enumerate(self.task_names):
            pos_weight = None
            if self.pos_weights is not None:
                pos_weight = self.pos_weights[idx].view(1)
            task_loss = F.binary_cross_entropy_with_logits(
                predictions[task], targets[task].float(), pos_weight=pos_weight
            )
            losses[task] = task_loss
            total = total + self.task_weights.get(task, 1.0) * task_loss

        if (
            auxiliary_outputs is not None
            and auxiliary_target is not None
            and self.auxiliary_weight > 0
        ):
            aux_loss = F.binary_cross_entropy_with_logits(
                auxiliary_outputs["two_tower_logit"], auxiliary_target.float()
            )
            losses["two_tower_aux"] = aux_loss
            total = total + self.auxiliary_weight * aux_loss

        return total, losses


def build_rank_mmoe_model(config: Dict) -> RankMMOEModel:
    return RankMMOEModel(**config)
