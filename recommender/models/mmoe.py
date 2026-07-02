"""MMoE 多任务排序模型。"""

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
        output_activation: bool = True,  # 后面会在任务塔中关闭：任务最终输出 logit 时，最后一层不能加 ReLU，否则输出无法表达负数，会影响二分类 logits。
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("输出维度必须为正数")
        if not hidden_units:
            raise ValueError("隐藏层单元列表不能为空")

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for idx, hidden_dim in enumerate(hidden_units):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if (
                idx < len(hidden_units) - 1 or output_activation
            ):  # 如果不是最后一层，或者最后一层需要激活函数，就添加 ReLU 激活函数。
                layers.append(nn.ReLU())
                if dropout > 0:  # 如果 dropout 大于 0，就添加 Dropout 层，防止过拟合。
                    layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.net = nn.Sequential(
            *layers
        )  # 用 nn.Sequential 将层列表组合成一个完整的网络。这样我们就可以直接调用 self.net(x) 来前向传播了。

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 输入：x = [batch_size, input_dim]
# 输出：[batch_size, hidden_units[-1]]


class TargetAttention(nn.Module):
    """
    当前候选物品作为 query/Q, 判断用户会不会点击/点赞/收藏/分享当前候选视频 target_item
    用当前候选物品作为 query/Q, 对用户历史行为序列做 attention, 得到“和当前物品相关的历史兴趣表示”"""

    def __init__(self, embedding_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("输入维度必须能被 多头注意力头数 整除")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = (
            self.head_dim**0.5
        )  # 注意力分数缩放因子，通常是 head_dim 的平方根。这样可以防止在计算 softmax 时分数过大导致梯度消失。
        # 定义四个线性映射：
        # q_proj：把目标物品 embedding 映射成 query
        # k_proj：把历史行为 embedding 映射成 key
        # v_proj：把历史行为 embedding 映射成 value
        # out_proj：把多头 attention 输出再映射一次，得到最终的行为表示。
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        target_embedding: torch.Tensor,  # 目标物品 embedding，形状为 [batch_size, embedding_dim]。这个是我们用来做 query 的。
        behavior_embeddings: torch.Tensor,  # 用户历史行为 embedding 序列，形状为 [batch_size, seq_len, embedding_dim]。这个是我们用来做 key 和 value 的。
        behavior_mask: Optional[
            torch.Tensor
        ] = None,  # 用户历史行为的 mask，形状为 [batch_size, seq_len]，其中 1 表示对应位置是 padding，不应该被 attention 关注。这个是可选的，如果没有提供，就表示所有历史行为都是有效的。
    ) -> torch.Tensor:
        # 优化点：用目标物品感知的注意力替代简单历史均值池化，
        # 让用户行为表示随当前待排序物品动态变化。
        batch_size, seq_len, _ = behavior_embeddings.shape
        # reshape 成多头形式
        # query 变化：[batch_size, embedding_dim] -> [batch_size, 1, num_heads, head_dim]
        query = self.q_proj(target_embedding).view(
            batch_size, 1, self.num_heads, self.head_dim
        )
        # key 变化：[batch_size, seq_len, embedding_dim] -> [batch_size, seq_len, num_heads, head_dim]
        key = self.k_proj(behavior_embeddings).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        # value 变化：[batch_size, seq_len, embedding_dim] -> [batch_size, seq_len, num_heads, head_dim]
        value = self.v_proj(behavior_embeddings).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        # 交换1，2维度，变成 [batch_size, num_heads, seq_len, head_dim] 方便后续计算注意力分数和加权求和。
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # [batch_size, num_heads, 1, head_dim] * [batch_size, num_heads, head_dim, seq_len] -> [batch_size, num_heads, 1, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        all_masked = None
        if behavior_mask is not None:  # behavior_mask：[batch_size, seq_len]
            mask = behavior_mask.unsqueeze(1).unsqueeze(
                2
            )  # [batch_size, seq_len]-> [batch_size, 1, 1, seq_len]

            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
            # torch.finfo(scores.dtype) # 返回 scores 的数据类型的浮点数信息(最小值/最大值/精度等)，.min 是该类型能表示的最小值(float32:-3.4e38)。我们用这个最小值来屏蔽 padding 的位置，这样在 softmax 时这些位置的权重会接近 0。
            # scores =
            # [
            #   [
            #     [[0.2, 0.8, -3.4e38]]
            #   ],
            #   [
            #     [[-3.4e38, -3.4e38, -3.4e38]]
            #   ]
            # ]
            all_masked = (
                behavior_mask.all(dim=1, keepdim=True).unsqueeze(1).unsqueeze(2)
            )
            # 对每个样本，看它整条历史序列是不是全部都是 True（表示这个位置是 padding，需要屏蔽）。
            # behavior_mask：[batch_size, seq_len]
            # behavior_mask.all(dim=1, keepdim=True) =
            # [
            #   [False], # 第 1 个样本：不是全 padding，所以是 False
            #   [True] # 第 2 个样本：全是 padding，所以是 True
            # ]
            # all_masked shape = [2, 1, 1, 1]
            # all_masked =
            # [
            #   [
            #     [[False]]
            #   ],
            #   [
            #     [[True]]
            #   ]
            # ]

            scores = torch.where(all_masked, torch.zeros_like(scores), scores)
            # torch.where(condition, A, B) 如果 condition 为 True，取 A，否则取 B
            # torch.zeros_like(scores) 生成一个和 scores 形状、类型、设备完全一样的全 0 张量。
            # scores：[batch_size, num_heads, 1, seq_len]
            # scores =
            # [
            #   [
            #     [[0.2, 0.8, -3.4e38]]
            #   ],
            #   [
            #     [[0.0, 0.0, 0.0]]
            #   ]
            # ]

        weights = F.softmax(scores, dim=-1)  # 在历史行为序列维度 seq_len 上做归一化
        # weights =
        # [
        #   [[[0.354, 0.646, 0.0]]],
        #   [[[0.333, 0.333, 0.333]]]
        # ]
        if (
            all_masked is not None
        ):  # 为了避免 softmax([0, 0, 0]) = [0.333, 0.333, 0.333] 这种情况
            weights = torch.where(all_masked, torch.zeros_like(weights), weights)
            # weights =
            # [
            #   [[[0.354, 0.646, 0.0]]],
            #   [[[0.0, 0.0, 0.0]]]
            # ]
        weights = self.dropout(weights)  # 随机丢弃一部分 attention 权重，减少模型过度依赖某几个历史行为
        # weights = [batch_size, num_heads, 1, seq_len]
        # value   = [batch_size, num_heads, seq_len, head_dim]
        # output = [batch_size, num_heads, 1, head_dim]
        output = torch.matmul(weights, value)  # 这一行用 attention 权重对 value 做加权求和
        output = output.transpose(1, 2).reshape(
            batch_size, self.embedding_dim
        )  # [batch_size, embedding_dim]
        return self.out_proj(output)  # 最后再通过一个线性层


"""每个 expert 在进入 MLP 前，先学习一组“特征选择权重”，
让不同 expert 更容易关注不同特征子空间，而不是所有 expert 都看完全一样的输入"""


class AttributeExpert(nn.Module):
    """Expert with a learnable feature mask, so experts specialize by attributes."""

    # 不同 expert 可能自动分工：
    # expert 1 更关注用户兴趣
    # expert 2 更关注物品属性
    # expert 3 更关注历史行为
    # expert 4 更关注用户-物品交互

    def __init__(
        self,
        input_dim: int,
        hidden_units: Sequence[int],  # expert 内部 MLP 的隐藏层结构 例如 (256, 128)
        dropout: float = 0.0,
        use_attribute_mask: bool = True,
    ):
        super().__init__()
        self.use_attribute_mask = use_attribute_mask
        if use_attribute_mask:
            # nn.Parameter(张量) 是 PyTorch 专门标记需要训练、要被优化器更新的权重；普通 torch.Tensor 只是普通数据，不会参与梯度更新。
            self.attribute_mask = nn.Parameter(torch.zeros(input_dim))
        else:
            self.register_parameter("attribute_mask", None)
        self.mlp = MLP(input_dim, hidden_units, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_x = x
        if self.use_attribute_mask:
            # 优化点：每个专家学习一组软特征掩码，鼓励不同专家关注
            # 不同属性子空间，避免所有专家使用完全相同的输入视角。
            masked_x = x * torch.sigmoid(
                self.attribute_mask
            )  # 因为attribute_mask初始值是 0,所以sigmoid(0)=0.5,所以初始时每个特征的权重都是 0.5,然后通过训练让它们学习到不同的特征重要性。
        return self.mlp(masked_x)


class PersonalizedMMOE(nn.Module):
    """
    在 baseline MMOE 基础上的扩展：
    1. 属性掩码专家增强专家分工；
    2. 个性化门控偏置让专家权重具备样本感知能力；
    3. 任务偏置为每个任务保留来自排序特征的直接修正路径。
    """

    def __init__(
        self,
        input_dim: int,
        task_names: Sequence[str],
        num_experts: int = 4,
        expert_units: Sequence[int] = (256, 128),
        gate_units: Sequence[int] = (128, 64),
        tower_units: Sequence[int] = (128, 64),
        dropout: float = 0.0,
        use_attribute_expert_mask: bool = True,  # expert 是否使用特征 mask
        use_personalized_gate: bool = True,  # 让不同样本有不同的 expert 路由倾向
        use_task_bias: bool = True,  # 给每个任务一条直接从输入特征到输出 logit 的修正路径。
    ):
        super().__init__()
        if not task_names:
            raise ValueError("任务塔不能为空")
        if num_experts <= 0:
            raise ValueError("专家数目必须为正数")

        self.task_names = list(task_names)
        self.use_personalized_gate = use_personalized_gate
        self.use_task_bias = use_task_bias
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
        # 门控网络
        self.gate_mlps = nn.ModuleList(
            [MLP(input_dim, gate_units, dropout=dropout) for _ in range(self.num_tasks)]
        )
        # 每个门控网络再接一个线性层，把 gate MLP 输出转成 expert 数量维度
        self.gate_logits = nn.ModuleList(
            [
                nn.Linear(gate_units[-1], num_experts, bias=False)
                for _ in range(self.num_tasks)
            ]
        )
        # 个性化门控：gate 不只是“任务级别”的选择，还会根据样本（当前用户、物品、行为特征）动态调整 expert 权重。
        self.personalized_gate_bias = nn.ModuleList(
            [
                nn.Linear(input_dim, num_experts, bias=False)
                for _ in range(self.num_tasks)
            ]
        )
        # 任务专属残差修正：每个任务都有一条直接从输入特征到输出 logit 的修正路径。
        self.task_bias = nn.ModuleList(
            [
                MLP(
                    input_dim,
                    list(tower_units) + [1],  # 最后一层输出 1 个值（表示该任务的预测结果）
                    dropout=dropout,
                    output_activation=False,
                )
                for _ in range(self.num_tasks)
            ]
        )
        # 任务塔网络
        self.task_towers = nn.ModuleList(
            [
                MLP(
                    expert_out_dim,
                    list(tower_units) + [1],  # 最后一层输出 1 个值（表示该任务的预测结果）
                    dropout=dropout,
                    output_activation=False,
                )
                for _ in range(self.num_tasks)
            ]
        )

    def forward(
        self, x: torch.Tensor, return_prob: bool = False  # 是否返回概率值
    ) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:  # 输出是两个字典：一个是每个任务的 logits，另一个是每个任务的 gate_weights
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        logits: Dict[str, torch.Tensor] = {}
        gate_weights: Dict[str, torch.Tensor] = {}
        # 每个任务单独计算 gate、expert 融合和 tower 输出。
        for task_idx, task_name in enumerate(self.task_names):
            gate_hidden = self.gate_mlps[task_idx](x)  # [batch_size, gate_units[-1]]
            gate_logit = self.gate_logits[task_idx](
                gate_hidden
            )  # [batch_size, num_experts]
            if self.use_personalized_gate:
                # 优化点：增加由输入特征直接生成的个性化门控偏置,input_dim-->num_experts
                # 根据不同样本的特征动态调整expert权重。
                # 任务级 gate_logit + 样本级 personalized_gate_bias
                gate_logit = gate_logit + self.personalized_gate_bias[task_idx](x)

            gate_weight = F.softmax(gate_logit, dim=-1)
            task_feature = torch.einsum("be,bed->bd", gate_weight, expert_outputs)
            task_logit = self.task_towers[task_idx](task_feature)
            if self.use_task_bias:
                # 优化点：任务专属残差路径保留原始排序信号，
                # 当expert混合对某个任务拟合不足时仍可提供直接修正。
                # 任务塔输出 + 任务专属残差路径输出
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
        user_vocab_size: int,  # 用户 ID 词表大小
        item_vocab_size: int,
        task_names: Sequence[str] = ("click", "follow", "like", "share"),
        category_vocab_size: Optional[int] = None,  # 视频类目词表大小，默认为 None
        gender_vocab_size: Optional[int] = None,  # 性别词表大小
        age_vocab_size: Optional[int] = None,
        embedding_dim: int = 64,
        num_experts: int = 4,
        num_heads: int = 4,
        expert_units: Sequence[int] = (256, 128),
        gate_units: Sequence[int] = (128, 64),
        tower_units: Sequence[int] = (128, 64),
        dropout: float = 0.0,
        use_attribute_expert_mask: bool = True,  # expert 是否使用特征 mask
        use_personalized_gate: bool = True,
        use_task_bias: bool = True,
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
        # 用户画像投影:拼接user_emb + gender_emb + age_emb，拼接后维度是 embedding_dim * 3，然后通过一个线性层投影回 embedding_dim
        self.user_profile_projection = nn.Linear(embedding_dim * 3, embedding_dim)

        # 物品属性投影：仅使用预测时可获得的 item 和 category 特征。
        # watching_times 是当前曝光后的行为结果，不能作为排序输入。
        self.target_projection = nn.Linear(embedding_dim * 2, embedding_dim)

        self.target_attention = TargetAttention(embedding_dim, num_heads, dropout)

        # 双塔网络：用户塔和物品塔，用户塔输入是用户画像和行为表示，物品塔输入是目标物品表示，输出都是 tower_units[-1] 维度的向量
        self.user_tower = MLP(embedding_dim * 2, tower_units, dropout=dropout)
        self.item_tower = MLP(embedding_dim, tower_units, dropout=dropout)
        tower_out_dim = tower_units[-1]
        # [user_vec, item_vec, target_emb, behavior_repr]
        rank_input_dim = tower_out_dim * 2 + embedding_dim * 2
        self.mmoe = PersonalizedMMOE(
            input_dim=rank_input_dim,
            task_names=task_names,
            num_experts=num_experts,
            expert_units=expert_units,
            gate_units=gate_units,
            tower_units=tower_units,
            dropout=dropout,
            use_attribute_expert_mask=use_attribute_expert_mask,
            use_personalized_gate=use_personalized_gate,
            use_task_bias=use_task_bias,
        )
        # 双塔辅助
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
    ) -> torch.Tensor:
        item_emb = self.item_embedding(item_ids)
        if self.category_embedding is not None and video_category_ids is not None:
            category_emb = self.category_embedding(video_category_ids)
        else:
            category_emb = self._zero_embedding_like(item_ids)

        return self.target_projection(torch.cat([item_emb, category_emb], dim=1))

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

        return self.user_profile_projection(
            torch.cat([user_emb, gender_emb, age_emb], dim=1)
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        behavior_sequence: Optional[torch.Tensor] = None,
        video_category_ids: Optional[torch.Tensor] = None,
        gender_ids: Optional[torch.Tensor] = None,
        age_ids: Optional[torch.Tensor] = None,
        behavior_mask: Optional[torch.Tensor] = None,
        return_prob: bool = False,
        return_auxiliary: bool = False,
    ):
        user_emb = self._make_user_embedding(user_ids, gender_ids, age_ids)
        target_emb = self._make_target_embedding(item_ids, video_category_ids)

        if behavior_sequence is None:
            behavior_repr = torch.zeros_like(target_emb)
        else:
            behavior_emb = self.item_embedding(behavior_sequence)
            if self.use_target_attention:
                behavior_repr = self.target_attention(
                    target_emb, behavior_emb, behavior_mask
                )
            elif behavior_mask is None:
                behavior_repr = behavior_emb.mean(dim=1)
            else:
                valid = (~behavior_mask).unsqueeze(-1).float()
                behavior_repr = (behavior_emb * valid).sum(dim=1) / valid.sum(
                    dim=1
                ).clamp_min(1.0)

        user_vec = self.user_tower(torch.cat([user_emb, behavior_repr], dim=1))
        item_vec = self.item_tower(target_emb)
        # 优化点：MMOE 不再只接收预先拼好的扁平特征，而是联合使用
        # 用户塔、物品塔、目标侧属性和行为上下文，为各任务提供更丰富的交互信号。
        rank_features = torch.cat(
            [user_vec, item_vec, target_emb, behavior_repr], dim=1
        )
        logits, gate_weights = self.mmoe(rank_features, return_prob=return_prob)

        if not return_auxiliary:
            return logits

        user_aux = self.user_aux_projection(user_vec)
        item_aux = self.item_aux_projection(item_vec)
        auxiliary = {
            "user_embedding": user_aux,
            "item_embedding": item_aux,
            # 优化点：双塔辅助目标在 MMOE 多任务损失之外，
            # 额外用召回式信号约束用户/物品表示。
            "two_tower_logit": (user_aux * item_aux).sum(dim=1, keepdim=True),
            "gate_weights": gate_weights,
        }
        return logits, auxiliary


class RankMultiTaskLoss(nn.Module):
    def __init__(
        self,
        task_names: Sequence[str] = ("click", "follow", "like", "share"),
        task_weights: Optional[Dict[str, float]] = None,  # 控制每个任务对总损失的贡献
        pos_weights: Optional[
            Dict[str, float]
        ] = None,  # 正样本权重：一个正样本对应几个负样本，解决正负样本不平衡，放大正样本损失
        auxiliary_weight: float = 0.1,  # 双塔辅助损失权重
        normalize_pos_weight: bool = False,
        task_weighting_method: str = "equal",
        gradnorm_alpha: float = 0.5,
        gradnorm_min_weight: float = 0.2,
        gradnorm_max_weight: float = 5.0,
    ):
        super().__init__()
        self.task_names = list(task_names)
        self.task_weights = task_weights or {}
        self.auxiliary_weight = auxiliary_weight
        self.normalize_pos_weight = normalize_pos_weight
        self.task_weighting_method = task_weighting_method
        self.gradnorm_alpha = gradnorm_alpha
        self.gradnorm_min_weight = gradnorm_min_weight
        self.gradnorm_max_weight = gradnorm_max_weight

        if task_weighting_method not in {"equal", "gradnorm", "uncertainty"}:
            raise ValueError(
                "task_weighting_method must be one of: "
                "equal, gradnorm, uncertainty"
            )
        if gradnorm_alpha < 0:
            raise ValueError("gradnorm_alpha must be non-negative")
        if not 0 < gradnorm_min_weight <= 1.0 <= gradnorm_max_weight:
            raise ValueError(
                "gradnorm weights require 0 < min_weight <= 1 <= max_weight"
            )

        if pos_weights is None:
            self.pos_weights = None
        else:
            weights = [float(pos_weights.get(task, 1.0)) for task in self.task_names]
            if any(weight <= 0 for weight in weights):
                raise ValueError("all pos_weights must be positive")
            self.register_buffer(
                "pos_weights", torch.tensor(weights, dtype=torch.float32)
            )

        if task_weighting_method == "gradnorm":
            self.gradnorm_weights = nn.Parameter(
                torch.ones(len(self.task_names), dtype=torch.float32)
            )
            self.register_buffer(
                "initial_task_losses",
                torch.full((len(self.task_names),), float("nan")),
            )
        else:
            self.register_parameter("gradnorm_weights", None)
            self.initial_task_losses = None

        if task_weighting_method == "uncertainty":
            self.log_task_variances = nn.Parameter(
                torch.zeros(len(self.task_names), dtype=torch.float32)
            )
        else:
            self.register_parameter("log_task_variances", None)

    def _task_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        pos_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        target = target.float()
        element_losses = F.binary_cross_entropy_with_logits(
            prediction,
            target,
            pos_weight=pos_weight,
            reduction="none", # 不使用样本平均，以防止损失过大。
        )
        if not self.normalize_pos_weight or pos_weight is None:
            return element_losses.mean()
        # 优化点：正样本权重归一化，避免正样本权重过大导致总损失过大，影响梯度更新。
        effective_weights = torch.where(
            target > 0.5,
            pos_weight.to(device=target.device, dtype=target.dtype),
            torch.ones_like(target),
        )
        return element_losses.sum() / effective_weights.sum().clamp_min(1.0)

    def current_task_weights(self) -> Dict[str, float]:
        if self.task_weighting_method == "gradnorm":
            values = self.gradnorm_weights.detach()
        elif self.task_weighting_method == "uncertainty":
            values = torch.exp(-self.log_task_variances.detach())
        else:
            values = torch.tensor(
                [self.task_weights.get(task, 1.0) for task in self.task_names],
                dtype=torch.float32,
            )
        return {
            task: float(value)
            for task, value in zip(self.task_names, values.cpu().tolist())
        }

    def gradnorm_objective(
        self,
        losses: Dict[str, torch.Tensor],
        shared_parameters: Sequence[torch.nn.Parameter],
    ) -> torch.Tensor:
        if self.task_weighting_method != "gradnorm":
            raise RuntimeError(
                "gradnorm_objective requires task_weighting_method=gradnorm"
            )

        task_losses = torch.stack([losses[task] for task in self.task_names])
        if torch.isnan(self.initial_task_losses).any():
            self.initial_task_losses.copy_(task_losses.detach().clamp_min(1e-8))

        shared_parameters = tuple(
            parameter for parameter in shared_parameters if parameter.requires_grad
        )
        if not shared_parameters:
            raise ValueError("GradNorm requires at least one shared parameter")

        base_gradient_norms = []
        for task_loss in task_losses:
            gradients = torch.autograd.grad(
                task_loss,
                shared_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            squared_norm = task_loss.new_tensor(0.0)
            for gradient in gradients:
                if gradient is not None:
                    squared_norm = squared_norm + gradient.detach().pow(2).sum()
            base_gradient_norms.append(squared_norm.sqrt().clamp_min(1e-12))

        base_gradient_norms = torch.stack(base_gradient_norms)
        gradient_norms = self.gradnorm_weights * base_gradient_norms
        relative_losses = (
            task_losses.detach() / self.initial_task_losses.clamp_min(1e-8)
        )
        inverse_training_rates = relative_losses / relative_losses.mean().clamp_min(
            1e-8
        )
        target_norms = (
            gradient_norms.detach().mean()
            * inverse_training_rates.pow(self.gradnorm_alpha)
        )
        return torch.abs(gradient_norms - target_norms).sum()

    @torch.no_grad()
    def normalize_gradnorm_weights(self) -> None:
        if self.task_weighting_method != "gradnorm":
            return
        self.gradnorm_weights.clamp_(
            min=self.gradnorm_min_weight,
            max=self.gradnorm_max_weight,
        )
        target_sum = float(len(self.task_names))
        difference = target_sum - float(self.gradnorm_weights.sum())
        if abs(difference) <= 1e-8:
            return

        if difference > 0:
            capacity = self.gradnorm_max_weight - self.gradnorm_weights
            self.gradnorm_weights.add_(
                difference * capacity / capacity.sum().clamp_min(1e-8)
            )
        else:
            capacity = self.gradnorm_weights - self.gradnorm_min_weight
            self.gradnorm_weights.sub_(
                -difference * capacity / capacity.sum().clamp_min(1e-8)
            )

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
            task_loss = self._task_loss(
                predictions[task],
                targets[task],
                pos_weight,
            )
            losses[task] = task_loss

            if self.task_weighting_method == "gradnorm":
                # GradNorm 单独更新任务权重；训练模型时不让主损失反向更新权重。
                total = total + self.gradnorm_weights[idx].detach() * task_loss
            elif self.task_weighting_method == "uncertainty":
                precision = torch.exp(-self.log_task_variances[idx])
                total = total + precision * task_loss + self.log_task_variances[idx]
            else:
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
            total = total + self.auxiliary_weight * aux_loss  # 再加上双塔辅助损失，得到最终总损失

        return total, losses


def build_rank_mmoe_model(config: Dict) -> RankMMOEModel:
    return RankMMOEModel(**config)
