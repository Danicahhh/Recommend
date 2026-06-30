"""
双塔召回模型 (Two-Tower Recall Model)

当前实现支持的特征：
- 用户侧：user_id、行为序列 hist_1-hist_10、gender、age
- 物品侧：item_id、video_category

模型结构：
- 用户塔：用户ID embedding + 行为序列 Transformer 编码 + DNN
- 物品塔：物品ID embedding + 类目 embedding + DNN
- 相似度：归一化后的内积（等价于余弦相似度）
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """位置编码模块，默认按 batch_first 输入使用。"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预生成位置编码矩阵，避免每次前向重复计算
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        if x.dim() == 3:
            seq_len = x.size(1)
            # 将 [seq_len, 1, d_model] 变为 [1, seq_len, d_model] 以便广播
            pe = self.pe[:seq_len].transpose(0, 1)  # [1, seq_len, d_model]
            x = x + pe
        else:
            # 保留旧格式兼容，当前项目调用路径不会进入这里
            x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerBehaviorEncoder(nn.Module):
    """使用 Transformer Encoder 编码用户历史行为序列。"""

    def __init__(
        self,
        item_vocab_size: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 100,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # 序列中的物品 ID embedding
        self.item_embedding = nn.Embedding(
            item_vocab_size, embedding_dim, padding_idx=0
        )

        # 位置编码
        self.pos_encoder = PositionalEncoding(embedding_dim, max_seq_len, dropout)

        # Transformer Encoder（batch_first=True）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        behavior_sequence: torch.Tensor,
        behavior_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            behavior_sequence: [batch, seq_len]，用户历史行为物品ID序列
            behavior_mask: [batch, seq_len]，True表示padding位置

        Returns:
            sequence_repr: [batch, embedding_dim]，序列表示向量
        """
        # 序列 embedding
        seq_emb = self.item_embedding(behavior_sequence)  # [batch, seq_len, emb_dim]

        # 位置编码
        seq_emb = self.pos_encoder(seq_emb)

        # True 表示 padding 位置，会被 Transformer 忽略
        all_padding = None
        if behavior_mask is not None:
            padding_mask = behavior_mask.bool()
            all_padding = padding_mask.all(dim=1)
            if all_padding.any():
                # PyTorch 的 nested-tensor 路径无法处理整个 batch 都为空的序列。
                # 临时开放一个位置完成计算，池化时仍使用原始 mask 并返回零表示。
                padding_mask = padding_mask.clone()
                padding_mask[all_padding, 0] = False
        else:
            padding_mask = None

        transformer_out = self.transformer_encoder(
            seq_emb, src_key_padding_mask=padding_mask
        )  # [batch, seq_len, emb_dim]

        # 对非 padding 位置做平均池化
        if behavior_mask is not None:
            valid_mask = (~behavior_mask).float().unsqueeze(-1)  # [batch, seq_len, 1]
            seq_sum = (transformer_out * valid_mask).sum(dim=1)  # [batch, emb_dim]
            valid_count = valid_mask.sum(dim=1).clamp(min=1)  # [batch, 1]
            sequence_repr = seq_sum / valid_count
        else:
            sequence_repr = transformer_out.mean(dim=1)  # [batch, emb_dim]

        sequence_repr = self.layer_norm(sequence_repr)
        if all_padding is not None and all_padding.any():
            sequence_repr = sequence_repr.masked_fill(all_padding.unsqueeze(1), 0.0)
        return sequence_repr


class DNN(nn.Module):
    """通用前馈网络模块。"""

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # LayerNorm 对 batch size 没有限制，最后一个 batch 只有一条样本时也稳定。
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class UserTower(nn.Module):
    """用户塔：融合用户 ID、行为序列以及可选的人口学特征。"""

    def __init__(
        self,
        user_vocab_size: int,
        item_vocab_size: int,
        gender_vocab_size: Optional[int] = None,
        age_vocab_size: Optional[int] = None,
        embedding_dim: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        tower_dims: list = [256, 128],
        output_dim: int = 64,
        dropout: float = 0.1,
        max_seq_len: int = 100,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.use_gender = gender_vocab_size is not None
        self.use_age = age_vocab_size is not None

        # 用户 ID embedding
        self.user_embedding = nn.Embedding(user_vocab_size, embedding_dim)

        # 可选用户侧离散特征 embedding
        self.gender_embedding = (
            nn.Embedding(gender_vocab_size, embedding_dim) if self.use_gender else None
        )
        self.age_embedding = (
            nn.Embedding(age_vocab_size, embedding_dim) if self.use_age else None
        )

        # 行为序列编码器
        self.behavior_encoder = TransformerBehaviorEncoder(
            item_vocab_size=item_vocab_size,
            embedding_dim=embedding_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            ff_dim=embedding_dim * 4,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # 融合用户表示与行为序列表示的 DNN
        fusion_input_dim = embedding_dim * 2
        if self.use_gender:
            fusion_input_dim += embedding_dim
        if self.use_age:
            fusion_input_dim += embedding_dim
        self.fusion_dnn = DNN(fusion_input_dim, tower_dims, dropout=dropout)

        # 输出投影层
        self.output_layer = nn.Linear(tower_dims[-1], output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        user_ids: torch.Tensor,
        behavior_sequence: torch.Tensor,
        behavior_mask: Optional[torch.Tensor] = None,
        gender_ids: Optional[torch.Tensor] = None,
        age_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            user_ids: [batch]，用户ID
            behavior_sequence: [batch, seq_len]，用户历史行为物品ID序列
            behavior_mask: [batch, seq_len]，True表示padding位置

        Returns:
            user_repr: [batch, output_dim]，用户表示向量
        """
        # 用户 ID embedding
        user_emb = self.user_embedding(user_ids)  # [batch, emb_dim]

        # 行为序列编码
        sequence_repr = self.behavior_encoder(
            behavior_sequence, behavior_mask
        )  # [batch, emb_dim]

        # 拼接用户侧特征
        fusion_parts = [user_emb, sequence_repr]

        if self.use_gender:
            if gender_ids is None:
                gender_ids = torch.zeros_like(user_ids)
            gender_emb = self.gender_embedding(gender_ids)
            fusion_parts.append(gender_emb)

        if self.use_age:
            if age_ids is None:
                age_ids = torch.zeros_like(user_ids)
            age_emb = self.age_embedding(age_ids)
            fusion_parts.append(age_emb)

        fusion_input = torch.cat(fusion_parts, dim=-1)

        # 前馈融合网络
        tower_out = self.fusion_dnn(fusion_input)  # [batch, tower_dims[-1]]

        # 输出投影
        user_repr = self.output_layer(tower_out)  # [batch, output_dim]
        user_repr = self.output_norm(user_repr)

        # 归一化后，内积即可近似余弦相似度
        user_repr = F.normalize(user_repr, p=2, dim=-1)

        return user_repr


class ItemTower(nn.Module):
    """物品塔：融合物品 ID 与可选类目特征。"""

    def __init__(
        self,
        item_vocab_size: int,
        video_category_vocab_size: Optional[int] = None,
        embedding_dim: int = 64,
        tower_dims: list = [256, 128],
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.use_video_category = video_category_vocab_size is not None

        # 物品 ID embedding
        self.item_embedding = nn.Embedding(item_vocab_size, embedding_dim)

        # 可选物品侧离散特征 embedding
        self.video_category_embedding = (
            nn.Embedding(video_category_vocab_size, embedding_dim)
            if self.use_video_category
            else None
        )

        # 前馈融合网络
        item_input_dim = embedding_dim * 2 if self.use_video_category else embedding_dim
        self.tower_dnn = DNN(item_input_dim, tower_dims, dropout=dropout)

        # 输出投影层
        self.output_layer = nn.Linear(tower_dims[-1], output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(
        self, item_ids: torch.Tensor, video_category_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            item_ids: [batch]，物品ID

        Returns:
            item_repr: [batch, output_dim]，物品表示向量
        """
        # 物品 ID embedding
        item_emb = self.item_embedding(item_ids)  # [batch, emb_dim]

        item_input = item_emb
        if self.use_video_category:
            if video_category_ids is None:
                video_category_ids = torch.zeros_like(item_ids)
            video_cat_emb = self.video_category_embedding(video_category_ids)
            item_input = torch.cat([item_emb, video_cat_emb], dim=-1)

        # 前馈融合网络
        tower_out = self.tower_dnn(item_input)  # [batch, tower_dims[-1]]

        # 输出投影
        item_repr = self.output_layer(tower_out)  # [batch, output_dim]
        item_repr = self.output_norm(item_repr)

        # 归一化后，内积即可近似余弦相似度
        item_repr = F.normalize(item_repr, p=2, dim=-1)

        return item_repr


class TwoTowerRecallModel(nn.Module):
    """双塔召回模型：用户塔 + 物品塔。"""

    def __init__(
        self,
        user_vocab_size: int,
        item_vocab_size: int,
        gender_vocab_size: Optional[int] = None,
        age_vocab_size: Optional[int] = None,
        video_category_vocab_size: Optional[int] = None,
        embedding_dim: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        user_tower_dims: list = [256, 128],
        item_tower_dims: list = [256, 128],
        output_dim: int = 64,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        temperature: float = 0.05,  # 相似度缩放系数
    ):
        super().__init__()
        self.output_dim = output_dim
        self.temperature = temperature

        # 用户塔
        self.user_tower = UserTower(
            user_vocab_size=user_vocab_size,
            item_vocab_size=item_vocab_size,
            gender_vocab_size=gender_vocab_size,
            age_vocab_size=age_vocab_size,
            embedding_dim=embedding_dim,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            tower_dims=user_tower_dims,
            output_dim=output_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # 物品塔
        self.item_tower = ItemTower(
            item_vocab_size=item_vocab_size,
            video_category_vocab_size=video_category_vocab_size,
            embedding_dim=embedding_dim,
            tower_dims=item_tower_dims,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        behavior_sequence: torch.Tensor,
        behavior_mask: Optional[torch.Tensor] = None,
        gender_ids: Optional[torch.Tensor] = None,
        age_ids: Optional[torch.Tensor] = None,
        video_category_ids: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Args:
            user_ids: [batch]，用户ID
            item_ids: [batch]，物品ID（正样本）
            behavior_sequence: [batch, seq_len]，用户历史行为物品ID序列
            behavior_mask: [batch, seq_len]，True表示padding位置
            return_embeddings: 是否返回embedding

        Returns:
            scores: [batch]，用户-物品匹配分数（越高越匹配）
            embeddings: dict，包含user_repr和item_repr（如果return_embeddings=True）
        """
        # 用户塔前向
        user_repr = self.user_tower(
            user_ids,
            behavior_sequence,
            behavior_mask,
            gender_ids=gender_ids,
            age_ids=age_ids,
        )  # [batch, output_dim]

        # 物品塔前向
        item_repr = self.item_tower(
            item_ids, video_category_ids=video_category_ids
        )  # [batch, output_dim]

        # 归一化后直接做内积，等价于余弦相似度
        scores = (user_repr * item_repr).sum(dim=-1) / self.temperature  # [batch]

        if return_embeddings:
            embeddings = {"user_repr": user_repr, "item_repr": item_repr}
            return scores, embeddings

        return scores, None

    def get_user_embedding(
        self,
        user_ids: torch.Tensor,
        behavior_sequence: torch.Tensor,
        behavior_mask: Optional[torch.Tensor] = None,
        gender_ids: Optional[torch.Tensor] = None,
        age_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """获取用户向量，用于离线索引构建或召回。"""
        return self.user_tower(
            user_ids,
            behavior_sequence,
            behavior_mask,
            gender_ids=gender_ids,
            age_ids=age_ids,
        )

    def get_item_embedding(
        self,
        item_ids: torch.Tensor,
        video_category_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """获取物品向量，用于离线索引构建或召回。"""
        return self.item_tower(item_ids, video_category_ids=video_category_ids)

    def predict_batch(
        self,
        user_repr: torch.Tensor,
        item_reprs: torch.Tensor,
    ) -> torch.Tensor:
        """批量计算用户与候选物品的匹配分数。"""
        # 矩阵乘法一次计算所有候选分数
        scores = torch.matmul(user_repr, item_reprs.t()) / self.temperature

        if user_repr.size(0) == 1:
            scores = scores.squeeze(0)

        return scores


def build_two_tower_model(
    user_vocab_size: int,
    item_vocab_size: int,
    gender_vocab_size: Optional[int] = None,
    age_vocab_size: Optional[int] = None,
    video_category_vocab_size: Optional[int] = None,
    embedding_dim: int = 64,
    transformer_heads: int = 4,
    transformer_layers: int = 2,
    user_tower_dims: list = None,
    item_tower_dims: list = None,
    output_dim: int = 64,
    dropout: float = 0.1,
    max_seq_len: int = 100,
    temperature: float = 0.05,
) -> TwoTowerRecallModel:
    """构建双塔召回模型，并补齐默认超参数。"""
    if user_tower_dims is None:
        user_tower_dims = [256, 128]
    if item_tower_dims is None:
        item_tower_dims = [256, 128]

    model = TwoTowerRecallModel(
        user_vocab_size=user_vocab_size,
        item_vocab_size=item_vocab_size,
        gender_vocab_size=gender_vocab_size,
        age_vocab_size=age_vocab_size,
        video_category_vocab_size=video_category_vocab_size,
        embedding_dim=embedding_dim,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        user_tower_dims=user_tower_dims,
        item_tower_dims=item_tower_dims,
        output_dim=output_dim,
        dropout=dropout,
        max_seq_len=max_seq_len,
        temperature=temperature,
    )

    return model
