# 多任务推荐系统：双塔召回与个性化 MMOE 排序

这是一个基于 PyTorch 的推荐系统实验项目，包含双塔召回模型和多任务排序模型。排序阶段以 MMOE 为基础，同时加入属性专家、个性化门控、任务残差、目标感知注意力和双塔辅助损失，用于联合预测点击、关注、点赞和分享。

## 项目结构

```text
.
├── data/
│   
├── models/
│   ├── baseline_mmoe.py            # DNN 扩展版标准 MMOE
│   ├── rank_mmoe.py                # 优化后的多任务排序模型
│   └── two_tower_recall.py         # 双塔召回模型
├── process/
│   ├── sample.py                   # CSV 数据抽样
│   └── two_tower_data.py           # 双塔数据集与特征映射
├── train/
│   ├── rank/
│   │   ├── train_rank_mmoe.py      # 排序模型训练入口
│   │   ├── ablation_rank_mmoe.py   # 排序模型消融实验
│   │   └── mmoe.ipynb
│   └── recall/
│       └── two_tower.ipynb         # 双塔召回训练与评估
├── requirements.txt
└── README.md
```

## 模型概览

### 双塔召回

召回模型分别编码用户和物品，并使用归一化向量的内积作为匹配分数：

- 用户塔：用户 ID、性别、年龄和历史行为序列；
- 行为编码：位置编码与 Transformer Encoder；
- 物品塔：物品 ID 和视频类目；
- 匹配函数：余弦相似度除以温度系数；
- 推理支持：独立导出用户向量、物品向量以及批量候选打分。

核心实现位于 `models/two_tower_recall.py`。

### 个性化 MMOE 排序

排序模型处理以下四个二分类任务：

```text
click / follow / like / share
```

相较于基础 MMOE，`models/rank_mmoe.py` 包含以下增强：

1. **目标感知多头注意力**
   - 当前候选物品 `target_emb` 作为 Query；
   - 历史行为物品作为 Key 和 Value；
   - 根据候选物品动态提取相关历史兴趣。

2. **属性专家**
   - 每个 Expert 在进入 MLP 前学习独立的特征掩码；
   - 引导不同 Expert 关注不同的特征子空间。

3. **个性化门控**
   - 每个任务拥有独立 Gate；
   - Gate 权重同时受任务网络和当前样本特征影响。

4. **任务专属残差**
   - 每个任务增加从原始排序特征到 Logit 的直接修正路径；
   - 减少 Expert 混合表示不足造成的信息损失。

5. **双塔辅助目标**
   - 对 `user_vec` 和 `item_vec` 分别投影后计算点积；
   - 使用点击标签计算辅助 BCE 损失；
   - 直接约束用户塔和物品塔的匹配表示。

总损失为：

```text
L = Σ task_weight[t] × BCE(task_logit[t], label[t])
    + auxiliary_weight × BCE(two_tower_logit, click_label)
```

双塔辅助头是排序模型的表征约束，不等同于带 batch 内负采样的标准双塔召回损失。由于用户塔使用了目标感知行为表示，排序模型中的 `user_vec` 也不能作为固定用户向量离线缓存。

## 数据格式

默认训练文件为：

```text
data/ctr_data_500k.csv
```

排序训练脚本需要以下字段：

| 字段 | 说明 |
| --- | --- |
| `user_id` | 用户 ID |
| `item_id` | 当前候选物品 ID |
| `video_category` | 当前候选视频类目 |
| `gender` | 用户性别 |
| `age` | 用户年龄 |
| `hist_1` ~ `hist_10` | 用户历史行为物品 ID，空值作为 padding |
| `click` | 点击标签，取值为 0 或 1 |
| `follow` | 关注标签，取值为 0 或 1 |
| `like` | 点赞标签，取值为 0 或 1 |
| `share` | 分享标签，取值为 0 或 1 |

> [!NOTE]
> 原始 Tenrec 数据中的 `watching_times`（官方代码中也称 `play_times`）是用户在该视频上的观看行为次数。项目不会读取或使用该字段，避免在曝光前排序任务中引入未来信息。CSV 可以保留此列，但模型会完全忽略它。

## 环境安装

建议使用 Python 3.9 或更高版本，并在独立虚拟环境中安装依赖。

项目运行的核心依赖为：

```bash
pip install torch pandas numpy scikit-learn
```

如需运行 Notebook：

```bash
pip install jupyter
```

## 快速开始

所有命令均在项目根目录执行。

### 训练排序模型

```bash
python train/rank/train_rank_mmoe.py \
  --data-path data/ctr_data_500k.csv \
  --epochs 5 \
  --batch-size 512 \
  --embedding-dim 64 \
  --num-experts 4 \
  --num-heads 4 \
  --auxiliary-weight 0.1
```

Windows PowerShell 可以使用单行命令：

```powershell
python train/rank/train_rank_mmoe.py --data-path data/ctr_data_500k.csv --epochs 5 --batch-size 512
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--sample-rows` | 全部 | 只读取前 N 行，适合快速验证 |
| `--epochs` | `1` | 训练轮数 |
| `--batch-size` | `512` | Batch 大小 |
| `--embedding-dim` | `64` | Embedding 维度 |
| `--num-experts` | `4` | Expert 数量 |
| `--num-heads` | `4` | Target Attention 头数 |
| `--dropout` | `0.1` | Dropout |
| `--auxiliary-weight` | `0.1` | 双塔辅助损失权重 |
| `--lr` | `1e-3` | AdamW 学习率 |
| `--weight-decay` | `1e-5` | 权重衰减 |
| `--device` | 自动选择 | 例如 `cpu`、`cuda` 或 `cuda:0` |
| `--baseline-mmoe` | 关闭 | 关闭个性化 Gate 和任务残差；当前仍保留属性专家掩码 |
| `--mean-pooling` | 关闭 | 使用历史均值池化替代 Target Attention |

快速冒烟测试：

```powershell
python train/rank/train_rank_mmoe.py --sample-rows 2000 --epochs 1 --device cpu
```

### 运行消融实验

```powershell
python train/rank/ablation_rank_mmoe.py --data-path data/ctr_data_500k.csv --sample-rows 20000 --epochs 3 --seeds 42 43 44
```

消融配置覆盖：

- A0：基础 MMOE，仅使用 ID；
- A1：加入物品侧特征；
- A2：加入属性专家掩码；
- A3：加入个性化 Gate；
- A4：加入任务专属残差；
- A5：加入 Target Attention；
- A6：加入双塔辅助损失；
- A7：加入用户画像特征；
- B1：单独验证 Target Attention；
- B2：单独验证双塔辅助损失。

实验输出默认保存在：

```text
train/rank/result/rank_mmoe_ablation/
├── per_seed_results.csv
├── summary.csv
└── seed_<seed>/
    ├── summary.csv
    └── <experiment>_history.json
```

评估指标包括每个任务的 LogLoss、AUC，以及四个任务有效 AUC 的平均值 `mean_auc`。

### 运行双塔召回

双塔召回当前通过 Notebook 执行：

```text
train/recall/two_tower.ipynb
```

数据加载与模型实现分别位于：

```text
process/two_tower_data.py
models/two_tower_recall.py
```

## 实验注意事项

- 当前排序训练入口只输出训练损失；如需验证集指标，请使用消融实验脚本。
- `watching_times` 属于当前交互产生的曝光后行为，不能作为 click、follow、like、share 的预测输入。
- `embedding_dim` 必须能被 `num_heads` 整除。
- 历史行为中的 `0` 被保留为 padding，真实物品 ID 会映射到非零索引。
- 训练脚本基于当前读取的数据动态建立 ID 映射；部署时需要持久化训练阶段的映射表。
- 类别不平衡时，可以通过 `RankMultiTaskLoss` 的 `pos_weights` 为不同任务设置正样本权重。
- 辅助损失权重过大可能让点击任务主导共享表示，应通过消融实验选择。

## 主要代码入口

- `models/rank_mmoe.py`：优化排序模型与多任务损失；
- `models/baseline_mmoe.py`：DNN 扩展版标准 MMOE，返回任务 Logit 与 Gate 权重；
- `models/two_tower_recall.py`：双塔召回；
- `train/rank/train_rank_mmoe.py`：排序训练；
- `train/rank/ablation_rank_mmoe.py`：多随机种子消融实验。
