# 多任务推荐系统

这是一个基于 PyTorch 的两阶段推荐系统实验项目：

- 双塔召回：编码用户与物品，支持 BCE、InfoNCE、Exact 和 FAISS 召回。
- MMoE 排序：联合预测 `click`、`follow`、`like`、`share`，支持个性化门控、目标注意力和双塔辅助损失。

所有正式流程均通过根目录的 `main.py` 执行，不再依赖 Notebook。

## 项目结构

```text
.
├── main.py
├── recommender/
│   ├── data_process/
│   │   ├── rank.py
│   │   └── recall.py
│   ├── models/
│   │   ├── mmoe.py
│   │   └── two_tower.py
│   ├── training/
│   │   ├── rank.py
│   │   └── recall.py
│   ├── evaluation.py
│   ├── retrieval.py
│   └── ablation.py
├── scripts/
│   └── sample_data.py
├── tests/
├── dataset/
└── outputs/
```

## 安装

建议使用 Python 3.9 或更高版本：

```bash
pip install -r requirements.txt
```

FAISS 仅用于近似最近邻召回；使用 `--no-use-faiss` 时会自动采用精确全量打分。

## 数据格式

默认数据文件为 `dataset/ctr_data_1M.csv`，需要包含：

| 字段 | 说明 |
| --- | --- |
| `user_id`、`item_id` | 用户和当前候选物品 ID |
| `video_category` | 候选物品类目 |
| `gender`、`age` | 用户画像 |
| `hist_1`～`hist_10` | 历史行为物品，缺失值作为 padding |
| `click`、`follow`、`like`、`share` | 四个二分类任务标签 |

`watching_times` 属于曝光后行为，不会作为排序特征使用。

## 排序

训练 MMoE：

```powershell
python main.py rank train --data-path dataset/ctr_data_1M.csv
```

排序训练会先保留全部 `click=1` 样本，并按 Tenrec 官方协议随机抽取两倍
`click=0` 样本，再将采样结果随机划分为 80% 训练集、10% 验证集和
10% 测试集。负样本采样和数据划分均由 `--seed` 控制。

排序训练和消融实验共用以下主要默认参数：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--epochs` | `10` | 最大训练轮数，可能因早停提前结束 |
| `--batch-size` | `1024` | 同时用于训练、验证和测试 DataLoader |
| `--embedding-dim` | `32` | Embedding 维度 |
| `--hidden-dim` | `128` | Expert、Gate 和 Tower 隐藏层维度 |
| `--num-layers` | `2` | MLP 隐藏层数 |
| `--num-experts` | `3` | MMoE Expert 数量 |
| `--num-heads` | `4` | 目标注意力头数 |
| `--dropout` | `0.1` | Dropout 比例 |
| `--auxiliary-weight` | `0.1` | 双塔辅助损失权重 |
| `--lr` | `1e-4` | AdamW 学习率 |
| `--weight-decay` | `0` | AdamW 权重衰减 |
| `--val-ratio` | `0.1` | 验证集比例 |
| `--test-ratio` | `0.1` | 测试集比例 |
| `--seed` | `100` | 负采样、数据划分和模型初始化种子 |

个性化 gate 和 task bias 默认开启，可分别关闭：

```powershell
python main.py rank train --no-use-personalized-gate
python main.py rank train --no-use-task-bias
```

快速 CPU 验证：

```powershell
python main.py rank train --sample-rows 2000 --epochs 1 --device cpu
```

快速运行多随机种子消融实验：

```powershell
python main.py rank ablation `
  --sample-rows 20000 `
  --epochs 3 `
  --batch-size 512 `
  --seeds 42 43 44
```

采样后的数据默认按 8:1:1
划分为训练集、验证集和测试集；验证集用于早停和选择最佳 epoch，
恢复最佳权重后只在测试集上进行一次最终评价。

排序阶段为每个任务分别计算 AUC、GAUC、LogLoss，并输出
`mean_auc`、`mean_gauc`、`mean_logloss`。GAUC 按有效用户样本数加权，
只有单一标签的用户不参与计算；最佳 epoch 按 `mean_gauc` 选择。

消融结果默认保存至 `outputs/rank/ablation/`，其中验证指标使用
`validation_*` 前缀，恢复最佳权重后的最终测试指标使用 `test_*` 前缀。

## 召回

召回模型内部固定使用连续物品 ID 映射，不再提供 `raw` 映射模式；
生成候选时会自动还原为原始 `item_id`。

训练双塔：

```powershell
python main.py recall train --data-path dataset/ctr_data_1M.csv --epochs 6 --loss-type infonce --eval-k 5 10 20
```
训练会在 `outputs/recall/` 保存最佳 Checkpoint 和历史指标。Checkpoint
包含模型配置、稳定的物品/画像/类目映射和词表规模。召回结果会自动转换回
原始 `item_id`，padding ID `0` 不会进入候选集。

验证阶段在完整候选物品库上按用户计算 `Recall@K`、`HitRate@K` 和
`NDCG@K`，`--eval-k` 可配置一个或多个 K；默认使用 `5 10 20`，并按最大
K 对应的验证集 `NDCG@K` 保存最佳 Checkpoint。只有验证集中至少存在一个
正样本物品的用户参与这些指标的宏平均。

加载 Checkpoint 并生成候选：

```powershell
python main.py recall generate `
  --checkpoint outputs/recall/two_tower_contiguous_infonce_best.pt `
  --data-path dataset/ctr_data_1M.csv `
  --top-k 20 `
  --num-users 20
```

精确召回：

```powershell
python main.py recall generate `
  --checkpoint outputs/recall/two_tower_contiguous_infonce_best.pt `
  --no-use-faiss
```

## 测试

如果系统中的 OpenMP 线程变量被设置成了 `0`，请先改为正整数：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

```powershell
python -m unittest discover -s tests -v
```

查看全部命令参数：

```powershell
python main.py --help
python main.py rank train --help
python main.py recall train --help
```
