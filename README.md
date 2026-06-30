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

快速 CPU 验证：

```powershell
python main.py rank train --sample-rows 2000 --epochs 1 --device cpu
```

运行多随机种子消融实验：

```powershell
python main.py rank ablation --sample-rows 20000 --epochs 3 --seeds 42 43 44
```

排序阶段为每个任务分别计算 AUC、GAUC、LogLoss，并输出
`mean_auc`、`mean_gauc`、`mean_logloss`。GAUC 按有效用户样本数加权，
只有单一标签的用户不参与计算；最佳 epoch 按 `mean_gauc` 选择。

消融结果默认保存至 `outputs/rank/ablation/`。

## 召回

训练双塔：

```powershell
python main.py recall train --data-path dataset/ctr_data_1M.csv --epochs 6 --loss-type infonce
```

训练会在 `outputs/recall/` 保存最佳 Checkpoint 和历史指标。Checkpoint
包含模型配置、数据映射模式和词表规模。

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

```powershell
python -m unittest discover -s tests -v
```

查看全部命令参数：

```powershell
python main.py --help
python main.py rank train --help
python main.py recall train --help
```
