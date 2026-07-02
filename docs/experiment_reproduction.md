# 排序实验完整复现流程

本文档描述当前仓库中排序实验的标准复现流程，覆盖：

1. 环境与数据检查；
2. 归一化 `pos_weight` 下的 MMoE 结构消融实验；
3. `pos_weight` 的两种归约方式与自动任务权重的对比实验；
4. 验证集选模、测试集评价和结果分析；
5. 常见故障与复现检查清单。

本文命令均从项目根目录执行。当前实现采用随机负采样和随机
`8:1:1` 划分，不是时间切分。

## 1. 固定实验环境

安装依赖：

```bash
pip install -r requirements.txt
```

如果 OpenMP 环境变量为空或为 `0`，先设置为正整数：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

记录软件和硬件环境：

```bash
python --version
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.version.cuda); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
git rev-parse HEAD
git status --short
```

正式实验前运行测试：

```bash
python -m unittest discover -s tests -v
```

查看当前命令行参数：

```bash
python main.py rank ablation --help
```

## 2. 数据要求与检查

默认数据文件：

```text
dataset/ctr_data_1M.csv
```

必要字段：

```text
user_id, item_id, video_category, gender, age,
hist_1 ... hist_10,
click, follow, like, share
```

检查文件和标签分布：

```bash
python - <<'PY'
import pandas as pd

path = "dataset/ctr_data_1M.csv"
frame = pd.read_csv(path, nrows=1_000_000)
tasks = ["click", "follow", "like", "share"]

print("rows:", len(frame))
print("columns:", frame.columns.tolist())
for task in tasks:
    print(
        task,
        "positive=", int((frame[task] == 1).sum()),
        "negative=", int((frame[task] == 0).sum()),
        "rate=", float(frame[task].mean()),
    )
PY
```

`--sample-rows N` 表示读取 CSV 的前 `N` 行，并不是从全量数据中随机抽取
`N` 行。所有正式对比必须使用相同的 `--sample-rows`。

## 3. 所有排序实验共用的协议

### 3.1 Click 负采样

每个 seed 分别执行：

1. 保留读取范围内所有 `click=1` 的样本；
2. 随机抽取两倍数量的 `click=0` 样本；
3. 得到 click 正负比例约为 `1:2` 的实验数据。

由于 click 已经做了负采样，任务权重实验中：

```text
click_pos_weight = 1
```

### 3.2 数据划分

负采样后，使用相同 seed 随机打乱并划分：

```text
训练集：80%
验证集：10%
测试集：10%
```

- 训练集更新参数；
- 验证集用于早停和选择最佳 epoch；
- 恢复最佳权重后，测试集只评价一次；
- 同一个 seed 下，各实验组的数据、初始化和 batch 顺序一致。

### 3.3 模型和训练参数

本文正式命令统一使用：

| 参数 | 数值 |
| --- | ---: |
| Embedding 维度 | 32 |
| 隐藏层维度 | 128 |
| MLP 层数 | 2 |
| Expert 数量 | 2 |
| Attention heads | 4 |
| Dropout | 0.1 |
| Batch size | 1024 |
| 最大 epoch | 10 |
| AdamW 学习率 | 1e-4 |
| Weight decay | 0 |
| 辅助损失权重 | 0.1 |
| 早停 patience | 2 |
| 早停 min delta | 0.001 |
| Seeds | 42、43、44、45、46 |

如需与其他结果比较，以上参数必须保持一致。

### 3.4 指标与最佳 epoch

每个任务分别计算：

- AUC；
- GAUC；
- LogLoss。

`mean_auc`、`mean_gauc` 和 `mean_logloss` 是四个有效任务指标的等权平均。
GAUC 按用户有效样本数加权，单一标签用户不参与该任务的 GAUC。

当前代码使用验证集 `mean_gauc` 选择最佳 epoch：

```text
monitor = mean_gauc
mode = max
```

这意味着最佳 checkpoint 可能牺牲 AUC 或 LogLoss。比较实验时必须对所有组
使用同一 checkpoint 规则，不能根据测试集重新选择 epoch。

## 4. 实验一：MMoE 结构消融

运行参数：

```bash
--experiment-suite architecture
```

实验组如下：

| 实验 | Item 特征 | Expert mask | Personalized gate | Task bias | Target attention | Aux loss | Profile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | × | × | × | × | × | × | × |
| A1 | ✓ | × | × | × | × | × | × |
| A2 | ✓ | ✓ | × | × | × | × | × |
| A3 | ✓ | ✓ | ✓ | × | × | × | × |
| A4 | ✓ | ✓ | ✓ | ✓ | × | × | × |
| A5 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × |
| A6 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × |
| A7 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| B1 | ✓ | × | × | × | ✓ | × | × |
| B2 | ✓ | × | × | × | × | ✓ | × |

A0→A7 是累加式消融；B1/B2 用于观察 target attention 和 auxiliary loss
脱离其他组件后的独立效果。所有结构实验均固定使用归一化 `pos_weight`
BCE 和 `1:1:1:1` 任务间权重，确保架构变化是组间唯一变量。

### 4.1 冒烟测试

```bash
python main.py rank ablation \
  --experiment-suite architecture \
  --data-path dataset/ctr_data_1M.csv \
  --sample-rows 200000 \
  --epochs 3 \
  --batch-size 1024 \
  --embedding-dim 32 \
  --hidden-dim 128 \
  --num-layers 2 \
  --num-experts 2 \
  --num-heads 4 \
  --seeds 42 \
  --device cuda \
  --output-dir outputs/rank/architecture_pilot
```

### 4.2 正式实验

```bash
python main.py rank ablation \
  --experiment-suite architecture \
  --data-path dataset/ctr_data_1M.csv \
  --sample-rows 1000000 \
  --epochs 10 \
  --batch-size 1024 \
  --embedding-dim 32 \
  --hidden-dim 128 \
  --num-layers 2 \
  --num-experts 2 \
  --num-heads 4 \
  --dropout 0.1 \
  --auxiliary-weight 0.1 \
  --lr 1e-4 \
  --weight-decay 0 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --early-stopping-patience 2 \
  --early-stopping-min-delta 0.001 \
  --seeds 42 43 44 45 46 \
  --device cuda \
  --output-dir outputs/rank/architecture_final
```

结构消融默认启用归一化 `pos_weight`。正样本权重只根据当前 seed 的训练集
计算，具体定义与 5.1 节相同。

## 5. 实验二：任务权重方法

运行参数：

```bash
--experiment-suite task-weighting
```

四个实验组都固定使用完整的 A7 架构：

| 实验 | 任务内部损失 | 任务间权重 |
| --- | --- | --- |
| TW0_pos_weight_mean | pos_weight BCE，按样本数平均 | 固定 1:1:1:1 |
| TW1_equal | pos_weight BCE，按有效权重和归一化 | 固定 1:1:1:1 |
| TW2_gradnorm | pos_weight BCE，按有效权重和归一化 | GradNorm |
| TW3_uncertainty | pos_weight BCE，按有效权重和归一化 | Uncertainty Weighting |

四组的 item/profile 特征、MMoE、attention、辅助损失、初始化、数据以及
主模型 AdamW 参数完全相同。GradNorm 和 Uncertainty Weighting 仅增加各自
任务权重所需的优化参数。`TW0_pos_weight_mean` 和 `TW1_equal` 都使用相同的
`pos_weight`；二者唯一差异是前者除以 batch 样本数，后者除以 batch 的
有效权重和，用于单独衡量损失归约方式的影响。

### 5.1 Pos weight

只使用当前 seed 的训练集统计：

```text
click = 1
follow = follow_negative / follow_positive
like = like_negative / like_positive
share = share_negative / share_positive
```

默认不截断。程序启动时会打印实际权重，并写入每个结果文件。

`TW0_pos_weight_mean` 使用 PyTorch 默认的样本平均：

$$
L_t =
\frac{1}{N}
\sum_i \operatorname{BCEWithLogits}_i(\mathrm{pos\_weight}_t)
$$

其余三组使用有效权重和归一化：

$$
L_t =
\frac{\sum_i \operatorname{BCEWithLogits}_i(\mathrm{pos\_weight}_t)}
{\sum_i w_{t,i}}
$$

两种方式中，正样本的有效权重都是 `pos_weight_t`，负样本权重都是 1；
差别仅在分母是样本数 \(N\)，还是有效权重和 \(\sum_i w_{t,i}\)。


### 5.2 固定等权

```text
main_loss = click_loss + follow_loss + like_loss + share_loss
total_loss = main_loss + 0.1 * auxiliary_loss
```

### 5.3 GradNorm

- 初始四个任务权重均为 1；
- 在 MMoE shared experts 上计算各任务梯度范数；
- `alpha=0.5`；
- 任务权重使用独立 Adam，默认学习率 `1e-3`；
- 四个任务权重之和保持为 4；
- 单任务权重限制在 `[0.2, 5.0]`；
- auxiliary loss 固定为 0.1，不参与 GradNorm。

GradNorm 每个 batch 需要额外计算四组共享层梯度，因此比另外两组更慢、
显存占用更高。

### 5.4 Uncertainty Weighting

每个任务学习一个 `log_variance`：

$$
L_{\text{main}}
=
\sum_t
\left[
\exp(-s_t)L_t+s_t
\right]
$$

所有 `s_t` 初始化为 0，因此初始有效权重均为 1。它们参与 AdamW 优化，
但不使用 weight decay。auxiliary loss 仍固定为 0.1。

### 5.5 冒烟测试

```bash
python main.py rank ablation \
  --experiment-suite task-weighting \
  --data-path dataset/ctr_data_1M.csv \
  --sample-rows 200000 \
  --epochs 3 \
  --batch-size 1024 \
  --embedding-dim 32 \
  --hidden-dim 128 \
  --num-layers 2 \
  --num-experts 2 \
  --num-heads 4 \
  --seeds 42 \
  --device cuda \
  --output-dir outputs/rank/task_weighting_pilot
```

### 5.6 正式实验

```bash
python main.py rank ablation \
  --experiment-suite task-weighting \
  --data-path dataset/ctr_data_1M.csv \
  --sample-rows 1000000 \
  --epochs 10 \
  --batch-size 1024 \
  --embedding-dim 32 \
  --hidden-dim 128 \
  --num-layers 2 \
  --num-experts 2 \
  --num-heads 4 \
  --dropout 0.1 \
  --auxiliary-weight 0.1 \
  --lr 1e-4 \
  --weight-decay 0 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --early-stopping-patience 2 \
  --early-stopping-min-delta 0.001 \
  --gradnorm-alpha 0.5 \
  --gradnorm-lr 1e-3 \
  --gradnorm-min-weight 0.2 \
  --gradnorm-max-weight 5.0 \
  --seeds 42 43 44 45 46 \
  --device cuda \
  --output-dir outputs/rank/task_weighting_final
```
```bash
python main.py rank ablation \
  --experiment-suite task-weighting \
  --seeds 42 43\
  --device cuda \
  --output-dir outputs/rank/task_weighting_final
```
默认直接使用训练集 `negative / positive`。如果出现梯度震荡或极端稀疏任务
权重过大，可以另开一组敏感性实验：

```bash
--pos-weight-cap 100
```

截断实验必须使用新的输出目录，不能覆盖正式实验：

```text
outputs/rank/task_weighting_cap100
```

## 6. 输出文件

每个实验目录包含：

```text
output_dir/
├── per_seed_results.csv
├── summary.csv
├── seed_42/
│   ├── summary.csv
│   ├── <experiment>_history.json
│   └── ...
├── seed_43/
└── ...
```

### 6.1 per_seed_results.csv

每行是一个 seed 下的一个实验，主要字段：

- `best_epoch`：验证集 `mean_gauc` 最佳 epoch；
- `validation_*`：最佳 epoch 的验证集指标；
- `test_*`：恢复最佳模型后的测试集指标；
- `pos_weight_*`：该 seed 训练集计算出的正样本权重；
- `validation_task_weight_*`：最佳 epoch 的任务权重；
- `completed_epochs`、`stopped_early`：训练过程信息。

### 6.2 summary.csv

对多个 seed 计算均值和总体标准差：

```text
<metric>_mean
<metric>_std
```

若只运行一个 seed，所有标准差都会是 0，不能用于稳定性结论。

### 6.3 history.json

记录：

- 实验配置和模型超参数；
- train/validation/test 样本数；
- pos weight；
- 每轮训练损失和验证指标；
- GradNorm/UW 每轮任务权重；
- 最佳验证指标和最终测试指标。

消融流程目前不保存模型 checkpoint，只保存评价结果。

## 7. 结果汇总

### 7.1 查看总体结果

```bash
python - <<'PY'
import pandas as pd

path = "outputs/rank/task_weighting_final/summary.csv"
frame = pd.read_csv(path)

columns = [
    "name",
    "best_epoch_mean",
    "test_mean_auc_mean",
    "test_mean_gauc_mean",
    "test_mean_logloss_mean",
    "validation_task_weight_click_mean",
    "validation_task_weight_follow_mean",
    "validation_task_weight_like_mean",
    "validation_task_weight_share_mean",
]
print(
    frame[columns]
    .sort_values("test_mean_gauc_mean", ascending=False)
    .to_string(index=False)
)
PY
```

### 7.2 计算预先约定的业务加权 GAUC

权重顺序为：

```text
click=0.50, like=0.25, follow=0.15, share=0.10
```

```bash
python - <<'PY'
import pandas as pd

path = "outputs/rank/task_weighting_final/summary.csv"
frame = pd.read_csv(path)
frame["weighted_test_gauc"] = (
    0.50 * frame["test_click_gauc_mean"]
    + 0.25 * frame["test_like_gauc_mean"]
    + 0.15 * frame["test_follow_gauc_mean"]
    + 0.10 * frame["test_share_gauc_mean"]
)

print(
    frame[
        [
            "name",
            "weighted_test_gauc",
            "test_mean_auc_mean",
            "test_mean_logloss_mean",
        ]
    ]
    .sort_values("weighted_test_gauc", ascending=False)
    .to_string(index=False)
)
PY
```

该加权指标用于最终离线报告，不会改变当前代码按验证集 `mean_gauc`
选择 checkpoint 的行为。

### 7.3 查看逐 seed 配对差异

```bash
python - <<'PY'
import pandas as pd

path = "outputs/rank/task_weighting_final/per_seed_results.csv"
frame = pd.read_csv(path)
frame["weighted_test_gauc"] = (
    0.50 * frame["test_click_gauc"]
    + 0.25 * frame["test_like_gauc"]
    + 0.15 * frame["test_follow_gauc"]
    + 0.10 * frame["test_share_gauc"]
)
wide = frame.pivot(
    index="seed",
    columns="name",
    values="weighted_test_gauc",
)

print(wide)
print("\nEffective-weight normalization - Sample mean")
print((wide["TW1_equal"] - wide["TW0_pos_weight_mean"]).describe())
print("\nGradNorm - Equal")
print((wide["TW2_gradnorm"] - wide["TW1_equal"]).describe())
print("\nUncertainty - Equal")
print((wide["TW3_uncertainty"] - wide["TW1_equal"]).describe())
PY
```

## 8. 离线选择规则

在开始实验前固定主指标和保护指标，避免看完测试结果再改变规则。

建议主指标：

```text
0.50 × click_GAUC
+ 0.25 × like_GAUC
+ 0.15 × follow_GAUC
+ 0.10 × share_GAUC
```

建议保护条件：

```text
相对对照组：
Mean AUC 下降不超过 0.002
Mean LogLoss 上升不超过 0.003
任一任务 GAUC 下降不超过 0.005
```

建议胜出条件：

1. 五个 seed 的平均加权 GAUC 至少提高 0.002；
2. 多数 seed 的提升方向一致；
3. 自动任务权重没有长期贴近上下限；
4. 没有明显训练震荡或 NaN；
5. 自动加权与 `TW1_equal` 差异很小时，优先选择更简单的 `TW1_equal`；
6. `TW0_pos_weight_mean` 与 `TW1_equal` 单独用于判断有效权重和归一化
   是否有稳定收益。

正式调参只能使用训练集和验证集。如果要调整 GradNorm alpha、学习率或
pos weight cap，必须使用新的实验目录；参数确定后，再运行一次锁定配置并
报告测试集结果。

## 9. 常见问题

### 9.1 CUDA OOM

GradNorm 显存开销最大。依次尝试：

```text
batch-size: 1024 → 512 → 256
```

所有实验组必须使用相同 batch size；不能只降低 GradNorm 的 batch size。

### 9.2 某任务没有正样本

程序会拒绝计算该任务的 `pos_weight`。增大 `--sample-rows`，不要把
`pos_weight` 人工设为任意大数。

### 9.3 Pos weight 极大

先确认训练集正样本数。如果数据有效但任务极稀疏，可以额外运行
`--pos-weight-cap 100` 敏感性实验。不要覆盖未截断实验。

### 9.4 GradNorm 权重不变化

确认：

- 使用了 `--experiment-suite task-weighting`；
- history 中 `TW2_gradnorm` 的 `task_weight_*` 随 epoch 变化；
- `--gradnorm-lr` 不是 0；
- 训练不只有极少 batch。

### 9.5 Uncertainty 权重几乎相同

训练初期四个归一化任务损失可能接近，因此权重变化较慢。应查看完整
epoch 曲线，不根据第一个 epoch 下结论。

## 10. 复现检查清单

每次正式实验确认：

- [ ] 记录 Git commit 和 `git status`；
- [ ] 记录 Python、PyTorch、CUDA 和 GPU；
- [ ] 使用相同数据文件和 `--sample-rows`；
- [ ] 使用相同五个 seeds；
- [ ] 使用相同模型和优化器参数；
- [ ] 各套实验使用独立输出目录；
- [ ] 不根据测试集选择 epoch；
- [ ] 检查每个 seed 的 split sizes 和 pos weights；
- [ ] 检查 GradNorm/UW 的任务权重轨迹；
- [ ] 汇报均值、标准差和逐 seed 配对差异；
- [ ] 保留完整命令、控制台日志、CSV 和 JSON。

建议保存控制台日志：

```bash
mkdir -p outputs/rank/logs

python main.py rank ablation \
  --experiment-suite task-weighting \
  --data-path dataset/ctr_data_1M.csv \
  --sample-rows 1000000 \
  --epochs 10 \
  --batch-size 1024 \
  --embedding-dim 32 \
  --hidden-dim 128 \
  --num-layers 2 \
  --num-experts 2 \
  --num-heads 4 \
  --dropout 0.1 \
  --auxiliary-weight 0.1 \
  --lr 1e-4 \
  --weight-decay 0 \
  --early-stopping-patience 2 \
  --early-stopping-min-delta 0.001 \
  --gradnorm-alpha 0.5 \
  --gradnorm-lr 1e-3 \
  --seeds 42 43 44 45 46 \
  --device cuda \
  --output-dir outputs/rank/task_weighting_final \
  2>&1 | tee outputs/rank/logs/task_weighting_final.log
```
