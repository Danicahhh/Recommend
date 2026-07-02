训练集：训练模型，更新参数
验证集：调超参数、早停、选最优模型、选择最优epoch、选择最佳 checkpoint
测试集：计算最终指标，只评估一次，作为最终消融结果，把测试集结果写进论文

pos_weight 是：
在某个任务里面，正样本太少，所以我要更重视正样本。
pos_weight：解决 click / like / follow / share 各自的正负样本不平衡问题

task_weight 是：
在多个任务之间，某些任务更重要或更难学，所以我要调整这个任务在总 loss 里的占比。
task_weight：解决 click / like / follow / share 四个任务之间的训练平衡问题