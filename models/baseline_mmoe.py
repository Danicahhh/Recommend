from typing import List, Sequence

import torch
import torch.nn as nn # 导入PyTorch的神经网络模块
import torch.nn.functional as F # 导入PyTorch的函数式接口，提供了许多常用的激活函数、损失函数等 

# 定义一个简单的全连接神经网络类 DNN，继承自 nn.Module。这个类可以用来构建专家网络、门控网络和任务塔网络。
class DNN(nn.Module):
    def __init__(
        self,
        input_dim: int, # 输入特征的维度。
        hidden_units: Sequence[int], # 隐藏层的单元数列表。比如 [128, 64] 就表示有两层隐藏层，第一层有 128 个神经元，第二层有 64 个神经元。
        dropout: float = 0.0, # Dropout 比例，默认不丢弃任何神经元。这个是防止过拟合的手段。
        use_output_activation: bool = True, # 是否在最后一层后面也加激活函数。默认加，后面在任务塔里会关闭它。
    ):
        super().__init__() # 调用父类的构造函数，初始化 nn.Module 的内部状态。
        if len(hidden_units) == 0:
            raise ValueError("隐藏层单元列表不能为空")

        layers: List[nn.Module] = [] # 用来存储构建的层。我们会把线性层、激活函数和 dropout 层都放在这个列表里，最后用 nn.Sequential 来组合成一个完整的网络。
        prev_dim = input_dim # 记录当前层的输入维度，初始值是输入特征的维度。每添加一层线性层后，我们会更新这个值为当前层的输出维度，以便下一层使用。
        for i, unit in enumerate(hidden_units): # 遍历隐藏层的单元数列表，依次构建每一层。
            layers.append(nn.Linear(prev_dim, unit)) # 添加一个线性层，输入维度是 prev_dim，输出维度是 unit。
            if i < len(hidden_units) - 1 or use_output_activation: # 如果不是最后一层，或者最后一层需要激活函数，就添加 ReLU 激活函数和 Dropout 层。
                layers.append(nn.ReLU())# 添加 ReLU 激活函数，增加网络的非线性表达能力。如果每一层都不加激活函数，那么多层线性层叠起来，数学上还是一层线性层
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            prev_dim = unit

        self.network = nn.Sequential(*layers) # 用 nn.Sequential 将层列表组合成一个完整的网络。这样我们就可以直接调用 self.network(x) 来前向传播了。

    def forward(self, x: torch.Tensor) -> torch.Tensor: # 定义前向传播函数，这个函数会被 PyTorch 在训练和推理时自动调用。输出维度为[batch_size, hidden_units[-1]]
        return self.network(x)

# 定义 MMOE 模型类，继承自 nn.Module。这个类实现了多任务学习中的 MMOE（Multi-gate Mixture-of-Experts）架构。MMOE 是一种“多任务学习”结构，可以同时处理多个任务
class MMOE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_name_list: Sequence[str], # 任务名称列表，每个任务对应一个输出。比如 ["click", "purchase"] 就表示有两个任务，一个是预测点击率，一个是预测购买率。  
        expert_nums: int = 4, # 专家网络的数量。每个专家网络都是一个 DNN，用来提取输入特征的不同方面的信息。默认是 4 个专家。
        expert_dnn_units: Sequence[int] = (128, 64), # 专家网络的隐藏层单元数列表。比如 (128, 64) 就表示每个专家网络有两层隐藏层，第一层有 128 个神经元，第二层有 64 个神经元。 
        gate_dnn_units: Sequence[int] = (128, 64), # 门控网络的隐藏层单元数列表。每个任务都有一个门控网络，用来计算每个专家对该任务的贡献权重。比如 (128, 64) 就表示每个门控网络有两层隐藏层，第一层有 128 个神经元，第二层有 64 个神经元。
        task_tower_dnn_units: Sequence[int] = (128, 64), # 任务塔网络的隐藏层单元数列表。每个任务都有一个任务塔网络，用来根据专家的输出和门控权重计算该任务的最终输出。比如 (128, 64) 就表示每个任务塔网络有两层隐藏层，第一层有 128 个神经元，第二层有 64 个神经元。
        dropout: float = 0.0,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if expert_nums <= 0:
            raise ValueError("expert_nums must be positive")
        if len(task_name_list) == 0:
            raise ValueError("task_name_list must not be empty")
#  维度	     list[str]	     Sequence[str]
# 类型性质	具体、可变类型	抽象、只读协议，不是真实类型
        self.task_name_list = list(task_name_list) # 将任务名称列表转换成一个普通的 Python 列表，方便后续使用。虽然输入是一个 Sequence，但我们需要一个具体的列表来存储和访问任务名称。
        self.num_tasks = len(task_name_list)

        self.experts = nn.ModuleList(
            [DNN(input_dim, expert_dnn_units, dropout=dropout) for _ in range(expert_nums)]
        ) # 使用 nn.ModuleList 来存储多个专家网络。每个专家网络都是一个 DNN，输入维度是 input_dim，隐藏层单元数是 expert_dnn_units，dropout 比例是 dropout。我们创建 expert_nums 个这样的专家网络。
        expert_out_dim = expert_dnn_units[-1] # 专家网络的输出维度是最后一层隐藏层的单元数。

        self.gate_mlps = nn.ModuleList(
            [DNN(input_dim, gate_dnn_units, dropout=dropout) for _ in range(self.num_tasks)]
        )# 使用 nn.ModuleList 来存储多个门控网络。每个门控网络都是一个 DNN，输入维度是 input_dim，隐藏层单元数是 gate_dnn_units，dropout 比例是 dropout。我们创建 num_tasks 个这样的门控网络，每个任务对应一个门控网络。
        self.gate_logits = nn.ModuleList(
            [nn.Linear(gate_dnn_units[-1], expert_nums, bias=False) for _ in range(self.num_tasks)]
        )# (激活函数)使用 nn.ModuleList 来存储多个线性层，用于计算门控权重的 logits。每个线性层的输入维度是 gate_dnn_units 的最后一层单元数，输出维度是 expert_nums（专家网络的数量），不使用偏置项。我们创建 num_tasks 个这样的线性层，每个任务对应一个线性层。

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
        )# 使用 nn.ModuleList 来存储多个任务塔网络。每个任务塔网络都是一个 DNN，输入维度是 expert_out_dim（专家网络的输出维度），隐藏层单元数是 task_tower_dnn_units，最后一层输出 1 个值（表示该任务的预测结果），dropout 比例是 dropout。我们创建 num_tasks 个这样的任务塔网络，每个任务对应一个任务塔网络。

    def forward(self, x: torch.Tensor, return_prob: bool = False):# 定义前向传播函数，输入是一个张量 x，表示输入特征。return_prob 参数表示是否返回概率值，如果为 True，则对任务塔的输出应用 sigmoid 函数将其转换为概率值。
        if not isinstance(x, torch.Tensor): # 检查输入 x 是否是一个 torch.Tensor，如果不是，则抛出一个类型错误。这个检查可以帮助我们在使用模型时及时发现输入数据类型不正确的问题。
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) # 使用 torch.stack 将这些张量沿着新的维度（dim=1）堆叠起来，得到一个形状为 (batch_size, expert_nums, expert_out_dim) 的张量，表示每个样本在每个专家网络上的输出。

        task_outputs = [] # 初始化一个空列表，用来存储每个任务的输出结果。
        gate_weights_list = [] # 初始化一个空列表，用来存储每个任务的门控权重。我们会在后面计算每个任务的门控权重，并将它们保存在这个列表里，以便后续分析和可视化。
        for task_idx in range(self.num_tasks):
            gate_hidden = self.gate_mlps[task_idx](x) # 对于每个任务，首先通过对应的门控网络对输入 x 进行前向传播（自动调用前向传播函数），得到一个形状为 (batch_size, gate_dnn_units[-1]) 的张量 gate_hidden，表示门控网络的隐藏层输出。    
            gate_weight = F.softmax(self.gate_logits[task_idx](gate_hidden), dim=-1) # 然后通过对应的线性层计算门控权重的 logits，得到一个形状为 (batch_size, expert_nums) 的张量。接着对这个张量应用 softmax 函数，沿着最后一个维度（专家网络的维度）进行归一化，得到一个形状为 (batch_size, expert_nums) 的张量 gate_weight，表示每个专家网络对该任务的贡献权重。这个权重是一个概率分布，所有专家的权重加起来等于 1。
            gate_weights_list.append(gate_weight)
            # B：batch_size 样本数
            # E：expert 专家数量
            # D：单个专家输出特征维度
            # 加权融合专家输出：task_feature = torch.einsum("BE,BED->BD", gate_weight, expert_outputs)等价于：
            # # 第一步：给权重增加最后一维，匹配D
            # gate_weight_expand = gate_weight.unsqueeze(-1)  # [B,E] -> [B,E,1]
            # # 第二步：逐元素相乘
            # weighted_expert = gate_weight_expand * expert_outputs  # [B,E,D]
            # # 第三步：在专家维度dim=1求和，消除E维
            # task_feature = weighted_expert.sum(dim=1)  # [B,D]
            task_feature = torch.einsum("BE,BED->BD", gate_weight, expert_outputs)
            task_logit = self.task_towers[task_idx](task_feature)
            task_outputs.append(torch.sigmoid(task_logit) if return_prob else task_logit)

        return task_outputs, gate_weights_list

# 这是一个“工厂函数”，作用是帮你快速创建 MMOE 模型，不用每次手写 MMOE(...)
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
