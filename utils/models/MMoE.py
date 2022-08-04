import torch
import torch.nn as nn
from utils.layers.ActiveLayer import activation_layer
from utils.layers.MPL import LinearLayer


class MMoE(nn.Module):
    """
    Multi-gate Mixture-of-Experts model.
    """

    def __init__(self, input_dim, num_expert, num_task, expert_dims, use_top_layer,
                 top_layer_dims=None, activation='relu', dropout=0, batch_norm=False, sigmoid=False):
        super(MMoE, self).__init__()
        self.input_dim = input_dim
        self.expert_dims = expert_dims
        self.num_expert = num_expert
        self.num_task = num_task
        self.user_top_layer = use_top_layer
        self.top_layer_dims = top_layer_dims
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.sigmoid = sigmoid

        # 网络定义
        experts, gates, top_layer = [], [], []
        for i in range(self.num_expert):
            exp_i = LinearLayer(self.input_dim, self.expert_dims, act_func=self.activation, batch_norm=self.batch_norm,
                                dropout=self.dropout, sigmoid=False)
            experts.append(exp_i)

        for task in range(self.num_task):
            gates.append(nn.Linear(self.input_dim, self.num_expert))
            if use_top_layer:
                top_i = LinearLayer(self.expert_dims[-1], self.top_layer_dims, act_func=self.activation,
                                    batch_norm=self.batch_norm, dropout=self.dropout, sigmoid=False)
                top_layer.append(top_i)

        self.experts = nn.ModuleList(experts)
        self.gates = nn.ModuleList(gates)
        self.mpls = nn.ModuleList(top_layer)

    def create_mpl(self, task_name, input_dim, dims):
        layers = nn.Sequential()
        for i, dim in enumerate(dims):
            if i != 0:
                layers.add_module('%s_act_%s' % (task_name, i - 1), activation_layer(self.activation, input_dim))
            layers.add_module('%s_dense_%s' % (task_name, i), nn.Linear(input_dim, dim))
            input_dim = dim
        return layers

    def forward(self, inputs):
        expert_outs, outputs = [], []
        # 计算experts输出
        for expert in self.experts:
            exp_i_out = expert(inputs)
            expert_outs.append(exp_i_out.unsqueeze(dim=1))  # (batch_size, 1, output_dim)
        expert_outs = torch.cat(expert_outs, dim=1)  # (batch_size, num_experts, output_dim)
        expert_outs = torch.transpose(expert_outs, 1, 2)  # (batch_size, output_dim, num_experts)
        # 计算每个task输出
        for i, gate in enumerate(self.gates):
            x = gate(inputs).unsqueeze(2)  # (batch_size, num_experts, 1)
            x = torch.bmm(expert_outs, x).squeeze(-1)
            if self.user_top_layer:
                x = self.mpls[i](x)
            outputs.append(x)
        return outputs
