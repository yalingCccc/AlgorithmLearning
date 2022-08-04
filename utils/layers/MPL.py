import torch
import torch.nn as nn
from utils.layers.ActiveLayer import activation_layer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LinearLayer(nn.Module):
    def __init__(self, input_dim, dims, act_func='Relu', batch_norm=False, dropout=0, sigmoid=False):
        super(LinearLayer, self).__init__()
        self.input_dim = input_dim
        self.dims = dims
        self.act_func = act_func
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.sigmoid = sigmoid

        fc_layers = []
        input_d = self.input_dim
        for index, d in enumerate(self.dims):
            if index != 0:
                # batch norm
                if self.batch_norm:
                    fc_layers.append(nn.BatchNorm1d(input_d))
                # activation
                fc_layers.append(activation_layer(act_func, input_d))
                # drop out
                if self.dropout != 0:
                    fc_layers.append(nn.Dropout(dropout))
            # linear
            fc_layers.append(nn.Linear(input_d, d))

            # update input size
            input_d = d
        if sigmoid:
            fc_layers.append(nn.Sigmoid())
        self.linear = nn.Sequential(*fc_layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            m.to(device)

    def forward(self, x):
        if self.sigmoid:
            return self.output_layer(self.linear(x))
        else:
            return self.linear(x)
