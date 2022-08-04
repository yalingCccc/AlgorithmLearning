import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class L2_Normalization(nn.Module):
    def __init__(self, epsilon=1e-12):
        super(L2_Normalization, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x_0 = x.clone()
        x = torch.sum(x ** 2, dim=-1).view(-1, 1)
        epsilon = torch.full_like(x, self.epsilon)
        x = torch.cat([x, epsilon], dim=1)
        x = torch.max(x, dim=-1)[0]
        output = x_0 / torch.sqrt(x).view(-1, 1)
        return output
