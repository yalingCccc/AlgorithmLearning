import torch
import torch.nn as nn
from functools import partial
import math

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DiceActivation(nn.Module):
    def __init__(self, input_dim):
        super(DiceActivation, self).__init__()
        self.input_dim = input_dim
        self.norm = nn.BatchNorm1d(input_dim)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.zeros(input_dim).float(), requires_grad=True).to(device)

    def forward(self, x):
        inputs_normed = self.norm(x)
        x_p = self.sigmoid(inputs_normed)
        return self.alpha * (1.0 - x_p) * x + x_p * x


class PreluActivation(nn.Module):
    def __init__(self, input_dim):
        super(PreluActivation, self).__init__()
        self.alpha = nn.Parameter(torch.nn.init.constant_(torch.zeros(input_dim), val=0.1), requires_grad=True).to(device)

    def forward(self, x):
        zero = torch.tensor(0.0).to(device)
        return torch.maximum(zero, x) + self.alpha * torch.minimum(zero, x)

def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def activation_layer(activation, input_dim):
    activation = activation.lower()
    if activation is None or activation == '':
        return None
    if activation == 'dice':
        return DiceActivation(input_dim)
    if activation == 'prelu':
        return PreluActivation(input_dim)
    if activation == 'grelu':
        return gelu
    if activation == 'leaky_relu' or activation == 'leaky':
        return partial(nn.LeakyReLU, alpha=0.01)
    if activation == 'relu':
        return nn.ReLU()
    else:
        return None