import matplotlib.pyplot as plt
import torch
from math import cos, pi

def adjust_learning_rate(optimizer, current_epoch,
                         max_epoch=50000,
                         warmup_epoch=0,
                         lr_min=0,
                         lr_max=0.1):
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    elif current_epoch < max_epoch:
        lr = lr_min + (lr_max - lr_min) * (
                    1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    else:
        lr = lr_min + (lr_max - lr_min) * (
                1 + cos(pi * (current_epoch-max_epoch) / (max_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr