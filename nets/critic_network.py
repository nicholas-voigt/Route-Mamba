import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

    def forward(self, x):
        return x

