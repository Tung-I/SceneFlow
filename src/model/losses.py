import torch
import torch.nn as nn


class HuberLoss(nn.Module):
    """The L1 loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask=None):
        # print(output.size)
        # print(target.size)
        if mask is None:
            loss = torch.mean(torch.sum((output - target + 1e-20) ** 2, 1) / 2.0)
        else:
            loss = torch.mean(mask * torch.sum((output - target) ** 2, 1) / 2.0)
        return loss

class MyL2Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask=None):
        # minimum = torch.min(target)
        # maximum = torch.max(target)
        # output = (output - minimum) / (maximum - minimum)
        # target = (target - minimum) / (maximum - minimum)
        loss = torch.norm(output - target, p=2, dim=1).mean()
        return loss
