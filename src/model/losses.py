import torch
import torch.nn as nn


class L1Loss(nn.Module):
    """The L1 loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask=None):

        if mask is None:
            loss = torch.mean(torch.sum((output - target) ** 2, 1) / 2.0)
        else:
            loss = torch.mean(mask * torch.sum((output - target) ** 2, 1) / 2.0)
        return loss
