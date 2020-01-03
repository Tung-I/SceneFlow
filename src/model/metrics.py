import torch
import torch.nn as nn
import numpy as np
from skimage.morphology import label



class EPE(nn.Module):
    """The End Point Error.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask=None):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.
        Returns:
            metric (torch.Tensor) (C): The dice scores for each class.
        """
        # Get the one-hot encoding of the prediction and the ground truth label.

        # maxi = torch.max(target)
        # mini = torch.min(target)
        # output = (output - mini) / (maxi - mini)
        # target = (target - mini) / (maxi - mini)
        if mask is not None:
            epe = torch.norm(output-target, p=2, dim=1)
            epe = epe * mask
            epe = torch.sum(epe) / torch.sum(mask)
        else:
            epe = torch.norm(output-target, p=2, dim=1).mean()

        return epe

class EndPointError(nn.Module):
    """The End Point Error.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask=None):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.
        Returns:
            metric (torch.Tensor) (C): The dice scores for each class.
        """
        # Get the one-hot encoding of the prediction and the ground truth label.

        error = torch.sqrt(torch.sum((output - target)**2, 1) + 1e-20)
        gtflow_len = torch.sqrt(torch.sum(target*target, 2) + 1e-20) # B,N
        if mask is not None:
            mask_sum = torch.sum(mask, 1)
            EPE = torch.sum(error * mask, 1)[mask_sum > 0] / (mask_sum[mask_sum > 0] + 1e-20)
            EPE = torch.mean(EPE)
        else:
            EPE = torch.mean(error)

        return EPE



class F1Score(nn.Module):
    """The accuracy
    """
    def __init__(self, threshold):
        super().__init__()
        self.th = threshold

    def forward(self, output, target, mask=None):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.
        Returns:
            metric (torch.Tensor) (C): The dice scores for each class.
        """
        # Get the one-hot encoding of the prediction and the ground truth label.

        err = torch.norm(output-target, p=2, dim=1)
        flow_len = torch.norm(target, p=2, dim=1)

        # print(error.shape)
        # print((error <= self.th).float().shape)
        # print(mask.shape)
        # print(((error/gtflow_len <= self.th)*mask))

        if mask is not None:
            f1 = ((err/flow_len <= self.th) * mask).byte()
            f1 = torch.sum(f1.float())
            f1 = f1 / (torch.sum(mask) + 1e-20)
        else:
            f1 = (err/flow_len <= self.th).byte()
            f1 = torch.mean(f1.float())

        return f1

