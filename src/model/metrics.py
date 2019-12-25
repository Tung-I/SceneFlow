import torch
import torch.nn as nn
import numpy as np
from skimage.morphology import label


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

        error = np.sqrt(np.sum((output - target)**2, 2) + 1e-20)

        gtflow_len = np.sqrt(np.sum(target*target, 2) + 1e-20) # B,N

        if mask is not None:
            mask_sum = np.sum(mask, 1)
            EPE = np.sum(error * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
            EPE = np.mean(EPE)
        else:
            EPE = np.mean(EPE)

        return EPE


class Accuracy(nn.Module):
    """The End Point Error.
    """
    def __init__(self, threshold):
        super().__init__()
        self.th = torch.tensor(threshold).cuda()

    def forward(self, output, target, mask=None):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.
        Returns:
            metric (torch.Tensor) (C): The dice scores for each class.
        """
        # Get the one-hot encoding of the prediction and the ground truth label.

        error = np.sqrt(np.sum((output - target)**2, 2) + 1e-20)

        gtflow_len = np.sqrt(np.sum(target*target, 2) + 1e-20) # B,N

        if mask is not None:
            acc = np.sum(np.logical_or((error <= self.th)*mask, (error/gtflow_len <= self.th)*mask), axis=1)
            mask_sum = np.sum(mask, 1)
            acc = acc[mask_sum > 0] / mask_sum[mask_sum > 0]
            acc = np.mean(acc)
        else:
            acc = np.sum(np.logical_or((error <= self.th), (error/gtflow_len <= self.th)), axis=1)
            acc = np.mean(acc)

        return acc

