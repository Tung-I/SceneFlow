import torch
import torch.nn as nn
import numpy as np
from skimage.morphology import label


class FlowL2Error(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask=None):
        flow = target['flow']
        flow_pred = output['flow']
        # minimum = torch.min(flow)
        # maximum = torch.max(flow)
        # flow_pred = (flow_pred - minimum) / (maximum - minimum)
        # flow = (flow - minimum) / (maximum - minimum)

        if mask is not None:
            epe = torch.norm(flow_pred-flow, p=2, dim=1)
            epe = epe * mask
            epe = torch.sum(epe) / torch.sum(mask)
        else:
            epe = torch.norm(flow_pred-flow, p=2, dim=1).mean()

        return epe


class DispL2Error(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask=None):
        disp = target['disp']
        disp_pred = output['disp']
        # minimum = torch.min(disp)
        # maximum = torch.max(disp)
        # disp_pred = (disp_pred - minimum) / (maximum - minimum)
        # disp = (disp - minimum) / (maximum - minimum)

        if mask is not None:
            epe = torch.norm(disp_pred-disp, p=2, dim=1)
            epe = epe * mask
            epe = torch.sum(epe) / torch.sum(mask)
        else:
            epe = torch.norm(disp_pred-disp, p=2, dim=1).mean()

        return epe


class DispNextL2Error(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask=None):
        disp_next = target['disp_next']
        disp_next_pred = output['disp_next']
        # minimum = torch.min(disp_next)
        # maximum = torch.max(disp_next)
        # disp_next_pred = (disp_next_pred - minimum) / (maximum - minimum)
        # disp_next = (disp_next - minimum) / (maximum - minimum)
        
        if mask is not None:
            epe = torch.norm(disp_next_pred-disp_next, p=2, dim=1)
            epe = epe * mask
            epe = torch.sum(epe) / torch.sum(mask)
        else:
            epe = torch.norm(disp_next_pred-disp_next, p=2, dim=1).mean()

        return epe

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

