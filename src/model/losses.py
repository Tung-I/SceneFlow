import torch
import torch.nn as nn


class FlowL2Error(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask=None):
        flow = target['flow']
        flow_pred = output['flow']
        minimum = torch.min(flow)
        maximum = torch.max(flow)
        flow_pred = (flow_pred - minimum) / (maximum - minimum)
        flow = (flow - minimum) / (maximum - minimum)

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
        minimum = torch.min(disp)
        maximum = torch.max(disp)
        disp_pred = (disp_pred - minimum) / (maximum - minimum)
        disp = (disp - minimum) / (maximum - minimum)

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
        minimum = torch.min(disp_next)
        maximum = torch.max(disp_next)
        disp_next_pred = (disp_next_pred - minimum) / (maximum - minimum)
        disp_next = (disp_next - minimum) / (maximum - minimum)

        if mask is not None:
            epe = torch.norm(disp_next_pred-disp_next, p=2, dim=1)
            epe = epe * mask
            epe = torch.sum(epe) / torch.sum(mask)
        else:
            epe = torch.norm(disp_next_pred-disp_next, p=2, dim=1).mean()

        return epe


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
        flow = target['flow']
        minimum = torch.min(flow)
        maximum = torch.max(flow)
        output = (output - minimum) / (maximum - minimum)
        flow = (flow - minimum) / (maximum - minimum)

        loss = torch.norm(output - flow, p=2, dim=1).mean()
        return loss
