import torch
import random
import numpy as np
from torchvision.utils import make_grid

from .base_logger import BaseLogger
from lib.visualize import flow_visualize_2d


class PWCSceneLogger(BaseLogger):
    """The KiTS logger for the segmentation task.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_images(self, epoch, train_batch, train_output, valid_batch, valid_output):
        """Plot the visualization results.
        Args:
            epoch (int): The number of trained epochs.
            train_batch (dict): The training batch.
            train_output (torch.Tensor): The training output.
            valid_batch (dict): The validation batch.
            valid_output (torch.Tensor): The validation output.
        """

        flow_train = train_batch['flow'].detach().cpu()
        optical_flow_train = flow_train[:, :2, :, :]
        disp_change_train = flow_train[:, 2:3, :, :]
        flow_pred_train = train_output['flow'].detach().cpu()
        optical_flow_pred_train = flow_pred_train[:, :2, :, :]
        disp_change_pred_train = flow_pred_train[:, 2:3, :, :]

        flow_valid = valid_batch['flow'].detach().cpu()
        optical_flow_valid = flow_valid[:, :2, :, :]
        disp_change_valid = flow_valid[:, 2:3, :, :]
        flow_pred_valid = valid_output['flow'].detach().cpu()
        optical_flow_pred_valid = flow_pred_valid[:, :2, :, :]
        disp_change_pred_valid = flow_pred_valid[:, 2:3, :, :]

        b, c, h, w = flow_train.size()
        pad_train = torch.zeros((b, 1, h, w))
        b, c, h, w = flow_valid.size()
        pad_valid = torch.zeros((b, 1, h, w))

        optical_flow_train = torch.cat([optical_flow_train, pad_train], 1)
        optical_flow_pred_train = torch.cat([optical_flow_pred_train, pad_train], 1)
        optical_flow_valid = torch.cat([optical_flow_valid, pad_valid], 1)
        optical_flow_pred_valid = torch.cat([optical_flow_pred_valid, pad_valid], 1)


        # print(image_train.size())
        # print(label_train.size())
        # print(pred_train.size())


        optical_flow_train = make_grid(optical_flow_train, nrow=1, normalize=True, scale_each=True, pad_value=1)
        optical_flow_pred_train = make_grid(optical_flow_pred_train, nrow=1, normalize=True, scale_each=True, pad_value=1)
        disp_change_train = make_grid(disp_change_train, nrow=1, normalize=True, scale_each=True, pad_value=1)
        disp_change_pred_train = make_grid(disp_change_pred_train, nrow=1, normalize=True, scale_each=True, pad_value=1)

        optical_flow_valid = make_grid(optical_flow_valid, nrow=1, normalize=True, scale_each=True, pad_value=1)
        optical_flow_pred_valid = make_grid(optical_flow_pred_valid, nrow=1, normalize=True, scale_each=True, pad_value=1)
        disp_change_valid = make_grid(disp_change_valid, nrow=1, normalize=True, scale_each=True, pad_value=1)
        disp_change_pred_valid = make_grid(disp_change_pred_valid, nrow=1, normalize=True, scale_each=True, pad_value=1)

        train_grid = torch.cat((optical_flow_train, optical_flow_pred_train, disp_change_train, disp_change_pred_train), dim=-1)
        valid_grid = torch.cat((optical_flow_valid, optical_flow_pred_valid, disp_change_valid, disp_change_pred_valid), dim=-1)
        self.writer.add_image('train', train_grid, epoch)
        self.writer.add_image('valid', valid_grid, epoch)