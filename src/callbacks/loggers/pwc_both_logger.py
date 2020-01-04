import torch
import random
import numpy as np
from torchvision.utils import make_grid

from .base_logger import BaseLogger
from lib.visualize import flow_visualize_2d


class PWCBothLogger(BaseLogger):
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

        # train_image = train_batch['rgb_l'].detach().cpu().numpy()
        # train_label = train_batch['flow'].detach().cpu().numpy()
        # train_pred = train_output.detach().cpu().numpy()
        # valid_image = valid_batch['rgb_l'].detach().cpu().numpy()
        # valid_label = valid_batch['flow'].detach().cpu().numpy()
        # valid_pred = valid_output.detach().cpu().numpy()


        flow_train = train_batch['flow'].detach().cpu()
        disp_train = train_batch['disparity'].detach().cpu()
        disp_next_train = train_batch['disparity_next'].detach().cpu()
        flow_pred_train = train_output['flow'].detach().cpu()
        disp_pred_train = train_output['disp'].detach().cpu()
        disp_next_pred_train = train_output['disp_next'].detach().cpu()

        flow_valid = valid_batch['flow'].detach().cpu()
        disp_valid = valid_batch['disparity'].detach().cpu()
        disp_next_valid = valid_batch['disparity_next'].detach().cpu()
        flow_pred_valid = valid_output['flow'].detach().cpu()
        disp_pred_valid = valid_output['disp'].detach().cpu()
        disp_next_pred_valid = valid_output['disp_next'].detach().cpu()


        b, c, h, w = flow_train.size()
        pad_train = torch.zeros((b, 1, h, w))
        b, c, h, w = flow_valid.size()
        pad_valid = torch.zeros((b, 1, h, w))

        flow_train = torch.cat((flow_train, pad_train), 1)
        flow_pred_train = torch.cat((flow_pred_train, pad_train), 1)
        flow_valid = torch.cat((flow_valid, pad_valid), 1)
        flow_pred_valid = torch.cat((flow_pred_valid, pad_valid), 1)


        flow_train = make_grid(flow_train, nrow=1, normalize=True, scale_each=True, pad_value=1)
        disp_train = make_grid(disp_train, nrow=1, normalize=True, scale_each=True, pad_value=1)
        disp_next_train = make_grid(disp_next_train, nrow=1, normalize=True, scale_each=True, pad_value=1)
        flow_pred_train = make_grid(flow_pred_train, nrow=1, normalize=True, scale_each=True, pad_value=1)
        disp_pred_train = make_grid(disp_pred_train, nrow=1, normalize=True, scale_each=True, pad_value=1)
        disp_next_pred_train = make_grid(disp_next_pred_train, nrow=1, normalize=True, scale_each=True, pad_value=1)

        flow_valid = make_grid(flow_valid, nrow=1, normalize=True, scale_each=True, pad_value=1)
        disp_valid = make_grid(disp_valid, nrow=1, normalize=True, scale_each=True, pad_value=1)
        disp_next_valid = make_grid(disp_next_valid, nrow=1, normalize=True, scale_each=True, pad_value=1)
        flow_pred_valid = make_grid(flow_pred_valid, nrow=1, normalize=True, scale_each=True, pad_value=1)
        disp_pred_valid = make_grid(disp_pred_valid, nrow=1, normalize=True, scale_each=True, pad_value=1)
        disp_next_pred_valid = make_grid(disp_next_pred_valid, nrow=1, normalize=True, scale_each=True, pad_value=1)

        train_grid = torch.cat((flow_train, disp_train, disp_next_train, flow_pred_train, disp_pred_train, disp_next_pred_train), dim=-1)
        valid_grid = torch.cat((flow_valid, disp_valid, disp_next_valid, flow_pred_valid, disp_pred_valid, disp_next_pred_valid), dim=-1)
        self.writer.add_image('train', train_grid, epoch)
        self.writer.add_image('valid', valid_grid, epoch)
