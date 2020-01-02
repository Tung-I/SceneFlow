import torch
import random
import numpy as np
from torchvision.utils import make_grid

from .base_logger import BaseLogger
from lib.visualize import flow_visualize_2d


class PWCDispchangeLogger(BaseLogger):
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

        image_train = train_batch['rgb_l']
        label_train = train_batch['dispchange']
        pred_train = train_output
        image_valid = valid_batch['rgb_l']
        label_valid = valid_batch['dispchange']
        pred_valid = valid_output

        # print(image_train.size())
        # print(label_train.size())
        # print(pred_train.size())


        train_img = make_grid(image_train, nrow=1, normalize=True, scale_each=True, pad_value=1)
        train_label = make_grid(label_train, nrow=1, normalize=True, scale_each=True, pad_value=1)
        train_pred = make_grid(pred_train, nrow=1, normalize=True, scale_each=True, pad_value=1)

        valid_img = make_grid(image_valid, nrow=1, normalize=True, scale_each=True, pad_value=1)
        valid_label = make_grid(label_valid, nrow=1, normalize=True, scale_each=True, pad_value=1)
        valid_pred = make_grid(pred_valid, nrow=1, normalize=True, scale_each=True, pad_value=1)

        train_grid = torch.cat((train_img, train_label, train_pred), dim=-1)
        valid_grid = torch.cat((valid_img, valid_label, valid_pred), dim=-1)
        self.writer.add_image('train', train_grid, epoch)
        self.writer.add_image('valid', valid_grid, epoch)
