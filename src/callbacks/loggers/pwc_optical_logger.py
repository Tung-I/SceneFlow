import torch
import random
import numpy as np
from torchvision.utils import make_grid

from .base_logger import BaseLogger
from lib.visualize import flow_visualize_2d


class PWCOpticalLogger(BaseLogger):
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

        # b, c, h, w = train_image.shape

        # label_train = np.zeros((1, 3, h, w))
        # pred_train = np.zeros((1, 3, h, w))
        # label_valid = np.zeros((1, 3, h, w))
        # pred_valid = np.zeros((1, 3, h, w))

        # label_train[0] = flow_visualize_2d(train_label[0])
        # pred_train[0] = flow_visualize_2d(train_pred[0])
        # label_valid[0] = flow_visualize_2d(valid_label[0])
        # pred_valid[0] = flow_visualize_2d(valid_pred[0])

        # image_train = torch.FloatTensor(train_image[0])
        # image_valid = torch.FloatTensor(valid_image[0])
        # label_train = torch.FloatTensor(label_train)
        # pred_train = torch.FloatTensor(pred_train)
        # label_valid = torch.FloatTensor(label_valid)
        # pred_valid = torch.FloatTensor(pred_valid)


        image_train = train_batch['rgb_l'].detach().cpu()
        image_next_train = train_batch['rgb_next_l'].detach().cpu()

        label_train = train_batch['flow'].detach().cpu()
        pred_train = train_output.detach().cpu()

        image_valid = valid_batch['rgb_l'].detach().cpu()
        image_next_valid = valid_batch['rgb_next_l'].detach().cpu()

        label_valid = valid_batch['flow'].detach().cpu()
        pred_valid = valid_output.detach().cpu()

        b, c, h, w = image_train.size()
        pad_train = torch.zeros((b, 1, h, w))
        b, c, h, w = image_valid.size()
        pad_valid = torch.zeros((b, 1, h, w))

        label_train = torch.cat((label_train, pad_train), 1)
        pred_train = torch.cat((pred_train, pad_train), 1)
        label_valid = torch.cat((label_valid, pad_valid), 1)
        pred_valid = torch.cat((pred_valid, pad_valid), 1)


        train_img = make_grid(image_train, nrow=1, normalize=True, scale_each=True, pad_value=1)
        train_img_next = make_grid(image_next_train, nrow=1, normalize=True, scale_each=True, pad_value=1)
        train_label = make_grid(label_train, nrow=1, normalize=True, scale_each=True, pad_value=1)
        train_pred = make_grid(pred_train, nrow=1, normalize=True, scale_each=True, pad_value=1)

        valid_img = make_grid(image_valid, nrow=1, normalize=True, scale_each=True, pad_value=1)
        valid_img_next = make_grid(image_next_valid, nrow=1, normalize=True, scale_each=True, pad_value=1)
        valid_label = make_grid(label_valid, nrow=1, normalize=True, scale_each=True, pad_value=1)
        valid_pred = make_grid(pred_valid, nrow=1, normalize=True, scale_each=True, pad_value=1)

        train_grid = torch.cat((train_img, train_img_next, train_label, train_pred), dim=-1)
        valid_grid = torch.cat((valid_img, valid_img_next, valid_label, valid_pred), dim=-1)
        self.writer.add_image('train', train_grid, epoch)
        self.writer.add_image('valid', valid_grid, epoch)
