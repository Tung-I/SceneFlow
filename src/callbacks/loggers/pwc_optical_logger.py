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
        
        # train_positive_list = np.unique(np.where(train_batch['flow'].cpu().numpy() != 0)[2])
        # if len(train_positive_list) == 0:
        #     train_slice_id = random.randint(0, train_output.size(2)-1)
        # else:
        #     train_slice_id = random.choice(train_positive_list)

        # valid_positive_list = np.unique(np.where(valid_batch['label'].cpu().numpy() != 0)[2])
        # if len(valid_positive_list) == 0:
        #     valid_slice_id = random.randint(0, valid_output.size(2)-1)
        # else:
        #     valid_slice_id = random.choice(valid_positive_list)

        train_image = train_batch['rgb_l'].detach().cpu().numpy()
        train_label = train_batch['flow'].detach().cpu().numpy()
        train_pred = train_output.detach().cpu().numpy()
        valid_image = valid_batch['rgb_l'].detach().cpu().numpy()
        valid_label = valid_batch['flow'].detach().cpu().numpy()
        valid_pred = valid_output.detach().cpu().numpy()

        # train_image = np.transpose(train_image, (0, 3, 1, 2))
        # train_label = np.transpose(train_label, (0, 3, 1, 2))
        # valid_image = np.transpose(valid_image, (0, 3, 1, 2))
        # valid_label = np.transpose(valid_label, (0, 3, 1, 2))

        b, c, h, w = train_image.shape
        # print(train_image.shape)
        # print(train_label.shape)
        # print(train_pred.shape)
        label_train = np.zeros((1, 3, h, w))
        pred_train = np.zeros((1, 3, h, w))
        label_valid = np.zeros((1, 3, h, w))
        pred_valid = np.zeros((1, 3, h, w))

        label_train[0] = flow_visualize_2d(train_label[0])
        pred_train[0] = flow_visualize_2d(train_pred[0])
        label_valid[0] = flow_visualize_2d(valid_label[0])
        pred_valid[0] = flow_visualize_2d(valid_pred[0])

        image_train = torch.FloatTensor(train_image[0])
        image_valid = torch.FloatTensor(valid_image[0])
        label_train = torch.FloatTensor(label_train)
        pred_train = torch.FloatTensor(pred_train)
        label_valid = torch.FloatTensor(label_valid)
        pred_valid = torch.FloatTensor(pred_valid)
        # for i in range(train_image.shape[0]):
        #     label_train[i] = flow_visualize_2d(train_label[i])
        #     pred_train[i] = flow_visualize_2d(train_pred[i])
        # for i in range(valid_image.shape[0]):
        #     label_valid[i] = flow_visualize_2d(valid_label[i])
        #     pred_valid[i] = flow_visualize_2d(valid_pred[i])


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
