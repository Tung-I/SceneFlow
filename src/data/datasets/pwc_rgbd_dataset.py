import csv
import glob
import torch
import numpy as np

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose


class PWCRGBDDataset(BaseDataset):
    """The Kidney Tumor Segmentation (KiTS) Challenge dataset (ref: https://kits19.grand-challenge.org/) for the 3D segmentation method.

    Args:
        data_split_csv (str): The path of the training and validation data split csv file.
        train_preprocessings (list of Box): The preprocessing techniques applied to the training data before applying the augmentation.
        valid_preprocessings (list of Box): The preprocessing techniques applied to the validation data before applying the augmentation.
        transforms (list of Box): The preprocessing techniques applied to the data.
        augments (list of Box): The augmentation techniques applied to the training data (default: None).
    """
    def __init__(self, data_split_csv, train_preprocessings, valid_preprocessings, transforms, augments=None, **kwargs):
        super().__init__(**kwargs)
        self.data_split_csv = data_split_csv
        self.train_preprocessings = compose(train_preprocessings)
        self.valid_preprocessings = compose(valid_preprocessings)
        self.transforms = compose(transforms)
        self.augments = compose(augments)
        self.data_paths = []

        # Collect the data paths according to the dataset split csv.
        with open(self.data_split_csv, "r") as f:
            type_ = 'Training' if self.type == 'train' else 'Validation'
            rows = csv.reader(f)
            for file_name, split_type in rows:
                if split_type == type_:
                    data_path = self.data_dir / f'{file_name}'
                    self.data_paths.append(data_path)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_path = self.data_paths[index]

        file = np.load(data_path)

        rgb_l = file['rgb_l'].astype('float32')
        # rgb_r = file['rgb_r'].astype('float32')
        rgb_next_l = file['rgb_next_l'].astype('float32')
        # rgb_next_r = file['rgb_next_r'].astype('float32')
        disparity = file['disparity'].astype('float32')
        disparity_next = file['disparity_next'].astype('float32')
        flow = file['flow'].astype('float32')

        disparity = np.expand_dims(disparity, axis=2) # (H, W, C)
        disparity_next = np.expand_dims(disparity_next, axis=2)

        if self.type == 'train':
            rgb_l, rgb_next_l, disparity, disparity_next, flow = self.train_preprocessings(rgb_l, rgb_next_l, disparity, disparity_next, flow)

        rgb_l, rgb_next_l = self.transforms(rgb_l, rgb_next_l, dtypes=[torch.float, torch.float])
        disparity, disparity_next = self.transforms(disparity, disparity_next, dtypes=[torch.float, torch.float])
        flow = self.transforms(flow, dtypes=[torch.float])

        rgb_l = rgb_l.permute(2, 0, 1).contiguous()
        rgb_next_l = rgb_next_l.permute(2, 0, 1).contiguous()
        disparity = disparity.permute(2, 0, 1).contiguous()
        disparity_next = disparity_next.permute(2, 0, 1).contiguous()
        flow = flow.permute(2, 0, 1).contiguous()


        # return {"rgb_l": rgb_l, "rgb_r": rgb_r, "rgb_next_l": rgb_next_l, "rgb_next_r": rgb_next_r, "disparity": disparity, "disparity_next": disparity_next, "flow": flow}
        return {"rgb_l": rgb_l, "rgb_next_l": rgb_next_l, 'disparity': disparity, 'disparity_next': disparity_next, "flow": flow}