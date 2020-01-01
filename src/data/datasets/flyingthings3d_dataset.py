import csv
import glob
import torch
import numpy as np

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose


class FlyingThings3DDataset(BaseDataset):
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
        rgb_r = file['rgb_r'].astype('float32')
        rgb_next_l = file['rgb_next_l'].astype('float32')
        rgb_next_r = file['rgb_next_r'].astype('float32')
        disparity = file['disparity'].astype('float32')
        disparity_next = file['disparity_next'].astype('float32')
        flow = file['flow'].astype('float32')



        rgb_l, rgb_r = self.transforms(rgb_l, rgb_r, dtypes=[torch.float, torch.float])
        rgb_next_l, rgb_next_r = self.transforms(rgb_next_l, rgb_next_r, dtypes=[torch.float, torch.float])
        disparity, disparity_next = self.transforms(disparity, disparity_next, dtypes=[torch.float, torch.float])
        flow = self.transforms(flow, dtypes=[torch.float])

        return {"rgb_l": rgb_l, "rgb_r": rgb_r, "rgb_next_l": rgb_next_l, "rgb_next_r": rgb_next_r, "disparity": disparity, "disparity_next": disparity_next, "flow": flow}
