import csv
import glob
import torch
import numpy as np
import nibabel as nib

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose


class FlyingThingsDataset(BaseDataset):
    """The Kidney Tumor Segmentation (KiTS) Challenge dataset (ref: https://kits19.grand-challenge.org/) for the 3D segmentation method.

    Args:
        data_split_csv (str): The path of the training and validation data split csv file.
        train_preprocessings (list of Box): The preprocessing techniques applied to the training data before applying the augmentation.
        valid_preprocessings (list of Box): The preprocessing techniques applied to the validation data before applying the augmentation.
        transforms (list of Box): The preprocessing techniques applied to the data.
        augments (list of Box): The augmentation techniques applied to the training data (default: None).
    """
    def __init__(self, data_split_csv, re_sample_size, train_preprocessings, valid_preprocessings, transforms, augments=None, **kwargs):
        super().__init__(**kwargs)
        self.data_split_csv = data_split_csv
        self.npoints = re_sample_size
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
        point1 = file['points1'].astype('float32')
        point2 = file['points2'].astype('float32')
        stereo1 = file['color1'].astype('float32')
        stereo2 = file['color2'] .astype('float32')
        flow = file['flow'].astype('float32')
        mask = file['valid_mask1']

        if self.type == 'train':
            n1 = point1.shape[0]
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            n2 = point2.shape[0]
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)

            point1 = point1[sample_idx1, :]
            point2 = point2[sample_idx2, :]
            stereo1 = stereo1[sample_idx1, :]
            stereo2 = stereo2[sample_idx2, :]
            flow = flow[sample_idx1, :]
            mask = mask[sample_idx1]
        else:
            point1 = point1[:self.npoints, :]
            point2 = point2[:self.npoints, :]
            stereo1 = stereo1[:self.npoints, :]
            stereo2 = stereo2[:self.npoints, :]
            flow = flow[:self.npoints, :]
            mask = mask[:self.npoints]

        point1_center = np.mean(point1, 0)
        point1 -= point1_center
        point2 -= point1_center

        # point1, point2 = self.transforms(point1, point2, dtypes=[torch.float, torch.float])
        # stereo1, stereo2 = self.transforms(point1, point2, dtypes=[torch.float, torch.float])
        # flow, mask = self.transforms(flow, mask, dtypes=[torch.float, torch.float])

        return {"point1": point1, "point2": point2, "stereo1": stereo1, "stereo2": stereo2, "flow": flow, "mask": mask}
