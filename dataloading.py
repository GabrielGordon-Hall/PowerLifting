import numpy as np
import torch
from torch.utils.data import Dataset


class CSVDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        features = self.dataframe.iloc[idx, :-1].values
        features = features.astype('float').reshape(1, 10)
        targets = self.dataframe.iloc[idx, -1]
        sample = (features, targets)
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays to Tensors"""

    def __call__(self, sample):
        features, targets = sample
        return (torch.from_numpy(features),
                torch.from_numpy(np.array(targets, ndmin=1, copy=False)))
