import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class RecoDataset(Dataset):
    """
    """ 
    def __init__(self, root, train=True):
        """
        Initialize file path and train/test mode.
        Inputs:
        - root: Path where the processed data file stored.
        - train: Train or test. Required.
        """
        self.root = root
        self.train = train

        if not self._check_exists:
            raise RuntimeError('Dataset not found.')

        if self.train:
            data = pd.read_csv(os.path.join(root, 'train.csv'))
            self.train_data = data.iloc[:, :-1].values
            self.target = data.iloc[:, -1].values
        else:
            data = pd.read_csv(os.path.join(root, 'test.csv'))
            self.test_data = data.iloc[:, :-1].values
    
    def __getitem__(self, idx):
        if self.train:
            dataI, targetI = self.train_data[idx, :], self.target[idx]
            Xv = torch.from_numpy(dataI.astype(np.float))
            Xi = torch.from_numpy(np.ones_like(dataI)).unsqueeze(-1)
            return Xi, Xv, targetI
        else:
            dataI = self.test_data.iloc[idx, :]
            Xv = torch.from_numpy(dataI.astype(np.float))
            Xi = torch.from_numpy(np.ones_like(dataI)).unsqueeze(-1)
            return Xi, Xv

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(self.root)