import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from models.deepfm import DeepFM
from dataset.dataset import RecoDataset
from dataset.get_features import DataProcessor

# dp = DataProcessor()
# dp.write_dataset()
# load data
train_data = RecoDataset('./', train=True)
loader_train = DataLoader(train_data, batch_size=100,
                          sampler=sampler.SubsetRandomSampler(range(int(len(train_data)*0.8))))
val_data = RecoDataset('./', train=True)
loader_val = DataLoader(val_data, batch_size=100,
                        sampler=sampler.SubsetRandomSampler(range(int(len(train_data)*0.8), len(train_data))))

# feature_sizes = np.loadtxt('./data/feature_sizes.txt', delimiter=',')
# feature_sizes = [int(x) for x in feature_sizes]
# 对于实数型特征，feature_sizes中标识一个[1]，对于one-hot特征，标识[类别总数]
feature_sizes = [1] * 14
print(feature_sizes)

model = DeepFM(feature_sizes, use_cuda=True)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
model.fit(loader_train, loader_val, optimizer, epochs=20)
