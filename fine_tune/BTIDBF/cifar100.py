import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import Dataset, Subset
from PIL import Image
import os
import numpy as np
import torch.nn.functional as F


class CIFAR(Dataset):
    def __init__(self, path, train, train_type=None, tf=None) -> None:
        super().__init__()
        datas = []
        labels = []
        dataset = CIFAR100(root=path, train=train, download=True)
        subset = Subset(dataset, np.random.choice(np.arange(len(dataset)), 5000, replace=False))
        for data in subset:
            datas.append(np.array(data[0]))
            labels.append(data[1])
        self.datas = datas 
        self.labels = labels
        self.tf = tf 
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index) :
        img, label = self.datas[index], self.labels[index]
        img = Image.fromarray(img)
        if not self.tf is None:
            img = self.tf(img)
        return img, label
    
