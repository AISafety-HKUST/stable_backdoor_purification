import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, img_list):
        self.img_list = img_list        
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        return self.img_list[idx][0], self.img_list[idx][1]


class CustomDataset_v2(Dataset):
    def __init__(self, img_list, target, transform=None):
        self.img_list = img_list
        self.transform = transform
       
        if isinstance(target, list):
            self.target = target
        else:
            self.target = [int(target)] * len(img_list)
        
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.target[idx]
