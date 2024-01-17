import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, img_list, label_list, transform=None):
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img = self.img_list[idx]
        label = self.label_list[idx]
        label = np.int64(label)
        if self.transform:
            img = self.transform(img)
        
        return img, label


class CustomDataset_v2(Dataset):
    def __init__(self, img_list, attack_target, transform=None):
        self.img_list = img_list
        self.transform = transform
        self.attack_target = int(attack_target)
        
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.attack_target

