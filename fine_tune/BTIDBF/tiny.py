import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class TinyImageNet(Dataset):
    def __init__(self, path, train, train_type=None, tf=None):
        self.path = path
        self.tf = tf
        self.image_paths = []
        self.labels = []
        self.train = train
        # Read the image paths and labels from the dataset files
        self.read_dataset_files()

    def read_dataset_files(self):
        if self.train:
            # Read the file containing class labels
            with open(os.path.join(self.path, "wnids.txt"), "r") as f:
                classes = f.readlines()

            # Read the file containing image paths and labels
            with open(os.path.join(self.path, "train", "train_annotations.txt"), "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.split("\t")
                image_name = parts[0]
                label = classes.index(parts[1].strip())
                image_path = os.path.join(self.path, "train", "images", image_name)

                self.image_paths.append(image_path)
                self.labels.append(label)
        else:
            # Read the file containing image paths for the test dataset
            with open(os.path.join(self.path, "val", "val_annotations.txt"), "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.split("\t")
                image_name = parts[0]
                image_path = os.path.join(self.path, "val", "images", image_name)

                self.image_paths.append(image_path)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        # Load the image
        image = Image.open(image_path)
        
        # Apply tfations (if any)
        if self.tf is not None:
            image = self.tf(image)

        return image, label

    def __len__(self):
        return len(self.image_paths)