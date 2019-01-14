from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F
import torch.utils.data as Data

class loader(Data.Dataset):
    def __init__(self, list_file, transform=None, target_transform=None):
        self.list_file = open(list_file).readlines()
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        image_path, label = self.list_file[index].split(' ')
        with open(image_path, 'rb') as f:
            sample = Image.open(f)
            sample = sample.convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return sample,label

    def __len__(self):
        return len(self.list_file)