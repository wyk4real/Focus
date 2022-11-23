import os
import torch
import cv2 as cv
import numpy as np
import tifffile as tiff
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def binarization(masks: torch.Tensor, threshold=0.2):
    masks[masks > threshold] = 1
    masks[masks < 1] = 0
    return masks


def preprocessing(Legs: torch.Tensor, Metals: torch.Tensor, data_augmentation=True):
    B, C, H, W = Legs.shape
    scales = []
    images = []
    masks = []
    targets = []
    for b in range(B):
        Leg = Legs[b].numpy().squeeze(0)
        Metal = binarization(Metals[b]).numpy().squeeze(0)
        if data_augmentation:
            R1 = np.random.random()
            if R1 < 0.5:
                Leg = cv.flip(Leg, flipCode=0)
            R2 = np.random.random()
            if R2 < 0.5:
                Leg = cv.flip(Leg, flipCode=1)
            R3 = np.random.random()
            if R3 < 0.5:
                noise = np.random.normal(loc=0, scale=R3 * 0.1, size=(512, 512)).astype(np.float32)
                Leg = Leg + noise
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
            R4 = np.random.random()
            if R4 < 0.5:
                Metal = cv.flip(Metal, flipCode=0)
            R5 = np.random.random()
            if R5 < 0.5:
                Metal = cv.flip(Metal, flipCode=1)
            R6 = np.random.random()
            if R6 < 0.5:
                Metal = cv.morphologyEx(Metal, cv.MORPH_OPEN, kernel, iterations=1)
            R7 = np.random.random()
            if R7 < 0.5:
                Metal = cv.morphologyEx(Metal, cv.MORPH_CLOSE, kernel, iterations=1)

        scale = (Leg * (1 - Metal)).max()
        image = (Leg / scale) * (1 - Metal)
        mask = Metal
        target = Leg / scale

        scales.append(torch.Tensor([[[scale]]]))
        images.append(transforms.ToTensor()(image))
        masks.append(transforms.ToTensor()(mask))
        targets.append(transforms.ToTensor()(target))

    return torch.stack([scale for scale in scales]), torch.stack([image for image in images]), \
           torch.stack([mask for mask in masks]), torch.stack([target for target in targets])


class MyData(Dataset):
    def __init__(self, Data_path: str, Legs_dir: str, Metal_dir: str):
        super(MyData, self).__init__()
        self.Data_path = Data_path
        self.Legs_dir = Legs_dir
        self.Metal_dir = Metal_dir
        self.Legs_list = os.listdir(os.path.join(self.Data_path, self.Legs_dir))
        self.Metal_list = os.listdir(os.path.join(self.Data_path, self.Metal_dir))

        self.Legs_list.sort()
        self.Metal_list.sort()

    def __getitem__(self, index):
        Legs_name = self.Legs_list[index]
        Metal_name = self.Metal_list[index]
        Leg = tiff.imread(os.path.join(self.Data_path, self.Legs_dir, Legs_name))
        Metal = tiff.imread(os.path.join(self.Data_path, self.Metal_dir, Metal_name))
        return transforms.ToTensor()(Leg), transforms.ToTensor()(Metal)

    def __len__(self):
        return len(self.Legs_list)
