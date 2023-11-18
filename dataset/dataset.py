import sys
sys.path.append('./')
from torch.utils.data import Dataset
import numpy as np
import random
import cv2
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import torch
import torch.nn as nn
# from options import opt
import scipy.io
import re
from dataset.data_augmentation import split_train, split_valid, split_test

root = '/work3/s212645/Spectral_Reconstruction/clean/'
datanames = ['BGU/','ARAD/']
# datanames = ['ARAD/', 'ICVL_Norm/']
# datanames = ['CAVE/', 'ARAD/', 'ICVL_Norm/']

class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, arg=True):
        self.datanames = datanames
        self.crop_size = crop_size
        self.arg = arg
        self.rgb = []
        self.hyper = []
        self.trainset = []
        for name in self.datanames:
            self.trainset.append(split_train(data_root+ 'clean/' + name,valid_ratio=valid_ratio, test_ratio=test_ratio, imsize=self.crop_size))
        for trainset in self.trainset:
            self.rgb.extend(trainset[1])
            self.hyper.extend(trainset[0])
        self.length = len(self.hyper)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        bgr = self.rgb[idx]
        hyper = self.hyper[idx]
        bgr = np.transpose(bgr, [2, 0, 1])
        hyper = np.transpose(hyper, [2, 0, 1])
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.length

class TestDataset(Dataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, arg=True):
        self.datanames = datanames
        self.crop_size = crop_size
        self.arg = arg
        self.rgb = []
        self.hyper = []
        self.testset = []
        for name in self.datanames:
            self.testset.append(split_test(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio, imsize=self.crop_size))
        for testset in self.testset:
            self.rgb.extend(testset[1])
            self.hyper.extend(testset[0])
        self.length = len(self.hyper)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        bgr = self.rgb[idx]
        hyper = self.hyper[idx]
        bgr = np.transpose(bgr, [2, 0, 1])
        hyper = np.transpose(hyper, [2, 0, 1])
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.length
    
class ValidDataset(Dataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, arg=True):
        self.datanames = datanames
        self.crop_size = crop_size
        self.arg = arg
        self.rgb = []
        self.hyper = []
        self.testset = []
        for name in self.datanames:
            self.testset.append(split_valid(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio, imsize=self.crop_size))
        for testset in self.testset:
            self.rgb.extend(testset[1])
            self.hyper.extend(testset[0])
        self.length = len(self.hyper)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        bgr = self.rgb[idx]
        hyper = self.hyper[idx]
        bgr = np.transpose(bgr, [2, 0, 1])
        hyper = np.transpose(hyper, [2, 0, 1])
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.length

if __name__ == '__main__':
    trainset = TrainDataset(root, 128, 0.1, 0.1)
    print(trainset.__len__())
    train_loader = DataLoader(dataset=trainset, batch_size=64, shuffle=True, num_workers=2,
                                pin_memory=True, drop_last=True)
    
    for i, (image, label) in enumerate(train_loader):
        print(image.shape)
        break