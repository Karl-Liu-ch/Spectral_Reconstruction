import torch
import numpy as np
import os
from torchvision.io import read_image
import os
import matplotlib.pyplot as plt
import scipy.io
import h5py
import cv2
from spectral import *
from utils import Loss_Fid, Loss_SAM, Loss_SSIM
from hsi_dataset import TrainDataset, ValidDataset
from options import opt
from torch.utils.data import DataLoader

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
ROOT = '/work3/s212645/Spectral_Reconstruction'
print(device)

TRAIN_RGB = '/Train_RGB/'
TRAIN_SP = '/Valid_Spec/'
spec = '/work3/s212645/Spectral_Reconstruction/Train_Spec/ARAD_1K_0001.mat'
rgb = '/work3/s212645/Spectral_Reconstruction/Train_RGB/ARAD_1K_0001.jpg'
bgr = cv2.imread(rgb)
bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
bgr = np.float32(bgr)
bgr = (bgr-bgr.min())/(bgr.max()-bgr.min())
bgr = torch.tensor(bgr.transpose(2, 0, 1), dtype=torch.float32).to(device)
with h5py.File(spec, 'r') as mat:
    hyper =np.float32(np.array(mat['cube']))
hyper = np.transpose(hyper, [2, 1, 0])
view = ImageView()
# view.set_data(hyper, (22, 12, 3))
sam = Loss_SAM().to(device)
fid = Loss_Fid().to(device)
ssim = Loss_SSIM().to(device)
lossmin = 1e12
argi = 0
argj = 1
argk = 2


train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True, arg=True, stride=opt.stride)
print(f"Iteration per epoch: {len(train_data)}")
val_data = ValidDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True, arg=True, stride=64)
print("Validation set samples: ", len(val_data))

train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=True)
val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)

for i, (images, labels) in enumerate(train_loader):
    labels = labels.cuda()
    images = images.cuda()
    for i in range(0,10):
        for j in range(10,20):
            for k in range(20, 31):
                subhypers = []
                for b in range(labels.shape[0]):
                    label = labels[b].permute(1,2,0).cpu().numpy()
                    view.set_data(label, (k, j, i))
                    subhyper = torch.tensor(view.data_rgb.transpose(2, 0, 1), dtype=torch.float32).to(device)
                    subhyper = subhyper.unsqueeze(0)
                    subhypers.append(subhyper)
                subhypers = torch.concat(subhypers, dim=0)
                loss = sam(images, subhypers)
                if loss < lossmin:
                    lossmin = loss
                    argi = i
                    argj = j
                    argk = k
                    print(loss)
                
    print(argk, argj, argi)
    break