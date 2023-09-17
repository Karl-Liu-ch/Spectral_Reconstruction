import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import datetime
import numpy as np
import random
import cv2
import h5py
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import os
from Models.VAE.cvae import train_cvae
from Models.VAE.cvaeSP import train_cvaeSP
from Models.VAE.cvaeSPCA import train_cvaeSPCA
from Models.GAN.cgan import CGAN
from Models.GAN.pix2pix import pix2pix
from hsi_dataset import TrainDataset, ValidDataset
from utils import AverageMeter, record_loss, Loss_MRAE, Loss_RMSE, Loss_PSNR
from options import opt

if __name__ == '__main__':
    # spec = train_cvae(opt)
    # spec = train_cvaeSP(opt)
    # spec = train_cvaeSPCA(opt)
    spec = CGAN(opt)
    # spec = pix2pix(opt)
    # spec.load_checkpoint()
    spec.train()