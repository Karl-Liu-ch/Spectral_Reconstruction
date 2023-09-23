from __future__ import division

import torch
import torch.nn as nn
import logging
import numpy as np
import os
from torch import nn, optim
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import SpectralAngleMapper
from torchmetrics.image.fid import FrechetInceptionDistance
from spectral import *
import cv2

def bgr2rgb(bgr):
    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bgr = np.float32(bgr)
    rgb = (bgr-bgr.min())/(bgr.max()-bgr.min())
    return rgb

def reconRGB(labels):
    # view = ImageView()
    subhypers = []
    for b in range(labels.shape[0]):
        label = labels[b].permute(1,2,0).cpu().numpy()
        # view.set_data(label, (29, 19, 9))
        rgb = get_rgb(label, (29, 19, 9))
        subhyper = torch.tensor(rgb.transpose(2, 0, 1), dtype=torch.float32).cuda()
        subhyper = subhyper.unsqueeze(0)
        subhypers.append(subhyper)
    subhypers = torch.concat(subhypers, dim=0)
    return subhypers

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))

class Loss_Fid(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fid = FrechetInceptionDistance(feature=2048)
        
    def forward(self, outputs, label):
        outputs = outputs * 255
        label = label * 255
        outputs = outputs.type(torch.cuda.ByteTensor)
        label = label.type(torch.cuda.ByteTensor)
        self.fid.update(label, real=True)
        self.fid.update(outputs, real=False)
        return self.fid.compute()

class Loss_SAM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sam = SpectralAngleMapper()
    
    def forward(self, outputs, label):
        sam_score = self.sam(outputs, label)
        sam_score = torch.mean(sam_score.view(-1))
        return sam_score

class Loss_SSIM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        
    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        ssim_score = self.ssim(outputs, label)
        ssim_score = torch.mean(ssim_score.view(-1))
        return ssim_score

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        if label.all() == False:
            error = torch.abs(outputs - label) / (label + 1e-5)
        else:
            error = torch.abs(outputs - label) / label
        # mrae = torch.mean(error.reshape(-1))
        mrae = torch.mean(error.view(-1))
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.reshape(-1)))
        # rmse = torch.sqrt(torch.mean(sqrt_error.view(-1)))
        return rmse

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close
    
if __name__ == '__main__':
    criterion_ssim = Loss_SSIM()
    criterion_sam = Loss_SAM()
    criterion_mrae = Loss_MRAE()
    criterion_rmse = Loss_RMSE()
    criterion_psnr = Loss_PSNR()
    # target = torch.randn([64, 31, 128, 128]).cuda()
    # pred = target * 0.75
    # print(criterion_sam(pred, target))
    
    