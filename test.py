import sys
sys.path.append('./')
import torch.nn as nn
import torch
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.optim as optim
import torch.backends.cudnn as cudnn
# from hsi_dataset import TrainDataset, ValidDataset
from dataset.dataset import TrainDataset, ValidDataset, TestDataset
from utils import AverageMeter, record_loss, Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_Fid, Loss_SAM, Loss_SSIM, reconRGB, Loss_SID, SAM, SaveSpectral
from options import opt
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
from Models.GAN.SNcwgan import SNCWGAN
import functools
import numpy as np
import scipy.io

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
if opt.multigpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_psnrrgb = Loss_PSNR()
criterion_sam = Loss_SAM()
criterion_sid = Loss_SID()
criterion_fid = Loss_Fid().cuda()
criterion_ssim = Loss_SSIM().cuda()

def test(Model):
    root = '/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/'
    test_data = TestDataset(data_root=opt.data_root, crop_size=opt.patch_size, valid_ratio = 0.1, test_ratio=0.1)
    print("Validation set samples: ", len(test_data))
    test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    Model.G.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_psnrrgb = AverageMeter()
    losses_sam = AverageMeter()
    losses_sid = AverageMeter()
    losses_fid = AverageMeter()
    losses_ssim = AverageMeter()
    for i, (input, target) in enumerate(test_loader):
        input = input.cuda()
        target = target.cuda()
        z = torch.randn_like(input).cuda()
        z = torch.concat([z, input], dim=1)
        z = Variable(z)
        with torch.no_grad():
            # compute output
            output = Model.G(z)
            for j in range(output.shape[0]):
                mat = {}
                mat['cube'] = np.transpose(output[j,:,:,:].cpu().numpy(), [1,2,0])
                mat['rgb'] = np.transpose(input[j,:,:,:].cpu().numpy(), [1,2,0])
                scipy.io.savemat(root + str(i * output.shape[0] + j).zfill(3) + '.mat', mat)
                SaveSpectral(output[j,:,:,:], i * output.shape[0] + j)
                print(i * output.shape[0] + j, 'saved')
            loss_mrae = criterion_mrae(output, target)
            loss_rmse = criterion_rmse(output, target)
            loss_psnr = criterion_psnr(output, target)
            loss_sam = criterion_sam(output, target)
            loss_sid = criterion_sid(output, target)
            rgb = reconRGB(output)
            loss_fid = criterion_fid(rgb, input)
            loss_ssim = criterion_ssim(rgb, input)
            loss_psrnrgb = criterion_psnrrgb(rgb, input)
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
        losses_sam.update(loss_sam.data)
        losses_sid.update(loss_sid.data)
        losses_fid.update(loss_fid.data)
        losses_ssim.update(loss_ssim.data)
        losses_psnrrgb.update(loss_psrnrgb.data)
    criterion_sam.reset()
    criterion_psnr.reset()
    criterion_psnrrgb.reset()
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg, losses_sid.avg, losses_fid.avg, losses_ssim.avg, losses_psnrrgb.avg

if __name__ == '__main__':
    model = SNCWGAN(opt)
    model.load_checkpoint()
    mrad, rmse, psnr, sam, sid, fid, ssim, psnrrgb = test(model)
    print(mrad, rmse, psnr, sam, sid, fid, ssim, psnrrgb)
    # mrad:0.4860 rmse:0.0653 psnr:23.8645 sam:0.1173 sid:0.0006 fid:105.5602 ssim:0.7992 psnrrgb:16.7894