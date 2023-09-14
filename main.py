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
from Models.VAE.cvae import ConditionalVAE
from Models.VAE.cvaeSP import ConditionalVAESP
from hsi_dataset import TrainDataset, ValidDataset
from utils import AverageMeter, record_loss, Loss_MRAE, Loss_RMSE, Loss_PSNR

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--end_epoch", type=int, default=400, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--outf", type=str, default='/work3/s212645/Spectral_Reconstruction/checkpoint/cvaeSP/', help='path log files')
parser.add_argument("--data_root", type=str, default='/work3/s212645/Spectral_Reconstruction')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='1', help='path log files')
opt = parser.parse_args()

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

class Spectral_Reconstruction():
    def __init__(self) -> None:
        # self.model = ConditionalVAE(in_channels=31, condition_channels=3, latent_dim=1024, hidden_dims=[32, 64, 128, 256, 512, 1024], img_size=opt.patch_size).cuda()
        self.model = ConditionalVAESP(in_channels=31, condition_channels=3, latent_dim=1024, hidden_dims=[64, 128], img_size=128).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
        
        # iterations
        per_epoch_iteration = 1000
        self.total_iteration = per_epoch_iteration*opt.end_epoch
        self.iteration = 0
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.total_iteration, eta_min=1e-6)
        
    def load_dataset(self):
        # load dataset
        print("\nloading dataset ...")
        self.train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True, arg=True, stride=opt.stride)
        print(f"Iteration per epoch: {len(self.train_data)}")
        self.val_data = ValidDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True, arg=True, stride=64)
        print("Validation set samples: ", len(self.val_data))

    def train(self):
        self.load_dataset()
        record_mrae_loss = 1000
        while self.iteration<self.total_iteration:
            self.model.train()
            losses = AverageMeter()
            train_loader = DataLoader(dataset=self.train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2,
                                    pin_memory=True, drop_last=True)
            val_loader = DataLoader(dataset=self.val_data, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            for i, (images, labels) in enumerate(train_loader):
                labels = labels.cuda()
                images = images.cuda()
                images = Variable(images)
                labels = Variable(labels)
                lr = self.optimizer.param_groups[0]['lr']
                self.optimizer.zero_grad()
                # [output, input, mu, log_var] = self.model(labels, images)
                [output, input, mu0, log_var0, mu1, log_var1, mu2, log_var2] = self.model(labels, images)
                # Loss_dict = self.model.loss_function(output, input, mu, log_var)
                Loss_dict = self.model.loss_function(output, input, mu0, log_var0, mu1, log_var1, mu2, log_var2)
                loss = Loss_dict['loss']
                loss.backward()
                loss_mrae = criterion_mrae(output, input)
                losses.update(loss_mrae.data)
                self.optimizer.step()
                self.scheduler.step()
                self.iteration = self.iteration+1
                if self.iteration % 20 == 0:
                    print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f'
                        % (self.iteration, self.total_iteration, lr, losses.avg))
                if self.iteration % 1000 == 0:
                    mrae_loss, rmse_loss, psnr_loss = self.validate(val_loader)
                    print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}')
                    # Save model
                    if torch.abs(mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or self.iteration % 5000 == 0:
                        print(f'Saving to {opt.outf}')
                        self.save_checkpoint(opt.outf)
                        if mrae_loss < record_mrae_loss:
                            record_mrae_loss = mrae_loss
                    # print loss
                    print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                        "Test RMSE: %.9f, Test PSNR: %.9f " % (self.iteration, self.iteration//1000, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss))
                
    def validate(self, val_loader):
        self.model.eval()
        losses_mrae = AverageMeter()
        losses_rmse = AverageMeter()
        losses_psnr = AverageMeter()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            with torch.no_grad():
                # compute output
                output = self.model.sample(input)
                loss_mrae = criterion_mrae(output, target)
                loss_rmse = criterion_rmse(output, target)
                loss_psnr = criterion_psnr(output, target)
            # record loss
            losses_mrae.update(loss_mrae.data)
            losses_rmse.update(loss_rmse.data)
            losses_psnr.update(loss_psnr.data)
        return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg
    
    def test(self, test_loader):
        pass
    
    def save_checkpoint(self, root):
        epoch = self.iteration // 1000
        state = {
            'epoch': epoch,
            'iter': self.iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        torch.save(state, os.path.join(root, 'net_epoch_beta1.pth'))
        
    def load_checkpoint(self, root):
        checkpoint = torch.load(os.path.join(root, 'net_epoch_beta1.pth'))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iteration = checkpoint['iter']
        print("pretrained model loaded")
    
if __name__ == '__main__':
    spec = Spectral_Reconstruction()
    spec.load_checkpoint(opt.outf)
    spec.train()