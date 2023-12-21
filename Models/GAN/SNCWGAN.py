import sys
sys.path.append('./')
import torch.nn as nn
import torch
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.optim as optim
import torch.backends.cudnn as cudnn
from options import opt
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
import functools
from Models.GAN.networks import *
from Models.Transformer.DTN import DTN
import numpy as np
from Models.GAN.Basemodel import BaseModel, criterion_mrae, AverageMeter, SAM
from Models.GAN.Utils import Log_loss, Itself_loss
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
if opt.multigpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NONOISE = opt.nonoise

class SNCWGAN(BaseModel):
    def __init__(self, opt, multiGPU=False):
        super().__init__(opt, multiGPU)
        self.lamda = 100
        self.lambdasam = 100
        self.root = '/work3/s212645/Spectral_Reconstruction/checkpoint/SNCWGAN/' + self.opt.G + '/'
        self.nonoise = False
        self.init_Net()
        self.init_metrics()
    
    def init_Net(self):
        if self.opt.G == 'DTN':
            self.G = DTN(in_dim=3, 
                    out_dim=31,
                    img_size=[128, 128], 
                    window_size=8, 
                    n_block=[2,2,2,2], 
                    bottleblock = 4)
            self.nonoise = True
            print('DTN, No noise')
        elif self.opt.G == 'res':
            self.G = ResnetGenerator(6, 31)
            self.nonoise = False
            print('ResnetGenerator, with noise')
        elif self.opt.G == 'dense':
            self.G = DensenetGenerator(inchannels = 6, 
                 outchannels = 31, 
                 num_init_features = 64, 
                 block_config = (6, 12, 24, 16),
                 bn_size = 4, 
                 growth_rate = 32, 
                 center_layer = 6
                 )
            self.nonoise = False
            self.lambdasam = 0
            print('DensenetGenerator, with noise')
        self.D = SN_Discriminator(34)
        super().init_Net()
        
    def train(self):
        super().train()
        record_mrae_loss = 1000
        while self.epoch<self.end_epoch:
            self.G.train()
            self.D.train()
            losses = AverageMeter()
            train_loader = DataLoader(dataset=self.train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=2,
                                    pin_memory=True, drop_last=True)
            val_loader = DataLoader(dataset=self.val_data, batch_size=self.opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            for i, (images, labels) in enumerate(train_loader):
                labels = labels.cuda()
                images = images.cuda()
                images = Variable(images)
                labels = Variable(labels)
                if self.nonoise:
                    z = images
                else:
                    z = torch.randn_like(images).cuda()
                    z = torch.concat([z, images], dim=1)
                    z = Variable(z)
                realAB = torch.concat([images, labels], dim=1)
                D_real = self.D(realAB)
                x_fake = self.G(z)
                fakeAB = torch.concat([images, x_fake],dim=1)
                
                # train D
                for p in self.D.parameters():
                    p.requires_grad = True
                self.optimD.zero_grad()
                D_real = self.D(realAB)
                loss_real = -D_real.mean(0).view(1)
                loss_real.backward()
                D_fake = self.D(fakeAB.detach())
                loss_fake = D_fake.mean(0).view(1)
                loss_fake.backward()
                self.optimD.step()
                
                # train G
                self.optimG.zero_grad()
                lrG = self.optimG.param_groups[0]['lr']
                for p in self.D.parameters():
                    p.requires_grad = False
                pred_fake = self.D(fakeAB)
                loss_G = -pred_fake.mean(0).view(1)
                lossl1 = self.lossl1(x_fake, labels) * self.lamda
                losssam = SAM(x_fake, labels) * self.lambdasam
                loss_G += lossl1 + losssam
                # train the generator
                loss_G.backward()
                self.optimG.step()
                
                loss_mrae = criterion_mrae(x_fake, labels)
                losses.update(loss_mrae.data)
                self.iteration = self.iteration+1
                if self.iteration % 20 == 0:
                    print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f'
                        % (self.iteration, self.total_iteration, lrG, losses.avg))
            # validation
            mrae_loss, rmse_loss, psnr_loss, sam_loss, sid_loss = self.validate(val_loader)
            print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}, SAM: {sam_loss}, SID: {sid_loss}')
            # Save model
            # if torch.abs(mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or self.iteration % 5000 == 0:
            print(f'Saving to {self.root}')
            self.save_checkpoint()
            if mrae_loss < record_mrae_loss:
                record_mrae_loss = mrae_loss
                self.save_checkpoint(True)
            # print loss
            print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                "Test RMSE: %.9f, Test PSNR: %.9f, SAM: %.9f, SID: %.9f " % (self.iteration, 
                                                                self.epoch, lrG, 
                                                                losses.avg, mrae_loss, rmse_loss, psnr_loss, sam_loss, sid_loss))
            self.epoch += 1
            self.schedulerD.step()
            self.schedulerG.step()
            
if __name__ == '__main__':
    spec = SNCWGAN(opt, multiGPU=opt.multigpu)
    if opt.loadmodel:
        try:
            spec.load_checkpoint()
        except:
            print('pretrained model loading failed')
    spec.train()