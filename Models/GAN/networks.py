import sys
sys.path.append('./')
import torch.nn as nn
import torch
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.optim as optim
import torch.backends.cudnn as cudnn
from hsi_dataset import TrainDataset, ValidDataset
from utils import AverageMeter, record_loss, Loss_MRAE, Loss_RMSE, Loss_PSNR
from options import opt
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
import functools

class UnetGenerator(nn.Module):
    def __init__(self, num_input, num_output):
        super().__init__()
        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(4,4), stride=(2,2), padding=(1,1)))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.LeakyReLU(0.2))
            return layer
        def TransposeConv(input_nums, output_nums):
            layer = []
            layer.append(nn.ConvTranspose2d(input_nums, output_nums, kernel_size=(4,4), stride=(2,2), padding=(1,1)))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU(True))
            return layer
        
        self.embed = nn.Conv2d(num_input, 64, kernel_size=1, stride=1, padding=0)
        self.down1 = nn.Sequential(*Conv(64, 128))
        self.down2 = nn.Sequential(*Conv(128, 256))
        self.down3 = nn.Sequential(*Conv(256, 512))
        self.down4 = nn.Sequential(*Conv(512, 1024))
        self.up4 = nn.Sequential(*TransposeConv(1024, 512))
        self.up3 = nn.Sequential(*TransposeConv(1024, 256))
        self.up2 = nn.Sequential(*TransposeConv(512, 128))
        self.up1 = nn.Sequential(*TransposeConv(256, 64))
        self.out = nn.Sequential(
            nn.Conv2d(128, num_output, kernel_size=3, stride=1, padding=1), 
            nn.Tanh()
        )
        
    def forward(self, input):
        x = self.embed(input)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        y4 = self.up4(x4)
        y3 = self.up3(torch.concat([x3, y4], 1))
        y2 = self.up2(torch.concat([x2, y3], 1))
        y1 = self.up1(torch.concat([x1, y2], 1))
        output = self.out(torch.concat([x, y1], 1))
        return output
    
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return F.sigmoid(self.model(input))
    
class wgan_Discriminator(nn.Module):
    def __init__(self, input_nums):
        super().__init__()
        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU(True))
            return layer

        self.Net = nn.Sequential(
            *Conv(input_nums, 64),
            *Conv(64, 128),
            *Conv(128, 256),
            *Conv(256, 512),
            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(), 
        )

    def forward(self, input):
        output = self.Net(input)
        return output
    
class cgan_Discriminator(nn.Module):
    def __init__(self, input_nums):
        super().__init__()
        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU(True))
            return layer

        self.Net = nn.Sequential(
            *Conv(input_nums, 64),
            *Conv(64, 128),
            *Conv(128, 256),
            *Conv(256, 512),
            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(), 
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.Net(input)
        return output
    
class sn_Discriminator(nn.Module):
    def __init__(self, input_nums):
        super().__init__()
        self.conv1 = nn.Conv2d(input_nums, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv3.weight.data, 1.)
        self.Net = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(self.conv1),
            nn.ReLU(True),
            nn.utils.parametrizations.spectral_norm(self.conv2),
            nn.ReLU(True),
            nn.utils.parametrizations.spectral_norm(self.conv3),
            nn.ReLU(True),
        )
        self.conv = nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1,1), padding=0)
        
    def forward(self, input):
        output = self.Net(input)
        output = self.conv(output)
        return F.sigmoid(output)