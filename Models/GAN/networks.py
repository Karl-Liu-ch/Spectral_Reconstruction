import sys
sys.path.append('./')
import torch.nn as nn
import torch
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import AverageMeter, record_loss, Loss_MRAE, Loss_RMSE, Loss_PSNR
from options import opt
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
import functools
from Models.GAN.HSCNN_Plus import HSCNN_Plus
from Models.GAN.SpectralNormalization import *

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs = 6, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks = 0):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, n_blocks=n_blocks)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, n_blocks=n_blocks)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, n_blocks=n_blocks)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, n_blocks=n_blocks)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, n_blocks=n_blocks)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks = 0):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            for i in range(n_blocks):
                down += [ResnetBlock(inner_nc, 'reflect')]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            for i in range(n_blocks):
                down += [ResnetBlock(inner_nc, 'reflect')]
            up = [uprelu, upconv, upnorm]
            for i in range(n_blocks):
                up += [ResnetBlock(outer_nc, 'reflect')]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            for i in range(n_blocks):
                down += [ResnetBlock(inner_nc, 'reflect')]
            up = [uprelu, upconv, upnorm]
            for i in range(n_blocks):
                up += [ResnetBlock(outer_nc, 'reflect')]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

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
        return self.model(input)
    
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
    
class SN_Discriminator(nn.Module):
    def __init__(self, input_nums):
        super(SN_Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_nums, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv4 = nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1,1), padding=0)
        
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv3.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv4.weight.data, 1.)
        self.Net = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(self.conv1),
            nn.ReLU(True),
            nn.utils.parametrizations.spectral_norm(self.conv2),
            nn.ReLU(True),
            nn.utils.parametrizations.spectral_norm(self.conv3),
            nn.ReLU(True),
            nn.utils.parametrizations.spectral_norm(self.conv4),
            nn.ReLU(True),
        )
        
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(), 
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.Net(input)
        output = self.out(output)
        return output
    
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type = 'reflect', norm_layer = nn.BatchNorm2d, use_dropout = False, use_bias = False):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return SNConv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)

def shift_back(inputs,step=2):          # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]

class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = SNConv2d(dim, dim_head * heads, bias=False)
        self.to_k = SNConv2d(dim, dim_head * heads, bias=False)
        self.to_v = SNConv2d(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = SNConv2d(dim_head * heads, dim, bias=True)
        self.dim = dim

    def forward(self, x_in):
        b, c, h, w = x_in.shape
        x = x_in.clone()
        q_inp = self.to_q(x).permute(0, 2, 3, 1).reshape(b,h*w,c)
        k_inp = self.to_k(x).permute(0, 2, 3, 1).reshape(b,h*w,c)
        v_inp = self.to_v(x).permute(0, 2, 3, 1).reshape(b,h*w,c)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.reshape(b, self.num_heads * self.dim_head, h, w)
        out_c = self.proj(x)
        out = out_c + x_in

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            SNConv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            SNConv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            SNConv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x)
        return out

class SN_MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim)
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        # x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        # out = x.permute(0, 3, 1, 2)
        return x

class SN_NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(SN_NLayerDiscriminator, self).__init__()
        use_bias = True
        kw = 4
        padw = 1
        sequence = [SNConv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                SNConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            SNConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
        ]

        sequence += [SNConv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
class SNPixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, n_block = 6):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(SNPixelDiscriminator, self).__init__()

        self.net = [
            SNConv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            SNConv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=True)]
        for i in range(n_block):
            self.net += [SNResnetBlock(ndf * 2, padding_type='reflect', use_bias=True, use_dropout=False),
            nn.ReLU(True)]
        
        self.net += [SNConv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=True)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

class SNResnetDiscriminator(nn.Module):
    def __init__(self, input_nums, n_layer=6):
        super().__init__()
        def ResBlock(input_nums, output_nums, n_blocks = 1):
            layer = []
            layer.append(SNConv2d(input_nums, output_nums, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
            layer.append(nn.ReLU(True))
            for i in range(n_blocks):
                layer += [SNResnetBlock(output_nums, padding_type='reflect', use_dropout=False, use_bias=True), nn.ReLU(True)]
            return layer
        
        model = []
        model += ResBlock(input_nums, 64, n_blocks=6)
        new_ch = 64
        for i in range(3):
            prev_ch = new_ch
            new_ch = prev_ch * 2
            model += ResBlock(prev_ch, new_ch, n_blocks=6)
        
        model += [ SNConv2d(new_ch, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
                        nn.AdaptiveAvgPool2d((1, 1)), 
                        nn.Flatten() ]
        self.Net = nn.Sequential(*model)

    def forward(self, input):
        output = self.Net(input)
        return output

class DenseLayer(nn.Module):
    def __init__(self, in_channels, bn_size, growth_rate) -> None:
        super().__init__()
        layer = [nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(bn_size * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)]
        self.Net = nn.Sequential(*layer)
        
    def forward(self, input):
        return torch.concat([input, self.Net(input)], dim=1)

class SNTransformDiscriminator(SNResnetDiscriminator):
    def __init__(self, input_nums = 34, n_layer = 3):
        super().__init__(input_nums = 34, n_layer = 3)
        def ResBlock(dim, input_nums, output_nums, n_blocks = 1):
            layer = []
            layer += [SN_MSAB(input_nums, dim, input_nums // dim, n_blocks)]
            layer.append(SNConv2d(input_nums, output_nums, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
            layer.append(nn.ReLU(True))
            return layer
        
        model = []
        model += ResBlock(input_nums, input_nums,  input_nums, n_blocks=3)
        new_ch = input_nums
        for i in range(n_layer):
            prev_ch = new_ch
            new_ch = prev_ch * 2
            model += ResBlock(input_nums, prev_ch, new_ch, n_blocks=3)
        
        model += [ SNConv2d(new_ch, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
                    nn.AdaptiveAvgPool2d((1, 1)), 
                    nn.Flatten() ]
        self.Net = nn.Sequential(*model)

class DenseBlock(nn.Module):
    def __init__(self, 
                num_layers,
                num_input_features,
                bn_size,
                growth_rate) -> None:
        super().__init__()
        model = []
        for i in range(num_layers):
            model += [DenseLayer(num_input_features + growth_rate * i, bn_size = bn_size, growth_rate =growth_rate)]
        self.Net = nn.Sequential(*model)
        
    def forward(self, input):
        return self.Net(input)

def ReverseTuples(tuples):
    new_tup = tuples[::-1]
    return new_tup

class DensenetGenerator(nn.Module):
    def __init__(self, 
                 inchannels, 
                 outchannels, 
                 num_init_features = 64, 
                 block_config = (6, 12, 24, 16), 
                 bn_size = 4, 
                 growth_rate = 32, 
                 center_layer = 6
                 ):
        super().__init__()
        def DownSample(num_input_features, num_output_features):
            layer = [nn.BatchNorm2d(num_input_features), 
                     nn.ReLU(inplace=True), 
                     nn.Conv2d(num_input_features, num_output_features, kernel_size=3, stride=2, padding=1, bias=False)]
            return layer
        
        def UpSample(num_input_features, num_output_features):
            layer = [nn.BatchNorm2d(num_input_features), 
                     nn.ReLU(inplace=True), 
                     nn.ConvTranspose2d(num_input_features, num_output_features, kernel_size=4, stride=2, padding=1, bias=False)]
            return layer
        
        layer = [nn.Conv2d(inchannels, num_init_features, kernel_size=1, stride=1, bias=False)]
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            layer += [DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
            )]
            num_features = num_features + num_layers * growth_rate
            layer += DownSample(num_input_features=num_features, num_output_features=num_features // 2)
            num_features = num_features // 2
        
        layer += [DenseBlock(
            num_layers=center_layer,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
        )]
        num_features = num_features + center_layer * growth_rate
        
        block_config = ReverseTuples(block_config)
        for i, num_layers in enumerate(block_config):
            layer += UpSample(num_input_features=num_features, num_output_features=num_features // 2)
            num_features = num_features // 2
            layer += [DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
            )]
            num_features = num_features + num_layers * growth_rate
        
        layer += [nn.BatchNorm2d(num_features), 
                  nn.ReLU(inplace=True), 
                  nn.Conv2d(num_features, outchannels, kernel_size=1, stride=1)]
        self.Net = nn.Sequential(*layer)
        
    def forward(self, input):
        return self.Net(input)

class Spectral_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        

if __name__ == '__main__':
    model = SNTransformDiscriminator(34, 3)
    model = model.cuda()
    
    input = torch.rand([1, 34, 128, 128])
    input = input.cuda()
    output = model(input)
    print(output.shape)