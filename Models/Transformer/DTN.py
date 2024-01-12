import sys
sys.path.append('./')
import torch.nn as nn
import torch
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.optim as optim
import torch.backends.cudnn as cudnn
from dataset.datasets import TrainDataset, ValidDataset, TestDataset
from utils import AverageMeter, record_loss, Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_Fid, \
    Loss_SAM, Loss_SSIM, reconRGB, Loss_SID, SAM
from options import opt
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
import functools
from Models.Transformer.MST_Plus_Plus import MSAB
from Models.Transformer.swin_transformer import SwinTransformerBlock
from Models.Transformer.Base import BaseModel
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
if opt.multigpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_sam = Loss_SAM()
criterion_sid = Loss_SID()
# criterion_fid = Loss_Fid().cuda()
# criterion_ssim = Loss_SSIM().cuda()

class SWTB(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=1, window_size=8):
        super().__init__()
        self.input_resolution = input_resolution
        self.model = SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.input_resolution[0] and W == self.input_resolution[1]
        x = rearrange(x, 'b c h w -> b (h w) c', h=H, w=W)
        x = self.model(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x

class ChannelAtten(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
    
    def forward(self, x):
        return self.channel_interaction(x)

class SpatialAtten(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )
    
    def forward(self, x):
        return self.spatial_interaction(x)

class DWCONV(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.dwconv(x)

class Adaptive_SWTB(nn.Module):
    def __init__(self, dim=31, 
                 input_resolution=(128,128), 
                 num_heads=1, 
                 window_size=8):
        super().__init__()
        self.model = SWTB(dim=dim, 
                          input_resolution=input_resolution, 
                          num_heads=num_heads, 
                          window_size=window_size)
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )
        # self.dwconv = DWCONV(dim)
        # self.channel_interaction = ChannelAtten(dim)
        # self.spatial_interaction = SpatialAtten(dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        attnx = self.model(x)
        convx = self.dwconv(x)
        channelinter = torch.sigmoid(self.channel_interaction(convx))
        spatialinter = torch.sigmoid(self.spatial_interaction(convx))
        channelx = attnx * channelinter
        spatialx = convx * spatialinter
        return channelx + spatialx

class Adaptive_MSAB(nn.Module):
    def __init__(self, dim=31, 
                 num_blocks=2, 
                 dim_head=31, 
                 heads=1):
        super().__init__()
        self.model = MSAB(dim=dim, num_blocks=num_blocks, dim_head=dim_head, heads=heads)
        
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )
        # self.dwconv = DWCONV(dim)
        # self.channel_interaction = ChannelAtten(dim)
        # self.spatial_interaction = SpatialAtten(dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        attnx = self.model(x)
        convx = self.dwconv(x)
        channelinter = torch.sigmoid(self.channel_interaction(convx))
        spatialinter = torch.sigmoid(self.spatial_interaction(convx))
        channelx = convx * channelinter
        spatialx = attnx * spatialinter
        return channelx + spatialx

class DTNBlock(nn.Module):
    def __init__(self, dim, dim_head, input_resolution, num_heads, window_size, num_block):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.input_resolution = tuple(input_resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        layer = []
        for i in range(num_block):
            layer += [Adaptive_MSAB(dim, num_blocks=2, dim_head=dim_head, heads=dim // dim_head)]
            layer += [Adaptive_SWTB(dim, self.input_resolution, num_heads=dim // dim_head, window_size=window_size)]
        self.model = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.model(x)

class DownSample(nn.Module):
    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(outchannel), 
            nn.ReLU(True)
        )
        
    def forward(self, x):
        return self.model(x)
class UpSample(nn.Module):
    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(inchannel, outchannel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(outchannel), 
            nn.ReLU(True)
        )
        
    def forward(self, x):
        return self.model(x)

def ReverseTuples(tuples):
    new_tup = tuples[::-1]
    return new_tup

class DTN(nn.Module):
    def __init__(self, in_dim, 
                 out_dim,
                 img_size = [128, 128], 
                 window_size = 8, 
                 n_block=[2,2,2,2], 
                 bottleblock = 4):
        super().__init__()
        dim = out_dim
        self.embedding = nn.Conv2d(in_dim, dim, 3, 1, 1, bias=False)
        self.stage = len(n_block)-1
        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for num_block in n_block:
            self.encoder_layers.append(nn.ModuleList([
                DTNBlock(dim = dim_stage, 
                         dim_head = dim, 
                         input_resolution = img_size, 
                         num_heads = dim_stage // dim, 
                         window_size = window_size, 
                         num_block = num_block),
                DownSample(dim_stage, dim_stage * 2),
            ]))
            img_size[0] = img_size[0] // 2
            img_size[1] = img_size[1] // 2
            dim_stage *= 2
        
        self.bottleneck = DTNBlock(dim = dim_stage, 
                         dim_head = dim, 
                         input_resolution = img_size, 
                         num_heads = dim_stage // dim, 
                         window_size = window_size, 
                         num_block = bottleblock)
        
        # Decoder
        self.decoder_layers = nn.ModuleList([])
        n_block.reverse()
        for num_block in n_block:
            img_size[0] = img_size[0] * 2
            img_size[1] = img_size[1] * 2
            self.decoder_layers.append(nn.ModuleList([
                UpSample(dim_stage, dim_stage // 2), 
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                DTNBlock(dim = dim_stage // 2, 
                         dim_head = dim, 
                         input_resolution = img_size, 
                         num_heads = (dim_stage // 2) // dim, 
                         window_size = window_size, 
                         num_block = num_block),
            ]))
            dim_stage //= 2
        
        self.mapping = nn.Conv2d(dim, out_dim, 3, 1, 1, bias=False)
        
    def forward(self, x):
        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        for (MSAB, FeaDownSample) in self.encoder_layers:
            fea = MSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-i]], dim=1))
            fea = LeWinBlcok(fea)

        # Mapping
        out = self.mapping(fea) + fea

        return out

if __name__ == '__main__':
    model = DTN(in_dim=3, 
                    out_dim=31,
                    img_size=[128, 128], 
                    window_size=8, 
                    n_block=[2,2,2,2], 
                    bottleblock = 4)
    spec = BaseModel(opt, model, model_name='DTN')
    if opt.loadmodel:
        try:
            spec.load_checkpoint()
        except:
            print('pretrained model loading failed')
    spec.train()