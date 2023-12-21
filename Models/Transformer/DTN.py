import sys
sys.path.append('./')
import torch.nn as nn
import torch
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.optim as optim
import torch.backends.cudnn as cudnn
from dataset.dataset import TrainDataset, ValidDataset, TestDataset
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
        # self.dwconv = nn.Sequential(
        #     nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
        #     nn.BatchNorm2d(dim),
        #     nn.GELU()
        # )
        # self.channel_interaction = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(dim, dim // 8, kernel_size=1),
        #     nn.GELU(),
        #     nn.Conv2d(dim // 8, dim, kernel_size=1),
        # )
        # self.spatial_interaction = nn.Sequential(
        #     nn.Conv2d(dim, dim // 16, kernel_size=1),
        #     nn.BatchNorm2d(dim // 16),
        #     nn.GELU(),
        #     nn.Conv2d(dim // 16, 1, kernel_size=1)
        # )
        self.dwconv = DWCONV(dim)
        self.channel_interaction = ChannelAtten(dim)
        self.spatial_interaction = SpatialAtten(dim)
    
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
        
        # self.dwconv = nn.Sequential(
        #     nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
        #     nn.BatchNorm2d(dim),
        #     nn.GELU()
        # )
        # self.channel_interaction = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(dim, dim // 8, kernel_size=1),
        #     nn.GELU(),
        #     nn.Conv2d(dim // 8, dim, kernel_size=1),
        # )
        # self.spatial_interaction = nn.Sequential(
        #     nn.Conv2d(dim, dim // 16, kernel_size=1),
        #     nn.BatchNorm2d(dim // 16),
        #     nn.GELU(),
        #     nn.Conv2d(dim // 16, 1, kernel_size=1)
        # )
        self.dwconv = DWCONV(dim)
        self.channel_interaction = ChannelAtten(dim)
        self.spatial_interaction = SpatialAtten(dim)
        
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


class TrainDTN():
    def __init__(self, opt, multiGPU = False) -> None:
        super().__init__()
        self.multiGPU = multiGPU
        self.opt = opt
        self.G = DTN(in_dim=3, 
                 out_dim=31,
                 img_size=[128, 128], 
                 window_size=8, 
                 n_block=[2,2,2,2], 
                 bottleblock = 4)
        if self.multiGPU:
            self.G = nn.DataParallel(self.G)
        self.G.cuda()
        
        self.epoch = 0
        self.end_epoch = opt.end_epoch
        per_epoch_iteration = 1000
        self.total_iteration = per_epoch_iteration*opt.end_epoch
        self.iteration = 0
        
        self.optimG = optim.Adam(self.G.parameters(), lr=self.opt.init_lr, betas=(0.9, 0.999))
        self.schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimG, self.total_iteration, eta_min=1e-6)
        self.lossl1 = nn.L1Loss()
        self.root = '/work3/s212645/Spectral_Reconstruction/checkpoint/DTN/'
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.metrics = {
            'MRAE':[],
            'RMSE':[],
            'PSNR':[],
            'SAM':[]
        }
    
    def load_dataset(self):
        # load dataset
        print("\nloading dataset ...")
        self.train_data = TrainDataset(data_root=self.opt.data_root, crop_size=self.opt.patch_size, valid_ratio = 0.1, test_ratio=0.1)
        print(f"Iteration per epoch: {len(self.train_data)}")
        self.val_data = ValidDataset(data_root=self.opt.data_root, crop_size=self.opt.patch_size, valid_ratio = 0.1, test_ratio=0.1)
        print("Validation set samples: ", len(self.val_data))
        self.test_data = TestDataset(data_root=self.opt.data_root, crop_size=self.opt.patch_size, valid_ratio = 0.1, test_ratio=0.1)
        print("Validation set samples: ", len(self.val_data))
    
    def train(self):
        self.load_dataset()
        record_mrae_loss = 1000
        while self.iteration<self.total_iteration:
            self.G.train()
            losses = AverageMeter()
            train_loader = DataLoader(dataset=self.train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=2,
                                    pin_memory=True, drop_last=True)
            val_loader = DataLoader(dataset=self.val_data, batch_size=self.opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            for i, (images, labels) in enumerate(train_loader):
                labels = labels.cuda()
                images = images.cuda()
                images = Variable(images)
                labels = Variable(labels)
                x_fake = self.G(images)
                # train G
                self.optimG.zero_grad()
                lrG = self.optimG.param_groups[0]['lr']
                lossl1 = self.lossl1(x_fake, labels)
                losssam = SAM(x_fake, labels)
                loss_G = lossl1 + losssam
                # train the generator
                loss_G.backward()
                self.optimG.step()
                self.schedulerG.step()
                
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
                
    def validate(self, val_loader):
        self.G.eval()
        losses_mrae = AverageMeter()
        losses_rmse = AverageMeter()
        losses_psnr = AverageMeter()
        losses_sam = AverageMeter()
        losses_sid = AverageMeter()
        # losses_fid = AverageMeter()
        # losses_ssim = AverageMeter()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            with torch.no_grad():
                # compute output
                output = self.G(input)
                loss_mrae = criterion_mrae(output, target)
                loss_rmse = criterion_rmse(output, target)
                loss_psnr = criterion_psnr(output, target)
                loss_sam = criterion_sam(output, target)
                loss_sid = criterion_sid(output, target)
                # rgb = reconRGB(output)
                # loss_fid = criterion_fid(rgb, input)
                # loss_ssim = criterion_ssim(rgb, input)
            # record loss
            losses_mrae.update(loss_mrae.data)
            losses_rmse.update(loss_rmse.data)
            losses_psnr.update(loss_psnr.data)
            losses_sam.update(loss_sam.data)
            losses_sid.update(loss_sid.data)
            # losses_fid.update(loss_fid.data)
            # losses_ssim.update(loss_ssim.data)
        criterion_sam.reset()
        criterion_psnr.reset()
        self.metrics['MRAE'].append(losses_mrae.avg)
        self.metrics['RMSE'].append(losses_rmse.avg)
        self.metrics['PSNR'].append(losses_psnr.avg)
        self.metrics['SAM'].append(losses_sam.avg)
        self.save_metrics()
        return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg, losses_sid.avg
    
    def test(self, test_loader):
        pass
    
    def save_metrics(self):
        name = 'metrics.pth'
        torch.save(self.metrics, os.path.join(self.root, name))
        
    def load_metrics(self):
        name = 'metrics.pth'
        checkpoint = torch.load(os.path.join(self.root, name))
        self.metrics = checkpoint
    
    def save_checkpoint(self, best = False):
        if self.multiGPU:
            state = {
                'epoch': self.epoch,
                'iter': self.iteration,
                'G': self.G.module.state_dict(),
                'optimG': self.optimG.state_dict(),
            }
        else:
            state = {
                'epoch': self.epoch,
                'iter': self.iteration,
                'G': self.G.state_dict(),
                'optimG': self.optimG.state_dict(),
            }
        if best: 
            name = 'net_epoch_best.pth'
        else:
            name = 'net_%04depoch.pth' % self.epoch
            oldname = 'net_%04depoch.pth' % (self.epoch - 5)
            if os.path.exists(os.path.join(self.root, oldname)):
                os.remove(os.path.join(self.root, oldname))
                print(oldname, ' Removed. ')
        torch.save(state, os.path.join(self.root, name))
        
    def load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.root, 'net_epoch_best.pth'))
        if self.multiGPU:
            self.G.module.load_state_dict(checkpoint['G'])
        else:
            self.G.load_state_dict(checkpoint['G'])
        self.optimG.load_state_dict(checkpoint['optimG'])
        self.iteration = checkpoint['iter']
        self.epoch = checkpoint['epoch']
        try:
            self.load_metrics()
        except:
            pass
        print("pretrained model loaded, iteration: ", self.iteration)

if __name__ == '__main__':
    spec = TrainDTN(opt, multiGPU=opt.multigpu)
    if opt.loadmodel:
        try:
            spec.load_checkpoint()
        except:
            print('pretrained model loading failed')
    spec.train()