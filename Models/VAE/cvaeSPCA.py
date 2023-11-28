import sys
sys.path.append('./')
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.optim as optim
import torch.backends.cudnn as cudnn
from dataset.dataset import TrainDataset, ValidDataset
# from hsi_dataset import TrainDataset, ValidDataset
from utils import AverageMeter, record_loss, Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_Fid, Loss_SAM, Loss_SSIM, reconRGB, Loss_SID
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
from options import opt
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
if opt.multigpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class channel_attn(nn.Module):
    def __init__(self, dimIn, heads = 8, dim_head = 64, dropout = 0.) -> None:
        super().__init__()
        self.dimIn = dimIn
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dimhidden = heads * dim_head
        self.to_q = nn.Linear(dimIn, self.dimhidden, bias=False)
        self.to_k = nn.Linear(dimIn, self.dimhidden, bias=False) 
        self.to_v = nn.Linear(dimIn, self.dimhidden, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.dimhidden, 1),
            nn.Sigmoid(),
            nn.Dropout(dropout), 
            nn.Flatten()
        )
         
    def forward(self, x_in):
        b, c, h, w = x_in.shape
        x = rearrange(x_in, 'b c h w -> b c (h w)')
        batch, n, dimin = x.shape
        assert dimin == self.dimIn  
        nh = self.heads
        dk = self.dimhidden // nh
        
        q = self.to_q(x).reshape(batch, n, nh, dk).transpose(1, 2)
        k = self.to_k(x).reshape(batch, n, nh, dk).transpose(1, 2)
        v = self.to_v(x).reshape(batch, n, nh, dk).transpose(1, 2)
        
        dist = torch.matmul(q, k.transpose(2,3)) * self.scale
        dist = torch.softmax(dist, dim = -1)
        att = torch.matmul(dist, v)
        att = att.transpose(1, 2).reshape(batch, n, self.dimhidden)
        
        out = self.to_out(att).reshape(b, c, 1, 1)
        
        return x_in * out


class ConvCABlock(nn.Module):
    def __init__(self, inchannels, img_size) -> None:
        super().__init__()
        self.inchannels = inchannels
        # self.conv = nn.Conv2d(self.inchannels, self.inchannels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn = nn.BatchNorm2d(self.inchannels)
        # self.relu = nn.ReLU(inplace=True)
        # self.ChannelAttention = CALayer(self.inchannels, reduction=4)
        self.ChannelAttention = channel_attn(img_size ** 2)
    
    def forward(self, x):
        identity = x
        # y = self.conv(x)
        # y = self.bn(y)
        # y = self.relu(y)
        y = self.ChannelAttention(x)
        return y + identity
    

class Adaptive_Channel_Attention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            # nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )

    def forward(self, x_in):
        
        B, C, H, W = x_in.shape
        N = H * W
        x = rearrange(x_in, 'b c h w -> b (h w) c')
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        v_ = v.reshape(B, C, N).contiguous().view(B, C, H, W)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # attention output
        attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)

        # convolution output
        conv_x = self.dwconv(v_)

        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2,-1).contiguous().view(B, C, H, W)
        channel_map = self.channel_interaction(attention_reshape)
        # S-Map (before sigmoid)
        spatial_map = self.spatial_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, N, 1)

        # S-I
        attened_x = attened_x * torch.sigmoid(spatial_map)
        # C-I
        conv_x = conv_x * torch.sigmoid(channel_map)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        x = attened_x + conv_x

        x = self.proj(x)
        x = self.proj_drop(x)
        
        x = rearrange(x, 'b (h w) c -> b c h w', h = H)

        return x

class ConditionalVAESPCA(nn.Module):

    def __init__(self, in_channels: int, condition_channels: int,) -> None:
        super().__init__()

        self.in_channels = in_channels
        hidden_dims = [64, 128]

        def ConvT(input_nums, output_nums):
            layer = []
            layer.append(nn.ConvTranspose2d(input_nums, output_nums, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU())
            return layer
        
        def Conv(input_nums, output_nums, stride):
            layer = []
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(3,3), stride=stride, padding=(1,1), bias=False))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU())
            return layer
        
        self.embed_class0 = nn.Conv2d(condition_channels, condition_channels, kernel_size=1, bias=False)
        self.embed_class1 = nn.Sequential(*Conv(condition_channels, hidden_dims[0], stride=2))
        self.embed_class2 = nn.Sequential(*Conv(hidden_dims[0], hidden_dims[1], stride=2))
        
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            
        self.hidden_dims = hidden_dims
        # Build Encoder
        self.encconv1 = nn.Sequential(
            *Conv(in_channels, hidden_dims[0], stride=2))
        
        self.encconv2 = nn.Sequential(*Conv(hidden_dims[0], hidden_dims[1], stride=2))
        
        # self.fc_mu0 = ConvCABlock(in_channels, 128)
        # self.fc_var0 = ConvCABlock(in_channels, 128)
        # self.fc_mu1 = ConvCABlock(hidden_dims[0], 64)
        # self.fc_var1 = ConvCABlock(hidden_dims[0], 64)
        # self.fc_mu2 = ConvCABlock(hidden_dims[1], 32)
        # self.fc_var2 = ConvCABlock(hidden_dims[1], 32)
        
        self.fc_mu0 = Adaptive_Channel_Attention(in_channels)
        self.fc_var0 = Adaptive_Channel_Attention(in_channels)
        self.fc_mu1 = Adaptive_Channel_Attention(hidden_dims[0])
        self.fc_var1 = Adaptive_Channel_Attention(hidden_dims[0])
        self.fc_mu2 = Adaptive_Channel_Attention(hidden_dims[1])
        self.fc_var2 = Adaptive_Channel_Attention(hidden_dims[1])

        # Build Decoder

        # self.decoder_input0 = ConvCABlock(hidden_dims[1] * 2, 32)
        # self.decoder_input1 = ConvCABlock(hidden_dims[0] * 2, 64)
        # self.decoder_input2 = ConvCABlock(in_channels + condition_channels, 128)
        
        self.decoder_input0 = Adaptive_Channel_Attention(hidden_dims[1] * 2)
        self.decoder_input1 = Adaptive_Channel_Attention(hidden_dims[0] * 2)
        self.decoder_input2 = Adaptive_Channel_Attention(in_channels + condition_channels)
        
        self.decconv1 = nn.Sequential(*ConvT(hidden_dims[1] * 2, hidden_dims[1]))
        self.decconv2 = nn.Sequential(*ConvT(hidden_dims[1] * 2, in_channels + condition_channels))

        self.final_layer = nn.Sequential(
                            nn.Conv2d((in_channels + condition_channels) * 2, out_channels= self.in_channels,
                                      kernel_size= 3, padding= 1, bias=False),
                            # nn.Tanh()
                            # nn.Sigmoid()
                            )

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        embed_input = self.embed_data(input)
        
        embed_input1 = self.encconv1(embed_input)
        embed_input2 = self.encconv2(embed_input1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu0 = self.fc_mu0(embed_input)
        log_var0 = self.fc_var0(embed_input) # [31, H, W]
        mu1 = self.fc_mu1(embed_input1)
        log_var1 = self.fc_var1(embed_input1) # [64, H / 2, W / 2]
        mu2 = self.fc_mu2(embed_input2)
        log_var2 = self.fc_var2(embed_input2) # [128, H / 4, W / 4]

        return [mu0, log_var0, mu1, log_var1, mu2, log_var2]

    def decode(self, z0, z1, z2):
        # z0: [256, H / 4, W / 4] torch.concat([z0, y2])
        result0 = self.decoder_input0(z0) # result0: [256, H / 4, H / 4]
        result0 = self.decconv1(result0) # result0: [128, H / 2, W / 2]
        # z1: [128, H / 2, W / 2] torch.concat([z1, y1])
        result1 = self.decoder_input1(z1) # result1 : [128, H / 2, W / 2]
        result1 = torch.concat([result1, result0], dim = 1) # result1: [256, H / 2, W / 2]
        result1 = self.decconv2(result1) # result1: [34, H, W]
        # z2: [34, H, W]
        result2 = self.decoder_input2(z2) # result2: [34, H, W]
        result2 = torch.concat([result2, result1], dim = 1) # result2: [68, H, W]
        result = self.final_layer(result2)
        return result

    def reparameterize(self, mu0, logvar0, mu1, logvar1, mu2, logvar2):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std0 = torch.exp(0.5 * logvar0)
        eps0 = torch.randn_like(std0)
        std1 = torch.exp(0.5 * logvar1)
        eps1 = torch.randn_like(std1)
        std2 = torch.exp(0.5 * logvar2)
        eps2 = torch.randn_like(std2)
        return [eps0 * std0 + mu0, eps1 * std1 + mu1, eps2 * std2 + mu2]

    def forward(self, input, y):
        
        embed_y0 = self.embed_class0(y)
        embed_y1 = self.embed_class1(embed_y0)
        embed_y2 = self.embed_class2(embed_y1)
        embedded_input = self.embed_data(input)

        [mu0, log_var0, mu1, log_var1, mu2, log_var2] = self.encode(embedded_input)

        [z2, z1, z0] = self.reparameterize(mu0, log_var0, mu1, log_var1, mu2, log_var2)

        z0 = torch.cat([z0, embed_y2], dim = 1)
        z1 = torch.cat([z1, embed_y1], dim = 1)
        z2 = torch.cat([z2, embed_y0], dim = 1)
        
        recon = self.decode(z0, z1, z2)
        
        return  [recon, input, mu0, log_var0, mu1, log_var1, mu2, log_var2]

    def loss_function(self, recons, input, mu0, log_var0, mu1, log_var1, mu2, log_var2):

        # kld_weight = 3e-3  # Account for the minibatch samples from the dataset
        # kld_weight = 1024 / 128 / 128 / 31  # Account for the minibatch samples from the dataset
        kld_weight = 1 / 128 / 128  # Account for the minibatch samples from the dataset
        recons_loss =F.l1_loss(recons, input)
        
        Mu0 = rearrange(mu0, 'b c h w -> b (c h w)')
        Log_var0 = rearrange(log_var0, 'b c h w -> b (c h w)')
        kld_weight0 = 1 / Mu0.shape[1]
        Mu1 = rearrange(mu1, 'b c h w -> b (c h w)')
        Log_var1 = rearrange(log_var1, 'b c h w -> b (c h w)')
        kld_weight1 = 1 / Mu1.shape[1]
        Mu2 = rearrange(mu2, 'b c h w -> b (c h w)')
        Log_var2 = rearrange(log_var2, 'b c h w -> b (c h w)')
        kld_weight2 = 1 / Mu2.shape[1]

        kld_loss0 = torch.mean(-0.5 * torch.sum(1 + Log_var0 - Mu0 ** 2 - Log_var0.exp(), dim = 1), dim = 0)
        kld_loss1 = torch.mean(-0.5 * torch.sum(1 + Log_var1 - Mu1 ** 2 - Log_var1.exp(), dim = 1), dim = 0)
        kld_loss2 = torch.mean(-0.5 * torch.sum(1 + Log_var2 - Mu2 ** 2 - Log_var2.exp(), dim = 1), dim = 0)

        # loss = recons_loss + kld_weight * (kld_weight0 * kld_loss0 + kld_weight1 * kld_loss1 + kld_weight2 * kld_loss2)
        loss = recons_loss + kld_weight * (kld_loss0 + kld_loss1 + kld_loss2)
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_weight * (kld_loss0 + kld_loss1 + kld_loss2)}

    def sample(self, y):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        embed_y0 = self.embed_class0(y)
        embed_y1 = self.embed_class1(embed_y0)
        embed_y2 = self.embed_class2(embed_y1)
        z2 = torch.randn(y.shape[0], 31, y.shape[2], y.shape[3]).cuda()
        z1 = torch.randn_like(embed_y1).cuda()
        z0 = torch.randn_like(embed_y2).cuda()
        
        z0 = torch.cat([z0, embed_y2], dim = 1)
        z1 = torch.cat([z1, embed_y1], dim = 1)
        z2 = torch.cat([z2, embed_y0], dim = 1)
        
        samples = self.decode(z0, z1, z2)
        return samples


# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_sam = Loss_SAM()
criterion_sid = Loss_SID()
# criterion_fid = Loss_Fid().to(device)
# criterion_ssim = Loss_SSIM().to(device)

class train_cvaeSPCA():
    def __init__(self, opt, multiGPU = False) -> None:
        self.opt = opt
        self.model = ConditionalVAESPCA(in_channels=31, condition_channels=3)
        self.multiGPU = multiGPU
        if self.multiGPU:
            self.model = nn.DataParallel(self.model)
        self.model.to(device)
        self.root = '/work3/s212645/Spectral_Reconstruction/checkpoint/cvaeSPCA/'
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
        
        # iterations
        self.epoch = 0
        self.end_epoch = opt.end_epoch
        per_epoch_iteration = 1000
        self.total_iteration = per_epoch_iteration*opt.end_epoch
        self.iteration = 0
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.total_iteration, eta_min=1e-6)
        
        self.metrics = {
            'MRAE':[],
            'RMSE':[],
            'PSNR':[],
            'SAM':[]
        }
        # make checkpoint dir
        if not os.path.exists(opt.outf):
            os.makedirs(opt.outf)
        if not os.path.exists(self.root):
            os.makedirs(self.root)
    
       
    def load_dataset(self):
        # load dataset
        print("\nloading dataset ...")
        self.train_data = TrainDataset(data_root=self.opt.data_root, crop_size=self.opt.patch_size, test_ratio=0.1)
        print(f"Iteration per epoch: {len(self.train_data)}")
        self.val_data = ValidDataset(data_root=self.opt.data_root, crop_size=self.opt.patch_size, test_ratio=0.1)
        print("Validation set samples: ", len(self.val_data))

    def train(self):
        self.load_dataset()
        record_mrae_loss = 1000
        while self.iteration<self.total_iteration:
            self.model.train()
            losses = AverageMeter()
            train_loader = DataLoader(dataset=self.train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=2,
                                    pin_memory=True, drop_last=True)
            val_loader = DataLoader(dataset=self.val_data, batch_size=self.opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            for i, (images, labels) in enumerate(train_loader):
                labels = labels.cuda()
                images = images.cuda()
                images = Variable(images)
                labels = Variable(labels)
                lr = self.optimizer.param_groups[0]['lr']
                self.optimizer.zero_grad()
                [output, input, mu0, log_var0, mu1, log_var1, mu2, log_var2] = self.model(labels, images)
                if self.multiGPU:
                    Loss_dict = self.model.module.loss_function(output, input, mu0, log_var0, mu1, log_var1, mu2, log_var2)
                else:
                    Loss_dict = self.model.loss_function(output, input, mu0, log_var0, mu1, log_var1, mu2, log_var2)
                loss = Loss_dict['loss']
                loss_mrae = criterion_mrae(output, input)
                loss += loss_mrae
                loss.backward()
                losses.update(loss_mrae.data)
                self.optimizer.step()
                self.scheduler.step()
                self.iteration = self.iteration+1
                if self.iteration % 20 == 0:
                    print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f'
                        % (self.iteration, self.total_iteration, lr, losses.avg))
            # if self.iteration % 1000 == 0:
            mrae_loss, rmse_loss, psnr_loss, sam_loss, sid_loss = self.validate(val_loader)
            print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}, SAM: {sam_loss}, SID: {sid_loss}')
            # Save model
            # if torch.abs(mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or self.iteration % 5000 == 0:
            print(f'Saving to {self.root}')
            self.save_checkpoint()
            if mrae_loss < record_mrae_loss:
                record_mrae_loss = mrae_loss
                self.save_checkpoint(best=True)
            # print loss
            print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                "Test RMSE: %.9f, Test PSNR: %.9f, SAM: %.9f, SID: %.9f " % (self.iteration, 
                                                                self.epoch, lr, 
                                                                losses.avg, mrae_loss, rmse_loss, psnr_loss, sam_loss, sid_loss))
            self.epoch += 1
            
    def validate(self, val_loader):
        self.model.eval()
        losses_mrae = AverageMeter()
        losses_rmse = AverageMeter()
        losses_psnr = AverageMeter()
        losses_sam = AverageMeter()
        losses_sid = AverageMeter()
        # losses_fid = AverageMeter()
        # losses_ssim = AverageMeter()
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)
            with torch.no_grad():
                # compute output
                if self.multiGPU:
                    output = self.model.module.sample(input)
                else:
                    output = self.model.sample(input)
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
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
        else:
            state = {
                'epoch': self.epoch,
                'iter': self.iteration,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
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
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iteration = checkpoint['iter']
        self.epoch = checkpoint['epoch']
        try:
            self.load_metrics()
        except:
            pass
        print("pretrained model loaded")

if __name__ == '__main__':
    # model = ConditionalVAESPCA(in_channels=31, condition_channels=3).cuda()
    # input = torch.rand([1, 31, 128, 128]).cuda()
    # y = torch.rand([1, 3, 128, 128]).cuda()
    # [recon, input, mu0, log_var0, mu1, log_var1, mu2, log_var2] = model(input, y)
    # print(recon.shape)
    # loss = model.loss_function(recon, input, mu0, log_var0, mu1, log_var1, mu2, log_var2)
    # print(loss['Reconstruction_Loss'].item())
    # print(loss['KLD'].item())
    # output = model.sample(y)
    # print(output.shape)
    
    spec = train_cvaeSPCA(opt, multiGPU=opt.multigpu)
    spec.load_checkpoint()
    spec.train()