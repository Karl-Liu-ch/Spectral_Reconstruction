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
from utils import AverageMeter, record_loss, Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_Fid, Loss_SAM, Loss_SSIM, reconRGB, Loss_SID, SAM
from options import opt
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
import functools
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

class UnetGenerator(nn.Module):
    def __init__(self, num_input, num_output):
        super(UnetGenerator, self).__init__()
        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.LeakyReLU(0.2))
            return layer
        def TransposeConv(input_nums, output_nums):
            layer = []
            layer.append(nn.ConvTranspose2d(input_nums, output_nums, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU(True))
            return layer
        
        self.embed = nn.Conv2d(num_input, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.down1 = nn.Sequential(*Conv(64, 128))
        self.down2 = nn.Sequential(*Conv(128, 256))
        self.down3 = nn.Sequential(*Conv(256, 512))
        self.down4 = nn.Sequential(*Conv(512, 1024))
        self.up4 = nn.Sequential(*TransposeConv(1024, 512))
        self.up3 = nn.Sequential(*TransposeConv(1024, 256))
        self.up2 = nn.Sequential(*TransposeConv(512, 128))
        self.up1 = nn.Sequential(*TransposeConv(256, 64))
        self.out = nn.Sequential(
            nn.Conv2d(128, num_output, kernel_size=3, stride=1, padding=1, bias=False), 
            # nn.Tanh()
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
        super(NLayerDiscriminator, self).__init__()
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
        # return F.sigmoid(self.model(input))
    
class pix2pixlsgan():
    def __init__(self, opt, multiGPU = False) -> None:
        super().__init__()
        self.multiGPU = multiGPU
        self.opt = opt
        self.G = UnetGenerator(3, 31)
        self.D = NLayerDiscriminator(34)
        if self.multiGPU:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
        self.G.cuda()
        self.D.cuda()
        
        self.epoch = 0
        self.end_epoch = opt.end_epoch
        per_epoch_iteration = 1000
        self.total_iteration = per_epoch_iteration*opt.end_epoch
        self.iteration = 0
        
        self.optimG = optim.Adam(self.G.parameters(), lr=self.opt.init_lr, betas=(0.9, 0.999))
        self.schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimG, self.total_iteration, eta_min=1e-6)
        self.optimD = optim.Adam(self.D.parameters(), lr=self.opt.init_lr, betas=(0.9, 0.999))
        self.schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimD, self.total_iteration, eta_min=1e-6)
        self.lossl1 = nn.L1Loss()
        self.criterionGAN = nn.MSELoss()
        self.lamda = 1
        self.root = '/work3/s212645/Spectral_Reconstruction/checkpoint/pix2pixlsgan/'
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
                realAB = torch.concat([images, labels], dim=1)
                x_fake = self.G(images)
                fakeAB = torch.concat([images, x_fake],dim=1)
                
                # train D
                for p in self.D.parameters():
                    p.requires_grad = True
                self.D.zero_grad()
                lrD = self.optimD.param_groups[0]['lr']
                self.optimD.zero_grad()
                # realAB = torch.concat([images, labels], dim=1)
                D_real = self.D(realAB)
                real_labels = torch.ones_like(D_real).cuda()
                fake_labels = torch.zeros_like(D_real).cuda()
                loss_real = self.criterionGAN(D_real, real_labels)
                loss_real.backward()
                # x_fake = self.G(images).detach()
                # fakeAB = torch.concat([images, x_fake],dim=1)
                D_fake = self.D(fakeAB.detach())
                loss_fake = self.criterionGAN(D_fake, fake_labels)
                loss_fake.backward()
                self.optimD.step()
                self.schedulerD.step()
                
                # train G
                self.G.zero_grad()
                self.optimG.zero_grad()
                lrG = self.optimG.param_groups[0]['lr']
                for p in self.D.parameters():
                    p.requires_grad = False
                # x_fake = self.G(images)
                # fakeAB = torch.concat([images, x_fake],dim=1)
                pred_fake = self.D(fakeAB)
                loss_G = self.criterionGAN(pred_fake, real_labels)
                lossl1 = self.lossl1(x_fake, labels) * self.lamda
                loss_sam = SAM(x_fake, labels) * self.lamda
                loss_G += lossl1 + loss_sam
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
                                                                self.epoch, lrD, 
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
                'D': self.D.module.state_dict(),
                'optimG': self.optimG.state_dict(),
                'optimD': self.optimD.state_dict(),
            }
        else:
            state = {
                'epoch': self.epoch,
                'iter': self.iteration,
                'G': self.G.state_dict(),
                'D': self.D.state_dict(),
                'optimG': self.optimG.state_dict(),
                'optimD': self.optimD.state_dict(),
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
            self.D.module.load_state_dict(checkpoint['D'])
        else:
            self.G.load_state_dict(checkpoint['G'])
            self.D.load_state_dict(checkpoint['D'])
        self.optimG.load_state_dict(checkpoint['optimG'])
        self.optimD.load_state_dict(checkpoint['optimD'])
        self.iteration = checkpoint['iter']
        self.epoch = checkpoint['epoch']
        try:
            self.load_metrics()
        except:
            pass
        print("pretrained model loaded, iteration: ", self.iteration)

if __name__ == '__main__':
    # input = torch.randn([1, 3, 128, 128]).cuda()
    # D = NLayerDiscriminator(34).cuda()
    # G = UnetGenerator(3, 31).cuda()
    
    # output = G(input)
    # print(output.shape)
    
    spec = pix2pixlsgan(opt, multiGPU=opt.multigpu)
    # spec.load_checkpoint()
    spec.train()