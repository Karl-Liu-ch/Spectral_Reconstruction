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
from utils import AverageMeter, record_loss, Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_Fid, \
    Loss_SAM, Loss_SSIM, reconRGB, Loss_SID, SAM
from options import opt
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
import functools
from Models.GAN.networks import *
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
if opt.multigpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_sam = Loss_SAM()
criterion_sid = Loss_SID()
# criterion_fid = Loss_Fid().cuda()
# criterion_ssim = Loss_SSIM().cuda()
class Log_loss(nn.Module):
    def __init__(self):
        # negation is true when you minimize -log(val)
        super(Log_loss, self).__init__()
       
    def forward(self, x, negation=True):
        # shape of x will be [batch size]
        log_val = torch.log(x)
        loss = torch.sum(log_val)
        if negation:
            loss = torch.neg(loss)
        return loss
    
class Itself_loss(nn.Module):
    def __init__(self):
        super(Itself_loss, self).__init__()
        
    def forward(self, x, negation=True):
        # shape of x will be [batch size]
        loss = torch.sum(x)
        if negation:
            loss = torch.neg(loss)
        return loss
    
class D2GAN():
    def __init__(self, opt, multiGPU = False) -> None:
        super().__init__()
        self.multiGPU = multiGPU
        self.opt = opt
        self.G = ResnetGenerator(6, 31)
        self.D1 = SN_Discriminator(34)
        self.D2 = SN_Discriminator(34)
        if self.multiGPU:
            self.G = nn.DataParallel(self.G)
            self.D1 = nn.DataParallel(self.D1)
            self.D2 = nn.DataParallel(self.D2)
        self.G.cuda()
        self.D1.cuda()
        self.D2.cuda()
        
        self.epoch = 0
        self.end_epoch = opt.end_epoch
        per_epoch_iteration = 1000
        self.total_iteration = per_epoch_iteration*opt.end_epoch
        self.iteration = 0
        
        self.optimG = optim.Adam(self.G.parameters(), lr=self.opt.init_lr, betas=(0.9, 0.999))
        self.schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimG, self.total_iteration, eta_min=1e-6)
        self.optimD1 = optim.Adam(self.D1.parameters(), lr=self.opt.init_lr, betas=(0.9, 0.999))
        self.schedulerD1 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimD1, self.total_iteration, eta_min=1e-6)
        self.optimD2 = optim.Adam(self.D2.parameters(), lr=self.opt.init_lr, betas=(0.9, 0.999))
        self.schedulerD2 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimD2, self.total_iteration, eta_min=1e-6)
        self.lossl1 = nn.L1Loss()
        self.lamda = 100
        self.lambdasam = 0
        self.alpha = 0.2
        self.beta = 0.1
        self.criterion_itself = Itself_loss()
        self.criterion_log = Log_loss()
        self.root = '/work3/s212645/Spectral_Reconstruction/checkpoint/D2GAN/'
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
            self.D1.train()
            self.D2.train()
            losses = AverageMeter()
            train_loader = DataLoader(dataset=self.train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=2,
                                    pin_memory=True, drop_last=True)
            val_loader = DataLoader(dataset=self.val_data, batch_size=self.opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            for i, (images, labels) in enumerate(train_loader):
                labels = labels.cuda()
                images = images.cuda()
                images = Variable(images)
                labels = Variable(labels)
                z = torch.randn_like(images).cuda()
                z = torch.concat([z, images], dim=1)
                z = Variable(z)
                realAB = torch.concat([images, labels], dim=1)
                x_fake = self.G(z)
                fakeAB = torch.concat([images, x_fake],dim=1)
                
                # train D
                for p in self.D1.parameters():
                    p.requires_grad = True
                for p in self.D2.parameters():
                    p.requires_grad = True
                self.optimD1.zero_grad()
                self.optimD2.zero_grad()
                D1_real = self.D1(realAB)
                D2_real = self.D2(realAB)
                D1loss_real = self.alpha * self.criterion_log(D1_real)
                D1loss_real.backward()
                D2loss_real = self.criterion_itself(D2_real, False)
                D2loss_real.backward()
                
                D1_fake = self.D1(fakeAB.detach())
                D2_fake = self.D2(fakeAB.detach())
                D1loss_fake = self.criterion_itself(D1_fake, False)
                D1loss_fake.backward()
                D2loss_fake = self.beta * self.criterion_log(D2_fake)
                D2loss_fake.backward()
                self.optimD1.step()
                self.schedulerD1.step()
                self.optimD2.step()
                self.schedulerD2.step()
                
                # train G
                self.optimG.zero_grad()
                lrG = self.optimG.param_groups[0]['lr']
                for p in self.D1.parameters():
                    p.requires_grad = False
                for p in self.D2.parameters():
                    p.requires_grad = False
                    
                pred_fake1 = self.D1(fakeAB)
                pred_fake2 = self.D2(fakeAB)
                errG1 = self.criterion_itself(pred_fake1)
                errG2 = self.criterion_log(pred_fake2, False)
                loss_G = errG2 * self.beta + errG1
                lossl1 = self.lossl1(x_fake, labels) * self.lamda
                losssam = SAM(x_fake, labels) * self.lambdasam
                loss_G += lossl1 + losssam
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
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            z = torch.randn_like(input).cuda()
            z = torch.concat([z, input], dim=1)
            z = Variable(z)
            with torch.no_grad():
                # compute output
                output = self.G(z)
                loss_mrae = criterion_mrae(output, target)
                loss_rmse = criterion_rmse(output, target)
                loss_psnr = criterion_psnr(output, target)
                loss_sam = criterion_sam(output, target)
                loss_sid = criterion_sid(output, target)
            # record loss
            losses_mrae.update(loss_mrae.data)
            losses_rmse.update(loss_rmse.data)
            losses_psnr.update(loss_psnr.data)
            losses_sam.update(loss_sam.data)
            losses_sid.update(loss_sid.data)
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
                'D1': self.D1.module.state_dict(),
                'D2': self.D2.module.state_dict(),
                'optimG': self.optimG.state_dict(),
                'optimD1': self.optimD1.state_dict(),
                'optimD2': self.optimD2.state_dict(),
            }
        else:
            state = {
                'epoch': self.epoch,
                'iter': self.iteration,
                'G': self.G.state_dict(),
                'D1': self.D1.state_dict(),
                'D2': self.D2.state_dict(),
                'optimG': self.optimG.state_dict(),
                'optimD1': self.optimD1.state_dict(),
                'optimD2': self.optimD2.state_dict(),
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
            self.D1.module.load_state_dict(checkpoint['D1'])
            self.D2.module.load_state_dict(checkpoint['D2'])
        else:
            self.G.load_state_dict(checkpoint['G'])
            self.D1.load_state_dict(checkpoint['D1'])
            self.D2.load_state_dict(checkpoint['D2'])
        self.optimG.load_state_dict(checkpoint['optimG'])
        self.optimD1.load_state_dict(checkpoint['optimD1'])
        self.optimD2.load_state_dict(checkpoint['optimD2'])
        self.iteration = checkpoint['iter']
        self.epoch = checkpoint['epoch']
        try:
            self.load_metrics()
        except:
            pass
        print("pretrained model loaded, iteration: ", self.iteration)

if __name__ == '__main__':
    # input = torch.randn([1, 6, 128, 128]).cuda()
    # D = Discriminator(31).cuda()
    # G = Generator(6, 31).cuda()
    
    # output = G(input)
    # print(output.shape)
    
    spec = D2GAN(opt, multiGPU=opt.multigpu)
    if opt.loadmodel:
        try:
            spec.load_checkpoint()
        except:
            print('pretrained model loading failed')
    spec.train()