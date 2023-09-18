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

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()

class Generator(nn.Module):
    def __init__(self, num_input, num_output):
        super(Generator, self).__init__()
        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU())
            return layer

        self.Net = nn.Sequential(
            *Conv(num_input, 64),
            *Conv(64, 256),
            *Conv(256, 256),
            nn.Conv2d(256, num_output, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.Net(input)
        return output
    
class Discriminator(nn.Module):
    def __init__(self, input_nums):
        super(Discriminator, self).__init__()
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
    
class CWGAN():
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.G = Generator(6, 31).cuda() # torch.concat([z, image])
        self.D = Discriminator(34).cuda()
        per_epoch_iteration = 1000
        self.total_iteration = per_epoch_iteration*opt.end_epoch
        self.iteration = 0
        self.optimG = optim.Adam(self.G.parameters(), lr=self.opt.init_lr, betas=(0.9, 0.999))
        self.schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimG, self.total_iteration, eta_min=1e-6)
        self.optimD = optim.Adam(self.D.parameters(), lr=self.opt.init_lr, betas=(0.9, 0.999))
        self.schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimD, self.total_iteration, eta_min=1e-6)
        self.lossl1 = nn.L1Loss()
        self.lamda = 1
        self.lambda_term = 10.0
        self.criterion = nn.BCELoss()
        self.root = '/work3/s212645/Spectral_Reconstruction/checkpoint/cwgan/'
        if not os.path.exists(self.root):
            os.makedirs(self.root)
    
    def load_dataset(self):
        # load dataset
        print("\nloading dataset ...")
        self.train_data = TrainDataset(data_root=self.opt.data_root, crop_size=self.opt.patch_size, bgr2rgb=True, arg=True, stride=self.opt.stride)
        print(f"Iteration per epoch: {len(self.train_data)}")
        self.val_data = ValidDataset(data_root=self.opt.data_root, crop_size=self.opt.patch_size, bgr2rgb=True, arg=True, stride=64)
        print("Validation set samples: ", len(self.val_data))
    
    def calculate_gradient_penalty(self, real_labels, fake_labels, images):
        eta = torch.FloatTensor(real_labels.size(0),1,1,1).uniform_(0,1).cuda()
        eta = eta.expand(real_labels.size(0), real_labels.size(1), real_labels.size(2), real_labels.size(3))

        interpolated = eta * real_labels + ((1 - eta) * fake_labels)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        interpolated = torch.concat([images, interpolated], dim=1)
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty
    
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
                z = torch.randn_like(images).cuda()
                z = torch.concat([z, images], dim=1)
                z = Variable(z)
                x_fake = self.G(z)
                fakeAB = torch.concat([images, x_fake],dim=1)
                
                # train D
                for p in self.D.parameters():
                    p.requires_grad = True
                self.optimD.zero_grad()
                realAB = torch.concat([images, labels], dim=1)
                D_real = self.D(realAB)
                loss_real = -D_real.mean(0).view(1)
                loss_real.backward()
                D_fake = self.D(fakeAB.detach())
                loss_fake = D_fake.mean(0).view(1)
                loss_fake.backward()
                gp = self.calculate_gradient_penalty(labels.data, x_fake.data, images.data)
                gp.backward()
                self.optimD.step()
                self.schedulerD.step()
                
                # train G
                self.optimG.zero_grad()
                lrG = self.optimG.param_groups[0]['lr']
                for p in self.D.parameters():
                    p.requires_grad = False
                pred_fake = self.D(fakeAB)
                loss_G = -pred_fake.mean(0).view(1)
                lossl1 = self.lossl1(x_fake, labels) * self.lamda
                loss_G += lossl1
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
                if self.iteration % 1000 == 0:
                    mrae_loss, rmse_loss, psnr_loss = self.validate(val_loader)
                    print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}')
                    # Save model
                    if torch.abs(mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or self.iteration % 5000 == 0:
                        print(f'Saving to {self.root}')
                        self.save_checkpoint(best=True)
                        if mrae_loss < record_mrae_loss:
                            record_mrae_loss = mrae_loss
                    # print loss
                    print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                        "Test RMSE: %.9f, Test PSNR: %.9f " % (self.iteration, self.iteration//1000, lrG, losses.avg, mrae_loss, rmse_loss, psnr_loss))
                
    def validate(self, val_loader):
        self.G.eval()
        losses_mrae = AverageMeter()
        losses_rmse = AverageMeter()
        losses_psnr = AverageMeter()
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
            # record loss
            losses_mrae.update(loss_mrae.data)
            losses_rmse.update(loss_rmse.data)
            losses_psnr.update(loss_psnr.data)
        return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg
    
    def test(self, test_loader):
        pass
    
    def save_checkpoint(self, best = False):
        epoch = self.iteration // 1000
        state = {
            'epoch': epoch,
            'iter': self.iteration,
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'optimG': self.optimG.state_dict(),
            'optimD': self.optimD.state_dict(),
        }
        if best: 
            name = 'net_epoch_best.pth'
        else:
            name = 'net_%depoch.pth' % epoch
        torch.save(state, os.path.join(self.root, name))
        
    def load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.root, 'net_epoch_best.pth'))
        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'])
        self.optimG.load_state_dict(checkpoint['optimG'])
        self.optimD.load_state_dict(checkpoint['optimD'])
        self.iteration = checkpoint['iter']
        print("pretrained model loaded")

if __name__ == '__main__':
    # input = torch.randn([1, 6, 128, 128]).cuda()
    # D = Discriminator(31).cuda()
    # G = Generator(6, 31).cuda()
    
    # output = G(input)
    # print(output.shape)
    
    spec = CWGAN(opt)
    spec.train()