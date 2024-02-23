import sys
sys.path.append('./')
import numpy as np
from pyDOE import lhs
from scipy.optimize import minimize
from Models.GAN.SNCWGAN import *
import torch
from utils import *
from options import opt
from dataset.datasets import TrainDataset, ValidDataset, TestDataset

class Base(object):
    
    def __init__(self):
        self.y = None
        self.bounds = None
        self.dim = None
            
    def __call__(self, x):
        x = np.array(x, ndmin=2)
        y = - np.apply_along_axis(lambda x: self.evaluate(x), 1, x)
        self.y = np.squeeze(y)
        return self.y
    
    def is_feasible(self, x):
        x = np.array(x, ndmin=2)
        if self.y is None:
            self.y = self.__call__(x)
        feasibility = np.logical_not(np.isnan(self.y))
        return feasibility
    
    def synthesize(self, x):
        pass

    def evaluate(self):
        pass
    
    def sample_design_variables(self, n_sample, method='random'):
        if method == 'lhs':
            x = lhs(self.dim, samples=n_sample, criterion='cm')
            x = x * (self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:,0]
        else:
            # x = np.random.uniform(self.bounds[:,0], self.bounds[:,1], size=(n_sample, self.dim))
            x = np.random.normal(size=(n_sample, self.dim))
        return np.squeeze(x)
    
    def sample_airfoil(self, n_sample, method='random'):
        x = self.sample_design_variables(n_sample, method)
        airfoils = self.synthesize(x)
        return airfoils

class SNCWGAN_BO(SNCWGAN):
    def load_dataset(self):
        # load dataset
        print("\nloading dataset ...")
        self.train_data = TrainDataset(data_root=self.opt.data_root, crop_size=self.opt.patch_size, valid_ratio = 0.1, test_ratio=0.1)
        print(f"Iteration per epoch: {len(self.train_data)}")
        self.val_data = ValidDataset(data_root=self.opt.data_root, crop_size=self.opt.patch_size, valid_ratio = 0.1, test_ratio=0.1)
        print("Validation set samples: ", len(self.val_data))
    
    def train(self):
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
                # D_real, D_real_feature = self.D(realAB)
                x_fake = self.G(z)
                fakeAB = torch.concat([images, x_fake],dim=1)
                
                # train D
                for p in self.D.parameters():
                    p.requires_grad = True
                self.optimD.zero_grad()
                D_real, D_real_feature = self.D(realAB)
                loss_real = -D_real.mean(0).view(1)
                loss_real.backward(retain_graph = True)
                D_fake, _ = self.D(fakeAB.detach())
                loss_fake = D_fake.mean(0).view(1)
                loss_fake.backward()
                self.optimD.step()
                
                # train G
                self.optimG.zero_grad()
                lrG = self.optimG.param_groups[0]['lr']
                for p in self.D.parameters():
                    p.requires_grad = False
                pred_fake, D_fake_feature = self.D(fakeAB)
                loss_G = -pred_fake.mean(0).view(1)
                lossl1 = self.lossl1(x_fake, labels) * self.lamda
                losssam = SAM(x_fake, labels) * self.lambdasam
                perceptual_loss = 0
                for k in range(len(D_fake_feature)):
                    perceptual_loss += nn.MSELoss()(D_real_feature[k].detach(), D_fake_feature[k])
                loss_G += lossl1 + losssam + perceptual_loss * self.lambdaperceptual
                # train the generator
                loss_G.backward()
                self.optimG.step()
                
                loss_mrae = criterion_mrae(x_fake, labels)
                losses.update(loss_mrae.data)
                self.iteration = self.iteration+1
                if self.iteration % 20 == 0:
                    print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f'
                        % (self.epoch, self.end_epoch, lrG, losses.avg))
            self.epoch += 1
            self.schedulerD.step()
            self.schedulerG.step()

class SNCWGAN_opt(Base):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.dim = 3
        self.model = SNCWGAN_BO(opt, multiGPU=opt.multigpu)
        self.model.root = '/work3/s212645/Spectral_Reconstruction/checkpoint/SNCWGAN/BO/'
        self.bounds = np.array([[0., 3.]])
        self.bounds = np.tile(self.bounds, [self.dim, 1])
        self.val_loader = DataLoader(dataset=self.model.val_data, batch_size=self.opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def synthesize(self, x):
        lambdal1 = 10 ** x[0]
        lambdasam = 10 ** x[1]
        lambdaperceptual = 10 ** x[2]
        self.model.end_epoch = 10
        self.model.lamda = lambdal1
        self.model.lambdasam = lambdasam
        self.model.lambdaperceptual = lambdaperceptual
        return lambdal1, lambdasam, lambdaperceptual
    
    def evaluate(self, x):
        self.synthesize(x)
        self.model.train()
        mrae_loss, rmse_loss, psnr_loss, sam_loss, sid_loss = self.model.validate(self.val_loader)
        self.model = SNCWGAN_BO(opt, multiGPU=opt.multigpu)
        self.model.root = '/work3/s212645/Spectral_Reconstruction/checkpoint/SNCWGAN/BO/'
        return (mrae_loss + rmse_loss + 1 / psnr_loss + sam_loss).detach().cpu().numpy()
