import sys
sys.path.append('./')
import numpy as np
from pyDOE import lhs
from scipy.optimize import minimize
from Models.GAN.SNCWGAN import *
import torch
from utils import *
from options import opt

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
    
class SNCWGAN_opt(Base):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.dim = 3
        self.model = SNCWGAN(opt, multiGPU=opt.multigpu)
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
        return mrae_loss + rmse_loss + 1 / psnr_loss + sam_loss
