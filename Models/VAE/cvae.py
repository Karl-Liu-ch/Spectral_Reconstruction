import sys
sys.path.append('./')
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.optim as optim
import torch.backends.cudnn as cudnn
from hsi_dataset import TrainDataset, ValidDataset
from utils import AverageMeter, record_loss, Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_Fid, Loss_SAM, Loss_SSIM, reconRGB
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
from options import opt

class ConditionalVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 condition_channels: int,
                 latent_dim: int,
                 hidden_dims,
                 img_size:int = 128,
                 **kwargs) -> None:
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.in_channels = in_channels

        self.embed_class = nn.Conv2d(condition_channels, condition_channels, kernel_size=1)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512, 1024]
            
        self.hidden_dims = hidden_dims
        in_channels += condition_channels # To account for the extra label channel
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
            
        self.hidden_size = (self.img_size // (2 ** len(hidden_dims)))
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * (self.hidden_size ** 2), latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * (self.hidden_size ** 2), latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim + condition_channels * self.img_size ** 2, hidden_dims[-1] * (self.hidden_size ** 2))

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= self.in_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0], self.hidden_size, self.hidden_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, y):
        embedded_class = self.embed_class(y)
        embedded_input = self.embed_data(input)

        x = torch.cat([embedded_input, embedded_class], dim = 1)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)
        y = nn.Flatten()(y)

        z = torch.cat([z, y], dim = 1)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self, recons, input, mu, log_var):

        # kld_weight = 3e-3  # Account for the minibatch samples from the dataset
        kld_weight = 100 / 128 / 128  # Account for the minibatch samples from the dataset
        # kld_weight = 1  # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss * kld_weight}

    def sample(self, y):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        y = nn.Flatten()(y)
        z = torch.randn(y.shape[0], self.latent_dim)

        z = z.cuda()

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_sam = Loss_SAM().cuda()
criterion_fid = Loss_Fid().cuda()
criterion_ssim = Loss_SSIM().cuda()

class train_cvae():
    def __init__(self, opt) -> None:
        self.opt = opt
        self.model = ConditionalVAE(in_channels=31, condition_channels=3, latent_dim=1024, hidden_dims=[32, 64, 128, 256, 512], img_size=128).cuda()
        self.root = '/work3/s212645/Spectral_Reconstruction/checkpoint/cvae/'
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
        
        # iterations
        per_epoch_iteration = 1000
        self.total_iteration = per_epoch_iteration*opt.end_epoch
        self.iteration = 0
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.total_iteration, eta_min=1e-6)
        
        # make checkpoint dir
        if not os.path.exists(opt.outf):
            os.makedirs(opt.outf)
        if not os.path.exists(self.root):
            os.makedirs(self.root)
    
    def load_dataset(self):
        # load dataset
        print("\nloading dataset ...")
        self.train_data = TrainDataset(data_root=self.opt.data_root, crop_size=self.opt.patch_size, bgr2rgb=True, arg=True, stride=self.opt.stride)
        print(f"Iteration per epoch: {len(self.train_data)}")
        self.val_data = ValidDataset(data_root=self.opt.data_root, crop_size=self.opt.patch_size, bgr2rgb=True, arg=True, stride=64)
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
                [output, input, mu, log_var] = self.model(labels, images)
                Loss_dict = self.model.loss_function(output, input, mu, log_var)
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
                if self.iteration % 1000 == 0:
                    mrae_loss, rmse_loss, psnr_loss, sam_loss, fid_loss, ssim_loss = self.validate(val_loader)
                    print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}, SAM: {sam_loss}, FID: {fid_loss}, SSIM: {ssim_loss}')
                    # Save model
                    if torch.abs(mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or self.iteration % 5000 == 0:
                        print(f'Saving to {self.opt.outf}')
                        self.save_checkpoint()
                        if mrae_loss < record_mrae_loss:
                            record_mrae_loss = mrae_loss
                    # print loss
                    print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                        "Test RMSE: %.9f, Test PSNR: %.9f, SAM: %.9f, FID: %.9f, SSIM: %.9f " % (self.iteration, 
                                                                                                 self.iteration//1000, lr, 
                                                                                                 losses.avg, mrae_loss, rmse_loss, psnr_loss, sam_loss, 
                                                                                                 fid_loss, ssim_loss))
                
    def validate(self, val_loader):
        self.model.eval()
        losses_mrae = AverageMeter()
        losses_rmse = AverageMeter()
        losses_psnr = AverageMeter()
        losses_sam = AverageMeter()
        losses_fid = AverageMeter()
        losses_ssim = AverageMeter()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            with torch.no_grad():
                # compute output
                output = self.model.sample(input)
                loss_mrae = criterion_mrae(output, target)
                loss_rmse = criterion_rmse(output, target)
                loss_psnr = criterion_psnr(output, target)
                loss_sam = criterion_sam(output, target)
                rgb = reconRGB(output)
                loss_fid = criterion_fid(rgb, input)
                loss_ssim = criterion_ssim(rgb, input)
            # record loss
            losses_mrae.update(loss_mrae.data)
            losses_rmse.update(loss_rmse.data)
            losses_psnr.update(loss_psnr.data)
            losses_sam.update(loss_sam.data)
            losses_fid.update(loss_fid.data)
            losses_ssim.update(loss_ssim.data)
        return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg, losses_fid.avg, losses_ssim.avg
    
    def test(self, test_loader):
        pass
    
    def save_checkpoint(self):
        epoch = self.iteration // 1000
        state = {
            'epoch': epoch,
            'iter': self.iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        torch.save(state, os.path.join(self.root, 'net_epoch_beta1.pth'))
        
    def load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.root, 'net_epoch_beta1.pth'))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iteration = checkpoint['iter']
        print("pretrained model loaded")

if __name__ == '__main__':
    # model = ConditionalVAE(in_channels=31, condition_channels=3, latent_dim=1024, hidden_dims=[32, 64, 128, 256, 512], img_size=64).cuda()
    # input = torch.rand([1, 31, 64, 64]).cuda()
    # y = torch.rand([1, 3, 64, 64]).cuda()
    # [output, input, mu, logvar] = model(input, y)
    # loss = model.loss_function(output, input, mu, logvar)
    # print(loss['Reconstruction_Loss'].item())
    # print(loss['KLD'].item())
    spec = train_cvae(opt)
    spec.load_checkpoint()
    spec.train()