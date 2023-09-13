import torch
from torch import nn
from torch.nn import functional as F

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
        kld_weight = 1024 / 128 / 128 / 31  # Account for the minibatch samples from the dataset
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

    # def generate(self, x, **kwargs):
    #     """
    #     Given an input image x, returns the reconstructed image
    #     :param x: (Tensor) [B x C x H x W]
    #     :return: (Tensor) [B x C x H x W]
    #     """

    #     return self.forward(x, **kwargs)[0]
    
if __name__ == '__main__':
    model = ConditionalVAE(in_channels=31, condition_channels=3, latent_dim=1024, hidden_dims=[32, 64, 128, 256, 512], img_size=64).cuda()
    input = torch.rand([1, 31, 64, 64]).cuda()
    y = torch.rand([1, 3, 64, 64]).cuda()
    [output, input, mu, logvar] = model(input, y)
    loss = model.loss_function(output, input, mu, logvar)
    print(loss['Reconstruction_Loss'].item())
    print(loss['KLD'].item())