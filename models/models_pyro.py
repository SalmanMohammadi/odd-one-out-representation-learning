import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms
import pyro
import pyro.distributions as dist

class Encoder(nn.Module):
    # input - 64x64x(n_channels)
    def __init__(self, z_dim, n_channels):
        super().__init__()
        self.c1 = nn.Conv2d(n_channels, 32, 4, stride=2)
        self.c2 = nn.Conv2d(32, 64, 4, stride=2)
        self.c3 = nn.Conv2d(64, 64, 4, stride=2)
        self.c4 = nn.Conv2d(64, 256, 4, stride=2)
        self.fc5 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_logvar = nn.Linear(256, z_dim)

    def forward(self, x):
        for l in [self.c1, self.c2, self.c3, self.c4]:
            x = nn.functional.relu(l(x))
        x = self.fc5(x)
        mu, logvar = self.fc_mu(x), self.fc_sig(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, z_dim, n_channels):
        super().__init__()
        
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.c1 = nn.ConvTranspose2d(64, 32, 4, stride=2)
        self.c2 = nn.ConvTranspose2d(32, 32, 4, stride=2)
        self.c3 = nn.ConvTranspose2d(32, 32, 4, stride=2)
        self.c4 = nn.ConvTranspose2d(32, n_channels, 4, stride=2)

    def forward(self, z):
        for l in [self.fc1, self.fc2, self.c1, self.c2, self.c3]:
            x = nn.functional.relu(l(x))
        x = self.c4(x)
        mu, logvar = fc_mu(x), fc_sig(x)
        return mu, logvar

class AdaGVAE(nn.Module): 
    # architecture from Locatello et. al. http://arxiv.org/abs/2002.02886
    def __init__(self, z_dim=10, n_channels=3, use_cuda=True, adaptive=True):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, n_channels)
        self.decoder = Decoder(z_dim, n_channels)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

        self.adaptive = adaptive

    # define the model p(x1|z)p(x2|z)p(z)
    # possible extension to p(x1|z)p(x2|f(z,z',S))p(z)p(z')p(S)
    def model(self, data):
        assert len(data) == 3
        (x1, x2) = data
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x1.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x1.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing 
            # the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))

    # define the guide (i.e. variational distribution) q(z|x1), q(z|x2)
    # data - x1, x2, k
    def guide(self, data):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x1) and q(z|x2)
            z_loc_1, z_logvar_1 = self.encoder.forward(x1)
            z_loc_2, z_logvar_2 = self.encoder.forward(x2)
            # reparameterize
            z_scale_1 = z_logvar_1.exp().pow(1/2)
            z_scale_2 = z_logvar_2.exp().pow(1/2)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
    
    def tau(self, z_loc_1, z_scale_1, z_loc_2, z_scale_2):
        

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img


def get_dsprites(train_size=300000, test_size=10000, batch_size=64):
    """
    Returns train and test DSprites dataset.
    """
    dsprites_loader = dsprites.DSpritesLoader()
    train_data = DataLoader(dsprites.DSPritesIID(size=train_size, dsprites_loader=dsprites_loader),
                            batch_size=batch_size, pin_memory=True)
    test_data = DataLoader(dsprites.DSPritesIID(size=test_size, dsprites_loader=dsprites_loader),
                            batch_size=batch_size)                    
    return train_data, test_data
