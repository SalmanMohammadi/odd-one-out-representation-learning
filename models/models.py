import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms
import torch.distributions as dist
import numpy as np

CUDA = torch.device('cuda')

class Encoder(nn.Module):
    # input - 64x64x(n_channels)
    def __init__(self, z_dim, n_channels):
        super().__init__()
        self.c1 = nn.Conv2d(n_channels, 32, 4, stride=2)
        self.c2 = nn.Conv2d(32, 32, 4, stride=2)
        self.c3 = nn.Conv2d(32, 64, 4, stride=2)
        self.c4 = nn.Conv2d(64, 64, 4, stride=2)
        self.fc5 = nn.Linear(256, 256)
        self.fc_loc = nn.Linear(256, z_dim)
        self.fc_logvar = nn.Linear(256, z_dim)

    def forward(self, x):
        for l in [self.c1, self.c2, self.c3, self.c4]:
            x = nn.functional.relu(l(x))
        x = x.view(-1, 256)
        x = self.fc5(x)
        loc, logvar = self.fc_loc(x), self.fc_logvar(x)
        return loc, logvar

class Decoder(nn.Module):
    def __init__(self, z_dim, n_channels):
        super().__init__()
        
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.c1 = nn.ConvTranspose2d(64, 64, 4, padding=1, stride=2)
        self.c2 = nn.ConvTranspose2d(64, 32, 4, padding=1, stride=2)
        self.c3 = nn.ConvTranspose2d(32, 32, 4, padding=1, stride=2)
        self.c4 = nn.ConvTranspose2d(32, n_channels, 4, padding=1, stride=2)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = x.view(-1, 64, 4, 4)
        for l in [self.c1, self.c2, self.c3]:
            x = nn.functional.relu(l(x))
        x = self.c4(x)
        return x

class AdaGVAE(nn.Module): 
    # architecture from Locatello et. al. http://arxiv.org/abs/2002.02886
    def __init__(self, z_dim=10, n_channels=3, use_cuda=True, adaptive=True, k=None):
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

        self.k = k
        self.adaptive = adaptive

    def sample(self, loc, logvar):
        std = torch.exp(0.5*logvar)
        p_z = torch.rand_like(std)
        return loc + p_z*std

    def forward(self, x1, x2):
        z_loc_1, z_logvar_1 = self.encoder(x1)
        z_loc_2, z_logvar_2 = self.encoder(x2)
        
        z_loc_1, z_logvar_1, z_loc_2, z_logvar_2 = self.average_posterior(z_loc_1, z_logvar_1, z_loc_2, z_logvar_2)

        z1 = self.sample(z_loc_1, z_logvar_1)
        z2 = self.sample(z_loc_2, z_logvar_2)

        x1_ = self.decoder(z1)
        x2_ = self.decoder(z2)

        return x1_, x2_, z_loc_1, z_logvar_1, z_loc_2, z_logvar_2

    def batch_forward(self, data, device):
        x1, x2, _ = data
        x1 = x1.to(device)
        x2 = x2.to(device)

        x1_, x2_, z_loc_1, z_logvar_1, z_loc_2, z_logvar_2 = self(x1, x2)

        return self.loss(x1, x2, x1_, x2_, z_loc_1, z_logvar_1, z_loc_2, z_logvar_2)

    # p(z1, z2 | x1, x2) = p(z1 | x1)p(z2 | x2)
    # averages aggregate posterior according to GVAE strategy in 
    # https://www.ijcai.org/Proceedings/2019/0348.pdf
    def average_posterior(self, z_loc_1, z_logvar_1, z_loc_2, z_logvar_2):
        z_x1 = dist.Normal(z_loc_1, z_logvar_1.exp().pow(1/2))
        z_x2 = dist.Normal(z_loc_2, z_logvar_2.exp().pow(1/2))
        
        # taking mean here - might take sum
        dim_kl = dist.kl.kl_divergence(z_x1, z_x2)
        tau = 0.5 * (torch.max(dim_kl, 1)[0][:,None] + torch.min(dim_kl, 1)[0][:,None])

        z_loc_1 = torch.where(dim_kl < tau, 0.5*(z_loc_1+z_loc_2), z_loc_1)
        z_loc_2 = torch.where(dim_kl < tau, 0.5*(z_loc_1+z_loc_2), z_loc_2)
        
        z_logvar_1 = torch.where(dim_kl < tau, 0.5*(z_logvar_1+z_logvar_2), z_logvar_1)
        z_logvar_2 = torch.where(dim_kl < tau, 0.5*(z_logvar_1+z_logvar_2), z_logvar_2)

        return z_loc_1, z_logvar_1, z_loc_2, z_logvar_2

    def loss(self, x1, x2, x1_, x2_, z_loc_1, z_logvar_1, z_loc_2, z_logvar_2):
        # returns total_loss, recon_1, recon_2, kl_1, kl_2
        # reconstruction loss
        # print(x1_, x1)
        r_1 = nn.functional.binary_cross_entropy_with_logits(x1_, x1, reduction='sum')
        r_2 = nn.functional.binary_cross_entropy_with_logits(x2_, x2, reduction='sum')

        # analytical kl divergences - look into mean?
        kl_1 = -0.5 * torch.sum(1 + z_logvar_1 - z_loc_1.pow(2) - z_logvar_1.exp())
        kl_2 = -0.5 * torch.sum(1 + z_logvar_2 - z_loc_2.pow(2) - z_logvar_2.exp())

        return r_1 + r_2 + kl_1 + kl_2, r_1, r_2, kl_1, kl_2

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_logvar = self.encoder(x)
        # sample in latent space
        z = self.sample(z_loc, z_logvar)
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img


def train(model, dataset, epoch, optimizer, device=CUDA, verbose=True, writer=None, log_interval=100, metrics_labels=None):
    """
    Trains the model for a single 'epoch' on the data
    """
    model.train()
    train_loss = 0
    metrics_mean = []
    dataset_len = len(dataset) * dataset.batch_size
    for batch_id, data in enumerate(dataset):
        optimizer.zero_grad()
        loss, (*metrics) = model.batch_forward(data, device=device)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        metrics_mean.append([x.item() for x in metrics])
        
        data_len = len(data[0])
        if batch_id % log_interval == 0 and verbose:
            print('Train Epoch: {}, batch: {}, loss: {}'.format(
                epoch, batch_id, loss.item() / data_len))
            metrics = [x.item()/data_len for x in metrics]
            if metrics_labels:
                print(", ".join(list(map(lambda x: "%s: %.5f" % x, zip(metrics_labels, metrics)))))
            else:
                print(metrics)

    metrics_mean = np.array(metrics_mean)
    metrics_mean = np.sum(metrics_mean, axis=0)/dataset_len

    if writer:
        writer.add_scalar('train/loss', train_loss /dataset_len, epoch)
        for label, metric in zip(metrics_labels, metrics_mean):
            writer.add_scalar('train/'+label, metric, epoch)
    if verbose:
        print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / dataset_len))

def test(model, dataset, verbose=True, device=CUDA, metrics_labels=None, writer=None):
    """
    Evaluates the model
    """
    model.eval()
    test_loss = 0
    metrics_mean = []
    with torch.no_grad():
        for batch_id, data in enumerate(dataset):
            loss, (*metrics) = model.batch_forward(data, device=device)
            metrics_mean.append([x.item() for x in metrics])
            test_loss += loss.item()

    test_loss /= len(dataset.dataset)
    metrics_mean = np.array(metrics_mean)
    metrics_mean = np.sum(metrics_mean, axis=0)/len(dataset.dataset)
    # metrics = [x.item()/len(dataset.dataset) for x in metrics]
    if verbose:
        if metrics_labels:
            print(", ".join(list(map(lambda x: "%s: %.5f" % x, zip(metrics_labels, metrics_mean)))))
        print("Eval: ", test_loss)
    if writer:
        writer.add_scalar('test/loss', test_loss)
        for label, metric in zip(metrics_labels, metrics_mean):
            writer.add_scalar('test/'+label, metric)
    return test_loss, metrics_mean