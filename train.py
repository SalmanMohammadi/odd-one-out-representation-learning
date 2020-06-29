import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms
import pyro
import pyro.distributions as dist
import models
from models import VAE, get_dsprites
from data import DSpritesLoader, DSPritesIID


vae = VAE()
