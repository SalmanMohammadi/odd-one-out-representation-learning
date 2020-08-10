import math
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms
import torch.distributions as dist
from itertools import islice
import numpy as np

CUDA = torch.device('cuda')

class CNNEmbedder(nn.Module):
    def __init__(self, n_channels):
        self.c1 = nn.Conv2d(n_channels, 32, kernel_size=4, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.c4 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(64)
        # self.fc = nn.Linear(32*4*4, 256)