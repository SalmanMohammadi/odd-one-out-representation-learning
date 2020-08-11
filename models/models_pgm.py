import math
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.distributions as dist
from itertools import islice
import numpy as np

CUDA = torch.device('cuda')

class CNNEmbedder(nn.Module):
    def __init__(self, n_channels):
        self.c1 = nn.Conv2d(n_channels, 32, kernel_size=4, stride=2)
        self.b1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.b2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.b3 = nn.BatchNorm2d(64)
        self.c4 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.b4 = nn.BatchNorm2d(64)
        

    def forward(self, x):
        for c, b in zip([self.c1, self.c2, self.c3, self.c4],
                        [self.b1, self.b2, self.b3, self.b4]):
            x = F.relu(b(c(x)))

class Relation(nn.Module):
    def __init__(self, ):
        