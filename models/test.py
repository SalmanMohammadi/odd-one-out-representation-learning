import torch
import models
import sys
sys.path.append("../data")
import dsprites_data as dsprites
from models import AdaGVAE, AdaTVAE
import matplotlib.pyplot as plt

device = torch.device("cuda")

dsprites_loader = dsprites.DSpritesLoader()
data = dsprites.DataLoader(dsprites.IterableDSpritesIIDTriplets(size=300000, dsprites_loader=dsprites_loader, k=1),
                            batch_size=1)
vae = AdaTVAE(n_channels=1)                            
data_ = next(iter(data))
vae.batch_forward(data_, torch.device('cuda'))

x1, x2, x3 = next(iter(data))
x1 = x1.to(device).reshape(64,1,64,64)
x2 = x2.to(device).reshape(64,1,64,64)
x3 = x3.to(device).reshape(64,1,64,64)

vae(x1, x2, x3)
x = vae.reconstruct_img(x1)
print(x.shape)
fig, axes = plt.subplots(2)
axes[0].imshow(x1[0].cpu().detach().squeeze(), cmap='Greys_r')
axes[1].imshow(x[0].cpu().detach().squeeze(), cmap='Greys_r')
axes[0].set_title("z")
axes[1].set_title("z_")
fig.subplots_adjust()
plt.show()