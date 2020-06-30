import torch
import argparse
import torchvision.datasets as dset
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import models
from models import AdaGVAE
from data import dsprites_data as dsprites
import matplotlib.pyplot as plt

CUDA = torch.device('cuda')
# config_mappings = {
#     'vae': configs.vanilla_vae,
#     'isvae': configs.isvae,
#     'lisvae': configs.lisvae,
# }

# def hparams_to_dict(kv):
#     if kv:
#         res = lambda kv: dict(list(map(lambda x: (x.split("=")[0], eval(x.split("=")[1])), kv.split(","))))
#         return res(kv)
#     else:
#         return {}

# parser = argparse.ArgumentParser()
# parser.add_argument("--model", type=str)
# parser.add_argument("--experiment_id", type=int, default=0)
# parser.add_argument("--experiment_name", type=str, default='')
# parser.add_argument("--hparams", type=hparams_to_dict, default='')
# parser.add_argument("--logging", type=str)
# args = parser.parse_args()

# experiment_id = '/' + str(args.experiment_id)
# experiment_name = '/' + args.experiment_name if args.experiment_name else ''
# model_path = 'tmp/' + args.model + experiment_name + experiment_id


# config = config_mappings[args.model]
# config = config._replace(hparams={**config.hparams, **args.hparams})

# print(config, model_path)
labels = ["recon_1", "recon_2", "kl_1", "kl_2"]
train_data, test_data = dsprites.get_dsprites(train_size=737280, test_size=10000, batch_size=64, k=1)
vae = AdaGVAE(n_channels=1)

opt = optim.Adam(vae.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)

# # writer = SummaryWriter(log_dir=model_path)
for epoch in range(26):
    models.train(vae, train_data, epoch, opt, verbose=True, 
            metrics_labels=labels)

_, metrics = models.test(vae, test_data, verbose=True, metrics_labels=labels)

with torch.no_grad():
    num_samples = 15
    x1, x2, _ = next(iter(test_data))
    x1 = x1[:15]
    x2 = x2[:15]
    fig, axes = plt.subplots(num_samples, 2, figsize=(15,15))
    for i in range(15):
        axes[i, 0].imshow(x1[i].squeeze(), cmap="Greys_r")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(x2[i].squeeze(), cmap="Greys_r")
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.axis('off')

    fig, axes = plt.subplots(num_samples, 2, figsize=(15,15))
    x1 = x1.to(CUDA)
    x2 = x2.to(CUDA)
    x1_ = vae.reconstruct_img(x1).cpu().detach()
    x2_ = vae.reconstruct_img(x2).cpu().detach()
    for i in range(15):
        axes[i, 0].imshow(x1_[i].squeeze(), cmap="Greys_r")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(x2_[i].squeeze(), cmap="Greys_r")
        axes[i, 1].axis('off')
    plt.title("recon")
    plt.tight_layout()
    plt.axis('off')
    plt.show()
# torch.save(model.state_dict(), model_path + ".pt")
# metrics_labels = ['hparam/'+x for x in config.model['metrics_labels']]
# writer.add_hparams(hparam_dict=config.hparams, metric_dict=dict(zip(metrics_labels, metrics)))