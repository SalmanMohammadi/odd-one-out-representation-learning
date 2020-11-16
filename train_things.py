import torch
import argparse
import torchvision.datasets as dset
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import models.models_disentanglement as models
import numpy as np
from models.models_disentanglement import AdaGVAE, TVAE, AdaTVAE, VAE, TCVAE, KLTVAE
from data import things_data
import matplotlib.pyplot as plt
from eval import dci, triplets, bvae, factorvae
from torch.utils.tensorboard import SummaryWriter 

CUDA = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", default="tmp")
parser.add_argument("--model", type=str)
parser.add_argument("--train", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--save", action="store_true")
parser.add_argument("--steps", type=int, default=300000) 
parser.add_argument("--experiment_name", type=str, default='')
parser.add_argument("--experiment_id", type=int, default=0)
parser.add_argument("--load", action="store_true")
parser.add_argument("--gamma", type=int, default=1)
parser.add_argument("--alpha", type=float, default=1)
parser.add_argument("--warm_up_steps", type=int, default=-1)
parser.add_argument("--k", type=int, default=None)
parser.add_argument("--b", type=int, default=1)
args = parser.parse_args()
np.random.seed(args.experiment_id)

model_dict = {
    'adagvae': AdaGVAE, 
    'tvae': TVAE, 
    'adatvae': AdaTVAE, 
    'vae': VAE, 
    'tcvae': TCVAE,
    'kltvae': KLTVAE
}

if args.train and args.test:
    parser.error("Can't have both --train and --test")

if args.model not in model_dict.keys():
    parser.error("Specify model: one of: " + ", ".join(model_dict.keys()))

if args.warm_up_steps >= args.steps:
    parser.error("--warm_up_steps can't be greater than --steps (default 300000).")

experiment_id = '/' + str(args.experiment_id)
experiment_name = args.experiment_name if args.experiment_name else ''
model_path = args.logdir + '/' + args.model + '/' + experiment_name + experiment_id

label_dict = {
    'adatvae': ["recon_1", "recon_2", "recon_3", "kl_1", "kl_2", "kl_3", "tc", "tc_1", "tc_2"],
    'tvae': ["recon_1", "recon_2", "recon_3", "kl_1", "kl_2", "kl_3", "y", "y_"],
    'vae': ["recon", "kl"],
    # 'tvae': ["recon_1", "recon_2", "recon_3", "kl_1", "kl_2", "kl_3"],
    'adagvae': ["recon", "kl", "recon_1", "recon_2", "kl_1", "kl_2"],
    # : ["recon_1", "kl_1"]
    'tcvae': ["recon", "kl", "tc"],
    'kltvae': ["recon_1", "recon_2", "recon_3", "kl_1", "kl_2", "kl_3", "y", "y_"],\
}

vae = None
labels = label_dict[args.model]

train_data, test_data = things_data.get_things(seed=args.experiment_id)
vae = model_dict[args.model](n_channels=3, gamma=args.gamma, alpha=args.alpha, warm_up=args.warm_up_steps, b=args.b, use_things=True)

print(args)
writer = SummaryWriter(log_dir=model_path)
if not args.test:
    opt = optim.Adam(vae.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)
    # models.train_things(vae, train_data, opt, verbose=True, writer=writer,
    #             metrics_labels=labels, num_steps=args.steps)
    if args.save:
        torch.save(vae.state_dict(), model_path + ".pt")
        
if not args.train:
    if args.load:
        vae.load_state_dict(torch.load( model_path + ".pt"))
    # _, metrics = models.test(vae, test_data, verbose=True, metrics_labels=labels, 
    #                             writer=writer, experiment_id=args.experiment_id)

    with torch.no_grad():
        num_samples = 15
        x1, *_ = next(iter(test_data))
        x1 = x1.reshape(64, 3, 128, 128)
        fig, axes = plt.subplots(num_samples, 2, figsize=(15,15), sharex=True, sharey=True)
        x1_ = vae.reconstruct_img(x1.clone().to(CUDA)).cpu().detach()
        for i in range(15):
            img = x1[i].T
            img_ = x1_[i].T
            axes[i, 0].imshow(img.squeeze(), cmap="Greys_r")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(img_.squeeze(), cmap="Greys_r")
            axes[i, 1].axis('off')

        axes[0, 0].set_title("source")
        axes[0, 1].set_title("recon")
        plt.tight_layout()
        plt.axis('off')
        writer.add_figure('test/reconstructions', fig)

        print(len(test_data.dataset))

        # triplet metric
        print("TRIPLET---")
        scores = triplets.calculate_triplet_score(vae, dataset=test_data.dataset)
        for label, metric in scores.items():
            print(label, ":", metric)
            writer.add_scalar(label, metric, args.experiment_id)

writer.flush()
writer.close()

    # metrics_labels = ['hparam/'+x for x in config.model['metrics_labels']]
    # writer.add_hparams(hparam_dict=config.hparams, metric_dict=dict(zip(metrics_labels, metrics)))
