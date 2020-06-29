import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from torch.utils.data import Dataset, DataLoader, RandomSampler, Sampler

class DSpritesLoader():
    def __init__(self, npz_path="./DSPRITES/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"):
        with np.load(npz_path, allow_pickle=True, encoding='latin1') as dataset_zip:
            self.metadata = dataset_zip['metadata'][()]
            self.X = np.reshape(dataset_zip['imgs'], (-1, 4096))
            self.Y = dataset_zip['latents_values']
            self.Y[:, 3] /= 2 * math.pi
            self.Y[:, 1] -= 1
        
class DSpritesIIDPairs(Dataset):
    def __init__(self, dsprites_loader, size=10000, k=None):

        self.size = size
        self.k = k
        self.dsprites_loader = dsprites_loader
        self.metadata = self.dsprites_loader.metadata
        self.latents_sizes = self.metadata['latents_sizes']
        if self.k:
            assert self.k in range(1, len(self.latents_sizes)-1)

        # An array to convert latent indices to indices in imgs
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                                np.array([1,])))

        z, z_, self.k_idx = self.sample_latent_pairs()
        self.z_indices = self.latent_to_index(z)
        self.z_hat_indices = self.latent_to_index(z_)

    def __len__(self):
        return self.size

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)
    
    # legacy single-observation sampling
    def sample_latent(self):
        samples = np.zeros((self.size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=self.size)

        return samples

    # sample pairs from ground truth generative model according to https://arxiv.org/pdf/2002.02886.pdf
    # locatello et. al. 2020 section 5.
    def sample_latent_pairs(self):
        z = np.zeros((self.size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            z[:, lat_i] = np.random.randint(lat_size, size=self.size)
        # sample k factors of variation which should not be shared between x1,x2
        # k ~ unif(1, d-1)
        if not self.k:
            k_samples = np.random.randint(1, (self.latents_sizes.size-1), size=self.size)
        else:
            k_samples = np.ones((self.size), dtype=int) * self.k

        k_idxs = [np.random.choice(np.arange(1, self.latents_sizes.size), size=k_, replace=False) 
                    for k_ in k_samples]
        z_ = np.array(z)
        for i, idx in enumerate(k_idxs):
            for j, lat_size in zip(idx, self.latents_sizes[idx]):
                z_[i, j] = np.random.randint(lat_size)
        return z, z_, k_idxs
        

    # returns pair of observations and indices of generative factors which 
    # differ between them
    def __getitem__(self, idx):
        z_idx = self.z_indices[idx]
        z_hat_idx = self.z_hat_indices[idx]

        X = np.array(self.dsprites_loader.X[z_idx], dtype=np.float32)
        X_ = np.array(self.dsprites_loader.X[z_hat_idx], dtype=np.float32)

        K = np.array(self.k_idx[idx], dtype=np.float32)
        Y_new = np.array(self.dsprites_loader.Y[idx], dtype=np.long)

        return (X, X_, K, Y_new.squeeze())

if __name__ == "__main__":
    dsprites_loader = DSpritesLoader()
    data = DataLoader(DSpritesIIDPairs(size=10000, dsprites_loader=dsprites_loader),
                                batch_size=1)
    x1, x2, k, *_ = next(iter(data))
    print(k)
    fig, axes = plt.subplots(2)
    axes[0].imshow(x1.reshape(64,64), cmap='Greys_r')
    axes[1].imshow(x2.reshape(64,64), cmap='Greys_r')
    axes[0].set_title("z")
    axes[1].set_title("z_")
    fig.subplots_adjust()
    plt.show()