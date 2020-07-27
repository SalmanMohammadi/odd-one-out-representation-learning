import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from torch.utils.data import Dataset, IterableDataset, DataLoader, RandomSampler, Sampler
import itertools

class DSpritesLoader():
    def __init__(self, npz_path="./data/DSPRITES/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"):
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

        X = torch.tensor(self.dsprites_loader.X[z_idx], dtype=torch.float32).view(-1, 64, 64)
        X_ = torch.tensor(self.dsprites_loader.X[z_hat_idx], dtype=torch.float32).view(-1, 64, 64)

        # cant return irregular tensors
        # K = np.array(self.k_idx[idx], dtype=np.float32)
        Y_new = torch.tensor(self.dsprites_loader.Y[idx], dtype=torch.long)

        return X, X_, Y_new.squeeze()

class IterableDSPritesIID(IterableDataset):
    # unused parameter k, just hacking the constructor to be the same
    def __init__(self, dsprites_loader, size=10000, batch_size=16, k=None):

        self.batch_size = batch_size
        self.dsprites_loader = dsprites_loader
        self.num_batches = size
        self.metadata = self.dsprites_loader.metadata
        self.latents_sizes = self.metadata['latents_sizes']
        # An array to convert latent indices to indices in imgs
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                                np.array([1,])))

    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        for i in range(self.num_batches):
            yield self.sample()

    def sample(self):
        z = self.sample_latent()
        z_idx = self.latent_to_index(z)

        X = torch.tensor(self.dsprites_loader.X[z_idx], dtype=torch.float32).view(-1, 64, 64)
        return X, z[:,1:]
    
    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)
    
    def sample_latent(self):
        samples = np.zeros((self.batch_size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=self.batch_size)

        return samples

class IterableDSpritesIIDPairs(IterableDataset):
    def __init__(self, dsprites_loader, size=300000, batch_size=64, k=None):

        self.k = k
        self.batch_size = batch_size
        self.num_batches = size
        self.dsprites_loader = dsprites_loader
        self.metadata = self.dsprites_loader.metadata
        self.latents_sizes = self.metadata['latents_sizes']
        if self.k:
            assert self.k in range(1, len(self.latents_sizes)-1)

        # An array to convert latent indices to indices in imgs
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                                np.array([1,])))
    # getter for latent sizes
    def latent_sizes(self):
        return self.latents_sizes

    # generative random sampler
    def sample(self):
        z, z_, k_idx = self.sample_latent_pairs()
        z_idx = self.latent_to_index(z)
        z_hat_idx = self.latent_to_index(z_)

        X = torch.tensor(self.dsprites_loader.X[z_idx], dtype=torch.float32).view(-1, 64, 64)
        X_ = torch.tensor(self.dsprites_loader.X[z_hat_idx], dtype=torch.float32).view(-1, 64, 64)

        # cant return irregular tensors
        # K = np.array(self.k_idx[idx], dtype=np.float32)
        return X, X_, (z[:,1:], z_) 
        
    def __iter__(self):
        for i in range(self.num_batches):
            yield self.sample()

    def __len__(self):
        return self.num_batches

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
        z = np.zeros((self.batch_size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            z[:, lat_i] = np.random.randint(lat_size, size=self.batch_size)
        # sample k factors of variation which should not be shared between x1,x2
        # k ~ unif(1, d-1)
        if not self.k:
            k_samples = np.random.randint(1, (self.latents_sizes.size-1), size=self.batch_size)
        else:
            k_samples = np.ones((self.batch_size), dtype=int) * self.k

        k_idxs = [np.random.choice(np.arange(1, self.latents_sizes.size), size=k_, replace=False) 
                    for k_ in k_samples]
        z_ = np.array(z)
        for i, idx in enumerate(k_idxs):
            for j, lat_size in zip(idx, self.latents_sizes[idx]):
                z_[i, j] = np.random.randint(lat_size)
        return z, z_, k_idxs

class IterableDSpritesIIDTriplets(IterableDSpritesIIDPairs):
    # generative random sampler
    def sample(self):
        z_1, z_2, z_3, k_idxs = self.sample_latent_triplets()
        z_1_idx, z_2_idx, z_3_idx = map(self.latent_to_index, (z_1, z_2, z_3))
        
        X_1 = torch.tensor(self.dsprites_loader.X[z_1_idx], dtype=torch.float32).view(-1, 64, 64)
        X_2 = torch.tensor(self.dsprites_loader.X[z_2_idx], dtype=torch.float32).view(-1, 64, 64)
        X_3 = torch.tensor(self.dsprites_loader.X[z_3_idx], dtype=torch.float32).view(-1, 64, 64)

        # cant return irregular tensors
        # K = np.array(self.k_idx[idx], dtype=np.float32)
        return X_1, X_2, X_3
        
    def __iter__(self):
        for i in range(self.num_batches):
            yield self.sample()

    def __len__(self):
        return self.num_batches

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
    def sample_latent_triplets(self):
        z_1 = np.zeros((self.batch_size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            z_1[:, lat_i] = np.random.randint(lat_size, size=self.batch_size)
        # sample k factors of variation which should not be shared between x1,x2
        # but will be the only factor shared between x1,x3 and x2,x3
        # k ~ unif(1, d-1)
        if not self.k:
            k_samples = np.random.randint(1, (self.latents_sizes.size-1), size=self.batch_size)
        elif self.k == 1:
            # k_samples = np.ones((self.batch_size), dtype=int)
            k_idxs = np.random.randint(1, self.latents_sizes.size, size=self.batch_size)
            lat_range = list(range(1, self.latents_sizes.size))
            k_idxs_1 = [lat_range[:k_]+lat_range[k_+1:] for k_ in k_idxs]
        else:
            k_samples = np.ones((self.batch_size), dtype=int) * self.k

        z_2 = np.array(z_1)
        z_3 = np.array(z_1)
        for i, (idx, idx_1) in enumerate(zip(k_idxs[:, None], k_idxs_1)):
            for j, lat_size in zip(idx, self.latents_sizes[idx]):
                z_2[i, j] = np.random.randint(lat_size)
            for j, lat_size in zip(idx_1, self.latents_sizes[idx_1]):
                z_3[i, j] = np.random.randint(lat_size)
        return z_1, z_2, z_3, k_idxs
    
    def sample_latent_pairs(self):
        return NotImplemented

        
def get_dsprites(train_size=300000, test_size=10000, batch_size=64, k=None, dataset=DSpritesIIDPairs):
    """
    Returns train and test DSprites dataset.
    """
    dsprites_loader = DSpritesLoader(npz_path='./data/DSPRITES/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    train_data = DataLoader(dataset(size=train_size, dsprites_loader=dsprites_loader, k=k),
                            batch_size=batch_size)#, pin_memory=True, num_workers=16)
    test_data = DataLoader(dataset(size=test_size, dsprites_loader=dsprites_loader, k=k),
                            batch_size=batch_size)#, pin_memory=True, num_workers=16)                    
    return train_data, test_data

if __name__ == "__main__":
    testing_infinite = True
    testing_pairs = False
    if testing_infinite:
        if testing_pairs:
            dsprites_loader = DSpritesLoader(npz_path='./DSPRITES/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
            data = DataLoader(IterableDSpritesIIDPairs(dsprites_loader=dsprites_loader, batch_size=64),
                                        batch_size=1)
            x1, x2, (z, z_) = next(iter(data))
            x1 = x1.reshape(64, 1, 64, 64)
            print(z.shape)
            print(x1.shape)
            print(x2.shape)
            x1 = x1[0][0]
            x2 = x2[0][0]
            fig, axes = plt.subplots(2)
            axes[0].imshow(x1.view(64,64), cmap='Greys_r')
            axes[1].imshow(x2.view(64,64), cmap='Greys_r')
            axes[0].set_title("z")
            axes[1].set_title("z_")
            fig.subplots_adjust()
            plt.show()
        else:
            dsprites_loader = DSpritesLoader(npz_path='./DSPRITES/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
            data = DataLoader(IterableDSpritesIIDTriplets(dsprites_loader=dsprites_loader, batch_size=5, k=1),
                                        batch_size=1)
            x1, x2, x3 = next(iter(data))
            print(x1.shape)
            x1 = x1.reshape(5, 64, 64)
            x2 = x2.reshape(5, 64, 64)
            x3 = x3.reshape(5, 64, 64)
            fig, axes = plt.subplots(5, 3, sharex=True, sharey=True)
            for i in range(5):
                axes[i][0].imshow(x1[i], cmap='Greys_r')
                axes[i][1].imshow(x2[i], cmap='Greys_r')
                axes[i][2].imshow(x3[i], cmap='Greys_r')
            axes[0][0].set_title("A")
            axes[0][1].set_title("B")
            axes[0][2].set_title("X")
            fig.subplots_adjust()
            plt.show()

    else:
        dsprites_loader = DSpritesLoader(npz_path='./DSPRITES/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = DataLoader(DSpritesIIDPairs(dsprites_loader=dsprites_loader),
                                    batch_size=64)
        x1, x2, _ = next(iter(data))
        print(x1.shape)
        print(x2.shape)
        x1 = x1[0][0]
        x2 = x2[0][0]
        fig, axes = plt.subplots(2)
        axes[0].imshow(x1.view(64,64), cmap='Greys_r')
        axes[1].imshow(x2.view(64,64), cmap='Greys_r')
        axes[0].set_title("z")
        axes[1].set_title("z_")
        fig.subplots_adjust()
        plt.show()