import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

OBJECT_COLORS = np.array(
    [[0.9096231780824386, 0.5883403686424795, 0.3657680693481871],
     [0.6350181801577739, 0.6927729880940552, 0.3626904230371999],
     [0.3764832455369271, 0.7283900430001952, 0.5963114605342514],
     [0.39548987063404156, 0.7073922557810771, 0.7874577552076919],
     [0.6963644829189117, 0.6220697032672371, 0.899716387820763],
     [0.90815966835861, 0.5511103319168646, 0.7494337214212151]])

BACKGROUND_COLORS = np.array([
    (0., 0., 0.),
    (.25, .25, .25),
    (.5, .5, .5),
    (.75, .75, .75),
    (1., 1., 1.),
])

class ColourDSpritesLoader():
    def __init__(self, npz_path="./data/DSPRITES/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"):
        with np.load(npz_path, allow_pickle=True, encoding='latin1') as dataset_zip:
            self.metadata = dataset_zip['metadata'][()]
            self.X = np.reshape(dataset_zip['imgs'], (-1, 1, 64, 64))
            self.Y = dataset_zip['latents_values']

class ColourDSprites(IterableDataset):
    def __init__(self, dsprites_loader, factors=None, size=300000, batch_size=64):
        self.batch_size = batch_size
        self.num_batches = size
        self.dsprites_loader = sprites_loader
        self.metadata = self.dsprites_loader.metadata
        if not factors:
            factors = list(range(6))
        self.factors_idx = factors
        self.obs_idx = [i for i in range(len(latents_sizes)) if i not in self.factors_idx]
        self.latents_sizes = self.metadata['latents_sizes'][1:]
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                                np.array([1,])))

    @property
    def factors_sizes(self):
        return np.array([BACKGROUND_COLORS.shape[0], OBJECT_COLORS.shape[0]] + self.latents_sizes[self.factors_idx])

    def sample_remaining_factors(self, latents):
        n = latents.shape[0]
        all_latents = np.zeros((n, self.latents_sizes.size))

        all_latents[:, self.factors_idx] = latents
        for i, lat_size in enumerate(self.latents_sizes[self.obs_idx]):
            all_latents[:, i] = np.random.randint(lat_size, size=n)
        return all_latents

    # generative random sampler
    def sample(self):
        c, z = self.sample_latent()
        z_idx = self.latent_to_index(z)
        X = torch.tensor(self.dsprites_loader.X[z_idx], dtype=torch.float32)

        background_color = np.expand_dims(np.expand_dims(BACKGROUND_COLORS[c[:,0]], 2), 2)
        object_color = np.expand_dims(np.expand_dims(OBJECT_COLORS[c[:,1]], 2), 2)
        
        X = X * object_color + (1. - X) * background_color
        return X
        
    def __iter__(self):
        for i in range(self.num_batches):
            yield self.sample()

    def __len__(self):
        return self.num_batches

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)
    
    def sample_latent(self):
        colours = np.zeros((self.batch_size, 2), dtype=np.int32)
        colours[:, 0] = np.random.randint(BACKGROUND_COLORS.shape[0], size=self.batch_size)
        colours[:, 1] = np.random.randint(OBJECT_COLORS.shape[0], size=self.batch_size)
        samples = np.zeros((self.batch_size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=self.batch_size)

        return colours, samples

class PGM(IterableDataset):
    def __init__(self, dataset, relations_p, num_batches, relation_types=[ANDRelation, RNDRelation],
                rows=3, cols=3):
        self.dataset = dataset
        self.num_relations = 1 + np.random.choice(len(relations_p), p=relations_p)
        self.relation_types = dict(enumerate(relation_types))
        self.rows=3
        self.cols=3

        
    def sample(self):
        # sample a (batch_size, num_relations) array of AND relationship indicies
        relations_idx = np.array([np.random.choice(self.latents_sizes.size, size=num_relations, replace=False) 
                                for _ in range(self.batch_size)])
        relations = np.array([[RNDRelation]*dataset.factors_sizes.size]*self.batch_size)
        relations[range(self.batch_size), relations_idx.T] = ANDRelation
        self.matrices = np.zeros(batch_size, )
        

    def __iter__(self):


class Relation(Object):
    @staticmethod
    def consistent(matrix):
        raise NotImplementedError()
    
    @staticmethod
    def sample(factor_size, size):
        # factor_size - number of possible value for ground truth factor
        # samples a (batch_size, rows, cols, 1) tensor from factors
        raise NotImplementedError()
        

class ANDRelation(Relation):
    @staticmethod
    def consistent(matrix):
    
    @staticmethod
    def sample(factor_size, size):

class RNDRelation(Relation):
    
    @staticmethod
    def consistent(matrix):
    
    @staticmethod
    def sample(factor_size, size):

# class IterableDSpritesIIDPairs(IterableDataset):
#     def __init__(self, dsprites_loader, size=300000, batch_size=64, k=None):
if __name__ == '__main__':
    dsprites_loader = ColourDSpritesLoader(npz_path='./DSPRITES/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    data = DataLoader(ColourDSprites(dsprites_loader=dsprites_loader),
                                        batch_size=1)
    x = next(iter(data))[0]
    print(x.shape)
    fix, axes = plt.subplots(15, sharex=True, sharey=True)
    for i in range(15):
        axes[i].imshow(x[i].T)
    plt.show()