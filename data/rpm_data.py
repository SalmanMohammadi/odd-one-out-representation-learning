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


class QuantizedColourDSprites(QuantizedColourDSprites):
    
    def 

class PGM(IterableDataset):
    def __init__(self, dataset, num_batches, relations_p=[1/3]*3, 
                rows=3, cols=3, factors=[5, 6, 3, 6, 32, 32]):
        self.dataset = dataset
        self.size = num_batches
        self.relations_p = relations_p
        self.rows = 3
        self.cols = 3
        self.factors = np.array(factors)

    def sample_no_relation(self, factor, size):
        # sample order of factors in matrix
        idx_duplicate = np.zeros((size*2, 3), dtype=int)
        rand_ = np.random.randint(3, size=size*2)
        idx_duplicate[range(size*2), rand_] = 1

        # vectorized random sampling without replacement trick
        idx = np.random.rand(size*2, factor).argpartition(2,axis=1)[:,:2]
        out = np.take(range(factor), idx)
        out = out[np.array([range(size*2)]*3).T, idx_duplicate].reshape(size,2,3)
        
        # sample final rows
        final = np.expand_dims(np.random.choice(factor, size=(size, 3)), 1)
        matrix = np.hstack((out, final))
        return matrix

    def sample_constant_relation(self, factor, size):
        matrix = np.zeros((size, 3, 3))
        for i in range(3):
            matrix[:, i, :] = np.array([np.random.choice(factor, size=size)]*3).T

        return matrix
        
    def sample(self):
        # sample a (batch_size, 3, 3, num_relations) array of AND relationship indicies
        # and a (batch_size, 5, num_relations) array of hard alternative answers
        # **** TODO URGENT ***
        # change num_relations to be per-sample in batch rather than constant over a batch
        num_relations = 1 + np.random.choice(3, p=self.relations_p)
        relations_idx = np.hstack([np.random.rand(self.size, len(factors)).argpartition(1,axis=1)[:,:1] 
                                for _ in range(num_relations)])
        matrix = np.zeros((batch_size, 3, 3, len(self.factors)), dtype=int)
        for j, factor in enumerate(self.factors):
            idx_ = np.any([relations_idx[:,i] == j for i in range(num_relations)], axis=0)
            matrix[idx_, :, :, j] = self.sample_constant_relation(factor, idx_.sum())
            matrix[(~idx_), :, :, j] = self.sample_no_relation(factor, (~idx_).sum())

        # todo sample x

        other_solutions = np.zeros((batch_size, 5, len(factors)), dtype=int)
        for i in range(5):
            other_solutions[:, i] = self.modify_solutions(matrix)

        return matrix, other_solutions

    def modify_solutions(self, matrix):
        alt_mat = np.copy(matrix)
        init_solutions = np.copy(matrix[:, -1, -1, :])
        m_relations_idx = relations_idx[range(self.size), np.random.randint(num_relations, size=batch_size)]
        idx = np.array(range(self.size))
        new_solutions = np.zeros((self.size, 6))
        # sample new fixed random factor per item in batch
        while np.any(idx):
            factors_ = np.hstack([np.random.rand(len(idx), factor).argpartition(1,axis=1)[:,:1] 
                                    for factor in factors])
            new_solutions[idx] = factors_[m_relations_idx[idx]]
            idx = new_solutions == init_solutions[idx, m_relations_idx[idx]]
        
        init_solutions[m_relations_idx] = new_solutions
        # resample non active relations
        for j, factor in enumerate(factors):
            # get indices for constant relations for the current batch
            idx = np.any([relations_idx[:,i] == j for i in range(num_relations)], axis=0)
            # ensure ~idx_ are of the form aaa or abc w.r.t alt_mat[~idx_, -1, :-1, j]
            # idx for all non constant relations of the form aa become aaa
            same_idx = np.logical_and(idx, alt_mat[:, -1, 0, j] == alt_mat[:, -1, 1, j])
            init_solutions[same_idx,  j] = alt_mat[same_idx, -1, 1, j]
            # idx for all non constant relations of the form ab/ba become abc/bac               
            # ensure it's randomly sampled and doesnt include alt_mat[~idx, -1, :-1, j]
            factor_set = np.array([range(factor)]*np.sum(~same_idx))
            factor_samples = np.ones((np.sum(~same_idx), factor), dtype=bool)
            factor_samples[range(np.sum(~same_idx)), alt_mat[~same_idx, -1, :-1, j].T] = 0
            factor_set = factor_set[factor_samples]#.reshape(len(idx), factor-2)
            init_solutions[~same_idx, j] = factor_set[np.random.choice(factor-2, size=np.sum(~same_idx))]
        return init_solutions


    def __iter__(self):
        pass

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