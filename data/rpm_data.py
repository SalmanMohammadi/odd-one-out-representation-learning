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
    def __init__(self, dsprites_loader, size=300000, batch_size=64, k=None):
        # DSprites ground truth model for disentanglement learning
        self.batch_size = batch_size
        self.num_batches = size
        self.dsprites_loader = dsprites_loader
        self.metadata = self.dsprites_loader.metadata
        self.latents_sizes = self.metadata['latents_sizes']
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                                np.array([1,])))

    # generative random sampler
    def sample(self):
        c, z = self.sample_latent()
        z_idx = self.latent_to_index(z)
        X = self.dsprites_loader.X[z_idx]

        background_color = np.expand_dims(np.expand_dims(BACKGROUND_COLORS[c[:,0]], 2), 2)
        object_color = np.expand_dims(np.expand_dims(OBJECT_COLORS[c[:,1]], 2), 2)
        
        X = X * object_color + (1. - X) * background_color
        return torch.tensor(X, dtype=torch.float32), np.hstack((c, z[:,1:]))
        
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

class ColourDSpritesTriplets(IterableDataset):
    def __init__(self, dsprites_loader, size=300000, batch_size=64, k=1):
        # 0 - background color (5 different values)
        # 1 - object color (6 different values)
        # 2 - shape (3 different values)
        # 3 - scale (6 different values)
        # 4 - orientation (40 different values)
        # 5 - position x (32 different values)
        # 6 - position y (32 different values)
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
    def __iter__(self):
        for i in range(self.num_batches):
            yield self.sample()

    def __len__(self):
        return self.num_batches

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)
    
    @property
    def factors_sizes(self):
        return np.array([BACKGROUND_COLORS.shape[0], OBJECT_COLORS.shape[0]] + list(self.latents_sizes[1:]))
    
    def sample(self):
        z_1, z_2, z_3, k_idxs = self.sample_latent_triplets()
        X1, X2, X3 = map(self.latent_to_observations, (z_1, z_2, z_3))
        
        # randomly sample positions
        positions = np.random.rand(self.batch_size, 3).argpartition(2,axis=1)
        new_x = np.zeros((self.batch_size, 3, 64, 64, 3))
        new_x[range(self.batch_size), :, :, :, positions[:, 0]] = X1
        new_x[range(self.batch_size), :, :, :, positions[:, 1]] = X2
        new_x[range(self.batch_size), :, :, :, positions[:, 2]] = X3

        X1, X2, X3 = map(np.squeeze, np.split(new_x, 3, axis=-1))
        X1 = torch.tensor(X1, dtype=torch.float32)
        X2 = torch.tensor(X2, dtype=torch.float32)
        X3 = torch.tensor(X3, dtype=torch.float32)
        positions = torch.tensor(positions, dtype=torch.long)
        return X1, X2, X3, positions

    def sample_latent_triplets(self):
        z_1 = np.zeros((self.batch_size, self.factors_sizes.size))
        for lat_i, lat_size in enumerate(self.factors_sizes):
            z_1[:, lat_i] = np.random.randint(lat_size, size=self.batch_size)
        # sample k factors of variation which should not be shared between x1,x2
        # but will be the only factor shared between x1,x3 and x2,x3
        # k ~ unif(1, d-1)
        if not self.k:
            k_samples = np.random.randint(1, (self.factors_sizes.size-1), size=self.batch_size)
        elif self.k == 1:
            # k_samples = np.ones((self.batch_size), dtype=int)
            k_idxs = np.random.randint(1, self.factors_sizes.size, size=self.batch_size)
            lat_range = list(range(1, self.factors_sizes.size))
            k_idxs_1 = [lat_range[:k_]+lat_range[k_+1:] for k_ in k_idxs]
        else:
            k_samples = np.ones((self.batch_size), dtype=int) * self.k

        z_2 = np.copy(z_1)
        z_3 = np.copy(z_1)
        for i, (idx, idx_1) in enumerate(zip(k_idxs[:, None], k_idxs_1)):
            for j, lat_size in zip(idx, self.factors_sizes[idx]):
                z_2[i, j] = np.random.randint(lat_size)
            for j, lat_size in zip(idx_1, self.factors_sizes[idx_1]):
                z_3[i, j] = np.random.randint(lat_size)
        return z_1, z_2, z_3, k_idxs

    def latent_to_observations(self, latents):
        c, z = latents[:, :2], latents[:, 2:]
        X = self.dsprites_loader.X[self.latent_to_index(np.insert(z, 0, 0, axis=1))]
        return self.colourize(c, X)

    def colourize(self, c, X):
        c = c.astype(int)
        background_color = np.expand_dims(np.expand_dims(BACKGROUND_COLORS[c[:,0]], 2), 2)
        object_color = np.expand_dims(np.expand_dims(OBJECT_COLORS[c[:,1]], 2), 2)
        
        return X * object_color + (1. - X) * background_color

class QuantizedColourDSprites():
    def __init__(self, dsprites_loader, factors=[1, 2, 4, 5], factors_sizes=[5, 6, 3, 3, 4, 4]):
        # 0 - background color (5 different values)
        # 1 - object color (6 different values)
        # 2 - shape (3 different values)
        # 3 - scale (3 different values)
        # 4 - position x (4 different values)
        # 5 - position y (4 different values)
        # DSprites ground truth model for abstract reasoning
        # quantization scheme following https://arxiv.org/pdf/1905.12506.pdf
        self.dsprites_loader = dsprites_loader
        self.metadata = self.dsprites_loader.metadata
        if not factors:
            factors = list(range(6))
        self.factors_idx = factors
        self.latents_sizes = self.metadata['latents_sizes']
        self.factors_true = np.array([5, 6, 3, 6, 32, 32])
        self.factors_sizes = np.array(factors_sizes)
        self.obs_idx = [i for i in range(len(self.latents_sizes)) if i not in self.factors_idx]
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                                np.array([1,])))

    def latent_to_observations(self, latents):
        for i in range(self.factors_sizes.size):
            if self.factors_sizes[i] != self.factors_true[i]:
                ratio = self.factors_true[i] / self.factors_sizes[i]
                latents[:, i] = np.floor(latents[:, i] * ratio)

        c, z = latents[:, :2], latents[:, 2:]
        new_z = np.zeros((latents.shape[0], self.latents_sizes.size))
        new_z[:, self.factors_idx] = z
        # sample factors not indexed by latents
        for i, lat_size in zip(self.obs_idx, self.latents_sizes[self.obs_idx]):
            new_z[:, i] = np.random.randint(lat_size, size=latents.shape[0])

        X = self.dsprites_loader.X[self.latent_to_index(new_z)]
        return self.colourize(c, X)

    def colourize(self, c, X):
        c = c.astype(int)
        background_color = np.expand_dims(np.expand_dims(BACKGROUND_COLORS[c[:,0]], 2), 2)
        object_color = np.expand_dims(np.expand_dims(OBJECT_COLORS[c[:,1]], 2), 2)
        
        return X * object_color + (1. - X) * background_color

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

class PGM(IterableDataset):
    def __init__(self, dataset, num_batches, batch_size=32, n_relations=None, 
                rows=3, cols=3, factors=[5, 6, 3, 3, 4, 4]):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.n_relations = n_relations
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
        matrix = np.zeros((size, 3, 3), dtype=int)
        for i in range(3):
            matrix[:, i, :] = np.array([np.random.choice(factor, size=size)]*3).T

        return matrix
        
    def sample(self):
        # sample a (batch_size, 3, 3, num_relations) array of AND relationship indicies
        # and a (batch_size, 5, num_relations) array of hard alternative answers
        if self.n_relations:
            num_relations = np.ones((self.batch_size)) * self.n_relations
        else:
            num_relations = 1 + np.random.choice(3, size=self.batch_size)

        relations_idx = np.zeros((self.batch_size, len(self.factors)), dtype=int)
        for i in range(1, 4):
            idx = num_relations == i
            if sum(idx) > 0:
                relations_factors = np.random.rand(sum(idx), len(self.factors)).argpartition(1,axis=1)[:,:i]
                relations_idx[idx, relations_factors.T] = 1

        matrix = np.zeros((self.batch_size, 3, 3, len(self.factors)), dtype=int)
        for j, factor in enumerate(self.factors):
            idx_ = relations_idx[:, j].astype(bool)
            matrix[idx_, :, :, j] = self.sample_constant_relation(factor, idx_.sum())
            matrix[(~idx_), :, :, j] = self.sample_no_relation(factor, (~idx_).sum())

        # TODO URGENT
        # ensure modify_solutions doesn't return alternative solutions already in solutions
        solutions = np.zeros((self.batch_size, 5, len(self.factors)), dtype=int)
        for i in range(5):
            solutions[:, i] = self.modify_solutions(matrix, relations_idx, num_relations)

        # randomly sample positions where the correct answer should be inserted
        positions = np.random.choice(6, size=self.batch_size)

        idx = np.ones((self.batch_size, 6), dtype=bool)
        idx[range(self.batch_size), positions] = 0
        alternative_solutions = np.zeros((self.batch_size, 6, len(self.factors)))
        alternative_solutions[~idx] = matrix[:, -1, -1]
        alternative_solutions[idx] = solutions.reshape(-1, 6)

        # sample ground truth factors
        matrix_observations = self.dataset.latent_to_observations(matrix.reshape(-1, 6)).reshape(-1, 3, 3, 3, 64, 64)
        alternative_observations = self.dataset.latent_to_observations(alternative_solutions.reshape(-1, 6)).reshape(-1, 6, 3, 64, 64)
        alternative_observations[~idx] = matrix_observations[:, -1, -1]        
        matrix_observations = torch.tensor(matrix_observations, dtype=torch.float32)
        alternative_observations = torch.tensor(alternative_observations, dtype=torch.float32)
        positions = torch.tensor(positions, dtype=torch.int64)

        matrix_values = self.embed_factors(matrix.reshape(-1, 6)).reshape(-1, 3, 3, 6)
        alternative_values = self.embed_factors(alternative_solutions.reshape(-1, 6)).reshape(-1, 6, 6)
        return matrix_observations, alternative_observations, positions, matrix_values, alternative_values

    def modify_solutions(self, matrix, relations_idx, num_relations):
        alt_mat = np.copy(matrix)
        init_solutions = np.copy(matrix[:, -1, -1, :])

        # find samples which have more than 1 active relation to resample
        idx = np.sum(relations_idx, 1) > 1
        # randomly sample a single relation to change
        active_relations = np.split(np.nonzero(relations_idx[idx])[1], np.cumsum(num_relations[idx]))[:-1]
        m_relations_idx = np.zeros((self.batch_size), dtype=int)
        m_relations_idx[idx] = np.array([np.random.choice(x) for x in active_relations])

        new_solutions = np.ones((self.batch_size), dtype=int) * -1
        # sample new fixed random factor per item in batch until
        # there are no samples which have the same factor as the original
        while np.any(idx):
            factors_ = np.hstack([np.random.choice(factor, size=sum(idx))[:, None] for factor in self.factors])
            new_solutions[idx] = factors_[range(sum(idx)), m_relations_idx[idx]]
            idx[idx] = new_solutions[idx] == init_solutions[idx, m_relations_idx[idx]] 
        
        idx = np.sum(relations_idx, 1) > 1
        init_solutions[idx, m_relations_idx[idx]] = new_solutions[idx]
        # resample non active relations
        for j, factor in enumerate(self.factors):
            # get indices for constant relations for the current batch
            idx = relations_idx[:, j].astype(bool)
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
    
    def __len__(self):
        return self.num_batches
    
    def embed_factors(self, factors):
        """
        Embeds factors linearly in [-0.5, 0.5]
        """
        result = np.array(factors, dtype=np.float32)
        max_vals = np.array(self.factors, dtype=np.float32) - 1.
        result /= np.expand_dims(max_vals, 0)
        return result - .5

    def __iter__(self):
        for i in range(self.num_batches):
            yield self.sample()

def get_datasets(train_size=100000, test_size=10000, batch_size=32):
    dsprites_loader = ColourDSpritesLoader(npz_path='./data/DSPRITES/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    dsprites_reasoning = QuantizedColourDSprites(dsprites_loader=dsprites_loader)
    train_abstract_reasoning = DataLoader(PGM(dsprites_reasoning, num_batches=train_size, batch_size=batch_size), 
                                batch_size=1)
    val_abstract_reasoning = DataLoader(PGM(dsprites_reasoning, num_batches=train_size//10, batch_size=batch_size),
                            batch_size=1)
    test_abstract_reasoning = DataLoader(PGM(dsprites_reasoning, num_batches=test_size, batch_size=batch_size), 
                                batch_size=1)

    return train_abstract_reasoning, val_abstract_reasoning, test_abstract_reasoning

def get_dsprites(train_size=300000, test_size=10000, batch_size=64, k=1, dataset=ColourDSpritesTriplets):
    """
    Returns train and test DSprites dataset.
    """
    dsprites_loader = ColourDSpritesLoader(npz_path='../data/DSPRITES/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    train_data = DataLoader(dataset(size=train_size, dsprites_loader=dsprites_loader, k=k, batch_size=batch_size),
                            batch_size=1)#, pin_memory=True, num_workers=16)
    test_data = DataLoader(dataset(size=test_size, dsprites_loader=dsprites_loader, k=k, batch_size=batch_size),
                            batch_size=1)#, pin_memory=True, num_workers=16)                    
    return train_data, test_data

def show_task(matrix, alternative_solutions, y):
    fig, axes = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            axes[i][j].imshow(matrix[i][j].T)
    fig, axes = plt.subplots(2, 3)
    alternative_solutions = alternative_solutions.reshape(2, 3, 3, 64, 64)
    for i in range(2):
        for j in range(3):
            axes[i][j].imshow(alternative_solutions[i][j].T)
    plt.show()

# class IterableDSpritesIIDPairs(IterableDataset):
#     def __init__(self, dsprites_loader, size=300000, batch_size=64, k=None):
if __name__ == '__main__':
    dsprites_loader = ColourDSpritesLoader(npz_path='./DSPRITES/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    data = DataLoader(ColourDSprites(dsprites_loader=dsprites_loader),
                                        batch_size=1)
    x = next(iter(data))[0][0]
    print(x.shape)
    fix, axes = plt.subplots(15, sharex=True, sharey=True)
    for i in range(15):
        axes[i].imshow(x[i].T)
    plt.show()


    # dsprites_loader = ColourDSpritesLoader(npz_path='./DSPRITES/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    # data = QuantizedColourDSprites(dsprites_loader=dsprites_loader)
    # pgm_data = DataLoader(PGM(data, num_batches=300000, batch_size=32), batch_size=1)
    # x, x_, y, _, _ = next(iter(pgm_data))
    # x, x_, y = x.squeeze(), x_.squeeze(), y.squeeze()
    # print(x.shape)
    # show_task(x[0], x_[0], y[0])

    # dsprites_loader = ColourDSpritesLoader(npz_path='./DSPRITES/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    # data = DataLoader(ColourDSpritesTriplets(dsprites_loader=dsprites_loader, batch_size=5, k=1), batch_size=1)
    # x1, x2, x3, pos = next(iter(data))
    # print("pos", pos)
    # print(x1.shape)
    # x1 = x1.reshape(5, 3, 64, 64)
    # x2 = x2.reshape(5, 3, 64, 64)
    # x3 = x3.reshape(5, 3, 64, 64)
    # fig, axes = plt.subplots(5, 3, sharex=True, sharey=True)
    # for i in range(5):
    #     axes[i][0].imshow(x1[i].T, cmap='Greys_r')
    #     axes[i][1].imshow(x2[i].T, cmap='Greys_r')
    #     axes[i][2].imshow(x3[i].T, cmap='Greys_r')
    # axes[0][0].set_title("A")
    # axes[0][1].set_title("B")
    # axes[0][2].set_title("X")
    # fig.subplots_adjust()
    # plt.show()