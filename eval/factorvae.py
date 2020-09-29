import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn import linear_model
import sys
sys.path.append('../')
from data import rpm_data as rpm
from data.rpm_data import ColourDSprites, ColourDSpritesLoader
from models.models_disentanglement import AdaGVAE, TVAE, batch_sample_latents

CUDA = torch.device("cuda")
def compute_factorvae_score(model, train_size=10000, test_size=5000, batch_size=64):
    dsprites_loader = ColourDSpritesLoader(npz_path='./data/DSPRITES/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    train_data = ColourDSprites(size=train_size, dsprites_loader=dsprites_loader, batch_size=batch_size)
    test_data = ColourDSprites(size=test_size, dsprites_loader=dsprites_loader, batch_size=batch_size) 
    
    variances = compute_variances(model, train_data, train_size)
    active_dims = prune_dims(variances)

    scores = {}
    if not active_dims.any():
        scores["train_accuracy"] = 0.
        scores["eval_accuracy"] = 0.
        scores["num_active_dims"] = 0
        return scores

    train_votes = generate_training_batch(model, train_data, train_size, variances, active_dims)
    classifier = np.argmax(train_votes, axis=0)
    other_idx = np.arange(train_votes.shape[1])

    train_acc = np.sum(train_votes[classifier, other_idx]) * 1. / np.sum(train_votes)

    test_votes = generate_training_batch(model, test_data, test_size, variances, active_dims)
    test_acc = np.sum(test_votes[classifier, other_idx]) * 1. / np.sum(test_votes)

    scores["train_accuracy"] = train_acc
    scores["test_accuracy"] = test_acc
    scores["num_active_dims"] = len(active_dims)
    return scores

def generate_training_sample(model, data, variances, active_dims):
    idx = np.random.randint(len(data.factors_sizes))
    z = np.hstack(data.sample_latent())
    z[:, idx] = z[0, idx]

    x = torch.tensor(data.latent_to_observations(z), dtype=torch.float32).to(CUDA).squeeze()
    mu = model.batch_representation(x).cpu().detach().numpy()

    local_var = np.var(mu, axis=0, ddof=1)
    argmin = np.argmin(local_var[active_dims]/variances[active_dims])
    return idx, argmin

def generate_training_batch(model, data, num_points, variances, active_dims):
    votes = np.zeros((len(data.factors_sizes), variances.shape[0]), dtype=np.int64)
    for _ in range(num_points):
        i, argmin = generate_training_sample(model, data, variances, active_dims)
        votes[i, argmin] += 1
    return votes

def prune_dims(variances, threshold=0.05):
    scale_z = np.sqrt(variances)
    return scale_z >= threshold

def compute_variances(model, data, num_points):
    data_ = DataLoader(data, batch_size=1)
    mu, z = batch_sample_latents(model, data_, num_points, batch_size=64)
    mu = mu.cpu().detach().numpy()
    assert mu.shape[0] == num_points
    return np.var(mu, axis=0, ddof=1)

if __name__ == "__main__":
    vae = TVAE(n_channels=3)
    vae.load_state_dict(torch.load("../tmp/tvae/gamma=0+k=1/1.pt"))
    for label, metric in compute_factorvae_score(vae).items():
        print(label, ":", metric)