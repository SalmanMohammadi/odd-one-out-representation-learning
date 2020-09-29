import numpy as np
import torch
from sklearn import linear_model
import sys
sys.path.append('../')
from data import rpm_data as rpm
from data.rpm_data import ColourDSprites, ColourDSpritesLoader
from models.models_disentanglement import AdaGVAE, TVAE, VAE, batch_sample_latents

CUDA = torch.device('cuda')

def calculate_b_vae_score(model, train_size=10000, test_size=5000, batch_size=64):
    dsprites_loader = ColourDSpritesLoader(npz_path='./data/DSPRITES/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    train_data = ColourDSprites(size=train_size*64, dsprites_loader=dsprites_loader, batch_size=batch_size)
    test_data = ColourDSprites(size=test_size*64, dsprites_loader=dsprites_loader, batch_size=batch_size) 
    x_train, y_train = generate_training_batch(train_data, model, train_size)
    x_test, y_test = generate_training_batch(test_data, model, test_size)
    model = linear_model.LogisticRegression()
    model.fit(x_train, y_train)
    
    train_accuracy = model.score(x_train, y_train)
    eval_accuracy = model.score(x_test, y_test)
    scores_dict = {}
    scores_dict["train_accuracy"] = train_accuracy
    scores_dict["eval_accuracy"] = eval_accuracy
    return scores_dict

def generate_training_batch(data, model, num_points):
    labels = np.zeros(num_points, dtype=np.int64)
    points = np.zeros((num_points, 10))
    for i in range(num_points):
        labels[i], points[i] = generate_training_sample(data, model)
    return points, labels

def generate_training_sample(data, model):

    idx = np.random.randint(len(data.factors_sizes))
    z_1 = np.hstack(data.sample_latent())
    z_2 = np.hstack(data.sample_latent())

    z_2[:, idx] = z_1[:, idx]

    x_1 = torch.tensor(data.latent_to_observations(z_1), dtype=torch.float32).to(CUDA).squeeze()
    x_2 = torch.tensor(data.latent_to_observations(z_2), dtype=torch.float32).to(CUDA).squeeze()
 
    mu_1 = model.batch_representation(x_1).cpu().detach().numpy()
    mu_2 = model.batch_representation(x_2).cpu().detach().numpy()

    return idx, np.mean(np.abs(mu_1 - mu_2), axis=0)

if __name__ == "__main__":
    vae = TVAE(n_channels=3)
    vae.load_state_dict(torch.load("../tmp/vae/1.pt"))
    for label, metric in calculate_b_vae_score(vae).items():
        print(label, ":", metric)