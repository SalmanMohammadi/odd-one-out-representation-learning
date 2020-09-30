import numpy as np
import scipy
from six.moves import range
from sklearn.ensemble import GradientBoostingClassifier
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter 
import sys
sys.path.append('../')
import models
from models.models_disentanglement import AdaGVAE, TVAE, batch_sample_latents
from data import dsprites_data as dsprites
from data import rpm_data as rpm
from data.dsprites_data import IterableDSPritesIID
from data.rpm_data import ColourDSprites

# Calculates the disentanglement, completeness, and informativeness scores
# Eastwood 2018 https://homepages.inf.ed.ac.uk/ckiw/postscript/iclr_final.pdf
# implementation based on disentanglement_lib 
# https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/dci.py
def compute_dci(model, dataset, train_size=10000, test_size=5000, batch_size=16):
    if dataset in ['colour_triplets', 'colour', 'colour_pairs']:
        train_data, test_data = rpm.get_dsprites(train_size=train_size, test_size=test_size, 
                                            dataset=ColourDSprites, batch_size=batch_size)
    else:
        train_data, test_data = dsprites.get_dsprites(train_size=train_size, test_size=test_size, 
                                            dataset=IterableDSPritesIID, batch_size=1)
    train_loc, train_y = batch_sample_latents(model, train_data, train_size, batch_size=batch_size)
    assert train_loc.shape[0] == train_size
    assert train_y.shape[0] == train_size
    test_loc, test_y = batch_sample_latents(model, test_data, test_size, batch_size=batch_size)
    assert test_loc.shape[0] == test_size
    assert test_y.shape[0] == test_size
    importance_matrix, train_err, test_err = compute_importance(train_loc, train_y, test_loc, test_y)
    
    assert importance_matrix.shape[0] == train_loc.shape[1]
    assert importance_matrix.shape[1] == train_y.shape[1]

    scores = {}
    scores["disentanglement"] = disentanglement(importance_matrix)
    scores["completeness"] = completeness(importance_matrix)
    # print(scores)
    return scores
    
def disentanglement(importance_matrix):
    latent_entropy = 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])
    if importance_matrix.sum() == 0:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return np.sum(latent_entropy*code_importance)

def completeness(importance_matrix):
    factor_entropy = 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])
    if importance_matrix.sum() == 0:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()

    return np.sum(factor_entropy*factor_importance)

# x_train, y_train, x_test, y_test shape (num_samples, num_factors)
def compute_importance(x_train, y_train, x_test, y_test):
    num_factors = y_train.shape[1]
    num_codes = x_train.shape[1]
    importance_matrix = np.zeros(shape=[num_codes, num_factors], dtype=np.float64)
    train_loss = []
    test_loss = []
    for i in range(num_factors):
        model = GradientBoostingClassifier()
        model.fit(x_train, y_train[:, i])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(x_train) == y_train[:, i]))
        test_loss.append(np.mean(model.predict(x_test) == y_test[:, i]))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)

# def compute_dci(loc_train, y_train, loc_test, y_test):
if __name__ == '__main__':
    
    def eval_model(path, id):
        print(path)
        vae = TVAE(n_channels=3)
        writer = SummaryWriter(log_dir=path)
        vae.load_state_dict(torch.load(path+".pt"))
        for label, metric in compute_dci(vae, dataset='colour').items():
            print(label, ":", metric)
            writer.add_scalar(label, metric, id)

        writer.flush()
        writer.close()

    paths =[
        "../tmp/tvae/gamma=0+k=1/",
        "../tmp/tvae/gamma=0+k=rnd/",
        "../tmp/tvae/gamma=1+k=1/",
        "../tmp/tvae/gamma=1+k=rnd/",
        "../tmp/vae/",
    ]
    for path in paths:
        for i in [4]:
            eval_model(path+str(i), i)

    # vae = TVAE(n_channels=3)
    # vae.load_state_dict(torch.load("../tmp/tvae/gamma=1+k=rnd/1.pt"))
    # print(compute_dci(vae, dataset='colour'))
    # vae = TVAE(n_channels=3)
    # vae.load_state_dict(torch.load("../tmp/tvae/gamma=1+k=rnd/2.pt"))
    # print(compute_dci(vae, dataset='colour'))
    
    # vae = TVAE(n_channels=3)
    # vae.load_state_dict(torch.load("../tmp/tvae/gamma=1+k=rnd/3.pt"))
    # print(compute_dci(vae, dataset='colour'))

    # vae = TVAE(n_channels=3)
    # vae.load_state_dict(torch.load("../tmp/tvae/gamma=1+k=rnd/4.pt"))
    # print(compute_dci(vae, dataset='colour'))
    
