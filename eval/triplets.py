import numpy as np
import scipy
from six.moves import range
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter 
import sys
sys.path.append('../')
import models
from models.models_disentanglement import TVAE, VAE, batch_sample_latent_triplets
from data import rpm_data as rpm
from data.rpm_data import ColourDSpritesTriplets

def calculate_triplet_score(model, train_size=15000, test_size=1, batch_size=16):
    train_data, test_data = rpm.get_dsprites(train_size=train_size, test_size=test_size, 
                                            dataset=ColourDSpritesTriplets, batch_size=batch_size)
    train_loc, train_y = batch_sample_latent_triplets(model, train_data, train_size, batch_size=batch_size)
    assert train_loc.shape[0] == train_size
    assert train_y.shape[0] == train_size
    train_acc, test_acc = predict_triplets(train_loc, train_y)
    scores = {}
    scores['mean_train'] = np.mean(train_acc)
    scores['std_train'] = np.std(train_acc)
    scores['mean_test'] = np.mean(test_acc)
    scores['std_test'] = np.std(test_acc)
    return scores

def predict_triplets(X, y, folds=5):
    y = y.numpy()
    train_loss = []
    test_loss = []
    # for i in range(folds):
    kf = KFold(n_splits=folds)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train[:, -1])
        pred = np.array(model.predict(X_train))
        pred_test = np.array(model.predict(X_test))
        train_loss.append(np.sum(pred == y_train[:, -1]) / X_train.shape[0])
        test_loss.append(np.sum(pred_test == y_test[:, -1]) / X_test.shape[0])

    return train_loss, test_loss


if __name__ == '__main__':
    vae = TVAE(n_channels=3)
    vae.load_state_dict(torch.load("../tmp/tvae/p/0.pt"))
    # vae = TVAE(n_channels=3)
    # vae.load_state_dict(torch.load("../tmp/tvae/abstract_reasoning/0.pt"))
    #(0.8804833333333333, 0.00548568845066668, 0.8496666666666666, 0.010088497300281043)
    # vae = VAE(n_channels=3)
    # vae.load_state_dict(torch.load("../tmp/vae/abstract_reasoning/0.pt"))
    # (0.8322499999999999, 0.005524440645390664, 0.7926, 0.009534498763263151)
    print(calculate_triplet_score(vae))
    
