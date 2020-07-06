import numpy as np
import scipy
from six.moves import range
from sklearn.ensemble import GradientBoostingClassifier
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter 
import models
from models import AdaGVAE
from data import dsprites_data as dsprites

def compute_dci(data, model, train_size=10000, test_size=5000, batch_size=16):
    scores = {}

def disentanglement():
    pass

def completeness():
    pass

def informativeness():
    pass

def importance_matrix(x_train, y_train, x_test, y_test):
    num_factors = y_train.shape[1]
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros(shape=[num_codes, num_factors],
                                dtype=np.float64)
    train_loss = []
    test_loss = []
    for i in range(num_factors):
        model = GradientBoostingClassifier()
        model.fit(x_train.T, y_train[i, :])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
        test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)

# def compute_dci(loc_train, y_train, loc_test, y_test):
if __name__='__main__':
    _, test_data = dsprites.get_dsprites(train_size=737280, test_size=737280, batch_size=64)
    vae = AdaGVAE(n_channels=1)
    
