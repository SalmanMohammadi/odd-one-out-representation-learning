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
from models.models_disentanglement import AdaGVAE, batch_sample_latents
from data import rpm_data as rpm
from data.rpm_data import ColourDSpritesTriplets

def calculate_triplet_score(model, dataset, train_size=10000, test_size=5000, batch_size=16):
    train_data, test_data = rpm.get_dsprites(train_size=train_size, test_size=test_size, 
                                            dataset=ColourDSpritesTriplets, batch_size=batch_size)
    train_loc, train_y = batch_sample_latents(model, train_data, train_size, batch_size=batch_size)
    assert train_loc.shape[0] == train_size
    assert train_y.shape[0] == train_size
    test_loc, test_y = batch_sample_latents(model, test_data, test_size, batch_size=batch_size)
    assert test_loc.shape[0] == test_size
    assert test_y.shape[0] == test_size


def predict_triplets(x_train, y_train, x_test, y_test, folds=5):
    num_factors = y_train.shape[1]
    num_codes = x_train.shape[1]
    train_loss = []
    test_loss = []
    for i in range(folds):
        model = GradientBoostingClassifier()
        model.fit(x_train, y_train[:, -1])
        train_loss.append(np.mean(model.predict(x_train) == y_train[:, -1]))
        test_loss.append(np.mean(model.predict(x_test) == y_test[:, -1]))
    return np.mean(train_loss), np.mean(test_loss)