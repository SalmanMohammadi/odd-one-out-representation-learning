import numpy as np
import scipy
from six.moves import range
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier
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

np.random.seed(100)

def calculate_triplet_score(model, train_size=25000, test_size=1, batch_size=16):
    train_data, test_data = rpm.get_dsprites(train_size=train_size, test_size=test_size, 
                                            dataset=ColourDSpritesTriplets, batch_size=batch_size, k=None)
    train_loc, train_y = batch_sample_latent_triplets(model, train_data, train_size, batch_size=batch_size)
    assert train_loc.shape[0] == train_size
    assert train_y.shape[0] == train_size
    train_acc, test_acc = predict_triplets(train_loc, train_y)
    scores = {}
    scores['mean_train_k'] = np.mean(train_acc)
    scores['std_train_k'] = np.std(train_acc)
    scores['mean_test_k'] = np.mean(test_acc)
    scores['std_test_k'] = np.std(test_acc)
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
        model =  GradientBoostingClassifier()
        model.fit(X_train, y_train[:, -1])
        pred = np.array(model.predict(X_train))
        pred_test = np.array(model.predict(X_test))
        train_loss.append(np.sum(pred == y_train[:, -1]) / X_train.shape[0])
        test_loss.append(np.sum(pred_test == y_test[:, -1]) / X_test.shape[0])
        # print(X_test.shape[0], X_train.shape[0])

    return train_loss, test_loss


if __name__ == '__main__':
    # vae = TVAE(n_channels=3)
    # path = "../tmp/tvae/log_p/0.pt"
    # vae.load_state_dict(torch.load(path))

    # vae = TVAE(n_channels=3)
    # path = "../tmp/tvae/no_log/0.pt"
    # vae.load_state_dict(torch.load(path))

    # vae = TVAE(n_channels=3)
    # path = "../tmp/tvae/p/0.pt"
    # vae.load_state_dict(torch.load(path))

    # vae = TVAE(n_channels=3)
    # path = "../tmp/tvae/abstract_reasoning/0.pt"
    # vae.load_state_dict(torch.load(path))
    # (0.8804833333333333, 0.00548568845066668, 0.8496666666666666, 0.010088497300281043)

    def eval_model(path, id):
        print(path)
        if "tvae" in path:
            vae = TVAE(n_channels=3)
        else:
            vae = VAE(n_channels=3)
        writer = SummaryWriter(log_dir=path)
        vae.load_state_dict(torch.load(path+".pt"))
        for label, metric in calculate_triplet_score(vae).items():
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
        for i in range(5):
            eval_model(path+str(i), i)
    # vae = TVAE(n_channels=3)
    # path = "../tmp/tvae/gamma=1+k=1/0.pt"
    # vae.load_state_dict(torch.load(path))
    # # (0.8322499999999999, 0.005524440645390664, 0.7926, 0.009534498763263151)
    # print(path)
    # print(calculate_triplet_score(vae))
    
# ./tmp/vae/abstract_reasoning/0.pt
# {'mean_train': 0.8270083333333333, 'std_train': 0.006882919680871891, 'mean_test': 0.8067, 'std_test': 0.006066666666666678}
# ./tmp/tvae/p/0.pt 
# {'mean_train': 0.8267749999999999, 'std_train': 0.007767659092530892, 'mean_test': 0.8090666666666667, 'std_test': 0.008284255481869741}
# ../tmp/tvae/no_log/0.pt
# {'mean_train': 0.817325, 'std_train': 0.007034360114624678, 'mean_test': 0.8002, 'std_test': 0.007671302945972643}
# ../tmp/tvae/log_p/0.pt
# {'mean_train': 0.8310000000000001, 'std_train': 0.008171724624174136, 'mean_test': 0.8134666666666666, 'std_test': 0.011443290125173295