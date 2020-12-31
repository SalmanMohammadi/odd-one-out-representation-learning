import numpy as np
import scipy
import torch.nn as nn
import torch.nn.functional as F
from six.moves import range
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter 
from torch.utils.data import random_split
from torch import Generator
import sys
sys.path.append('../')
import models
from models.models_disentanglement import TVAE, VAE, TCVAE, TVAE, AdaGVAE, batch_sample_latent_triplets
from data import rpm_data as rpm
from data.rpm_data import ColourDSpritesTriplets
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

CUDA = torch.device('cuda')
np.random.seed(100)

class Network(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim*3, z_dim*3)
        self.fc2 = nn.Linear(z_dim*3, int((z_dim*3)/2))
        self.fc3 = nn.Linear(int((z_dim*3)/2), 3)
        self.cuda()

    def forward(self, x):
        for l in [self.fc1, self.fc2, self.fc3]:
            x = nn.functional.relu(l(x))
        return x

    def score(self, x, y):
        pred = self.predict(x)
        return np.sum(pred == y) / x.shape[0]

    def predict(self, x):
        self.eval()
        x = torch.tensor(x, dtype=torch.float32)
        # dat = TensorDataset(x)
        data = DataLoader(x, batch_size=512, shuffle=False)
        res = []
        for batch_id, x in enumerate(data):
            x = x.to(CUDA)
            y_ = self.forward(x)
            pred = torch.max(nn.functional.softmax(y_), dim=1, keepdim=True)[1]
            res.append(np.array(pred.detach().cpu()))

        res = np.concatenate(res).ravel()
        return res

    def fit(self, x_train, y_train, epochs=30):
        self.train()
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        dat = TensorDataset(x_train, y_train)
        opt = optim.Adam(self.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        data = DataLoader(dat, batch_size=16, shuffle=True)
        
        for i in range(epochs):
            print("----EPOCH----", i)
            for batch_id, batch in enumerate(data):
                opt.zero_grad()
                x, y = batch
                x = x.to(CUDA).squeeze()
                y = y.to(CUDA).squeeze()
                y_ = self(x)
                print(y_)
                loss = nn.functional.cross_entropy(y_, y, reduction='mean')
                loss.backward()
                opt.step()
                print("Batch:", batch_id, ", Loss:", loss.item())



def calculate_triplet_score(model, train_size=10000, test_size=5000, batch_size=16, dataset="dsprites", seed=0):
    print("Dataset", dataset)
    if dataset == "dsprites":
        train_data, test_data = rpm.get_dsprites(train_size=train_size, test_size=test_size, 
                                            dataset=ColourDSpritesTriplets, batch_size=batch_size, k=None)
    else:
        train_data, test_data = random_split(dataset, [train_size, test_size], generator=Generator().manual_seed(seed))
        train_data = DataLoader(train_data, batch_size=batch_size)
        test_data = DataLoader(test_data, batch_size=batch_size)
    train_loc, train_y = batch_sample_latent_triplets(model, train_data, train_size, batch_size=batch_size)
    assert train_loc.shape[0] == train_size
    assert train_y.shape[0] == train_size
    test_loc, test_y = batch_sample_latent_triplets(model, test_data, test_size, batch_size=batch_size)
    assert test_loc.shape[0] == test_size
    assert test_y.shape[0] == test_size
    train_acc, test_acc = predict_triplets(train_loc, train_y, test_loc, test_y)
    scores = {}
    scores['mean_train_k'] = train_acc
    # scores['std_train_k'] = np.std(train_acc)
    scores['triplet_10k'] = test_acc
    # scores['std_test_k'] = np.std(test_acc)
    return scores


def predict_triplets(X_train, y_train, X_test, y_test, folds=5):
    y_train = y_train.numpy()
    y_test = y_test.numpy()
    train_loss = []
    test_loss = []
    model =  GradientBoostingClassifier()
    # model = Network(64)
    # print(model)
    model.fit(X_train, y_train[:, -1])
    train_loss = model.score(X_train, y_train[:, -1])
    test_loss = model.score(X_test, y_test[:, -1])

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

    def eval_model(path, new_path, id):
        print(path)
        if "tvae" in path:
            vae = TVAE(n_channels=3)
        elif "tcvae" in path:
            vae = TCVAE(n_channels=3)
        elif "adagvae" in path:
            vae = AdaGVAE(n_channels=3)
        else:
            vae = VAE(n_channels=3)
        # writer = SummaryWriter(log_dir=new_path)
        vae.load_state_dict(torch.load(path+".pt"))
        res = calculate_triplet_score(vae)
        for label, metric in res.items():
            print(label, ":", metric)
            # writer.add_scalar(label, metric, id)
        return res
        # writer.flush()
        # writer.close()

    base_path = "../tmp_paper/"
    new_path = "../tmp_triplet/"
    paths =[
        # "tcvae/b=2/",
        # "tcvae/b=6/",
        # "tcvae/b=16/",
        # "tvae/gamma=1+k=rnd/",
        # "tvae/gamma=6+k=rnd/",
        # "tvae/gamma=16+k=rnd/",
        # "adagvae/b=1/",
        # "adagvae/b=6/",
        # "adagvae/b=16/",
        "vae/b=1/",
        "vae/b=6/",
        "vae/b=16/",
    ]
    scores = []
    for path in paths:
        for i in range(5):
            scores.append(eval_model(base_path + path+str(i), new_path+path+str(i), i)["triplet_10k"])

    print("Median", np.median(scores))
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