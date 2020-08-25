import math
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.distributions as dist
from itertools import islice
from .models_disentanglement import TVAE
import numpy as np

CUDA = torch.device('cuda')

class CNNEmbedder(nn.Module):
    def __init__(self, n_channels=3):
        super().__init__()
        self.c1 = nn.Conv2d(n_channels, 32, kernel_size=4, stride=2)
        self.b1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.b2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.b3 = nn.BatchNorm2d(64)
        self.c4 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.b4 = nn.BatchNorm2d(64)
        # self.fc1 = nn.Linear()

    def forward(self, x):
        for c, b in zip([self.c1, self.c2, self.c3, self.c4],
                        [self.b1, self.b2, self.b3, self.b4]):
            x = F.relu(b(c(x)))
        return x

class RelationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
    
    def forward(self, x):
        for fc in [self.fc1, self.fc2, self.fc3, self.fc4]:
            x = F.relu(fc(x))
        return x
    
class WReN(nn.Module):
    def __init__(self, n_channels=3, embedding_dim=256, embedder=CNNEmbedder):
        super().__init__()
        self.embedder = embedder()

        self.relation_net = RelationNetwork()

        # scores embeddings
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=0.5)
    
    def score_embeddings(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)

    def pair_context_answer_embeddings(self, context_embeddings, answer_embeddings):
        return torch.cat((context_embeddings.unsqueeze(1).repeat(1,6,1,1), 
                    answer_embeddings.unsqueeze(2).repeat(1, 1, 8, 1)), -1)

    def forward(self, context, answers):
        # x - (batch_size, (context, answers), 3, 64, 64)
        # context - (batch_size, 8, 3, 64, 64)
        # answers - (batch_size, 6, 3, 64, 64)
        
        context_embeddings = self.embedder(context).view(-1, 256)
        answer_embeddings = self.embedder(answers).view(-1, 256)

        context_embeddings = context_embeddings.reshape(-1, 8, 256)
        answer_embeddings = answer_embeddings.reshape(-1, 6, 256)
        paired_embeddings = self.pair_context_answer_embeddings(context_embeddings, answer_embeddings)

        # g_theta
        rn_paired_embeddings = self.relation_net(paired_embeddings.view(-1, 512))
        rn_paired_embeddings = rn_paired_embeddings.view(-1, 6, 8, 512).sum(-2)

        # f_theta
        scores = self.score_embeddings(rn_paired_embeddings).squeeze()#.sum(-1)
        return scores

    def loss(self, scores, y):
        ce_loss = F.cross_entropy(scores, y, reduction='sum').div(scores.shape[0])
        accuracy = (torch.argmax(scores, 1) == y).sum().item() / scores.shape[0]

        return ce_loss, accuracy

    def batch_forward(self, x, device):
        context, answers, y = x

        # drop the last panel from the context
        context = context.view(-1, 9, 3, 64, 64)[:, :-1]
        context = context.reshape(-1, 3, 64, 64).to(device)
        answers = answers.view(-1, 3, 64, 64).to(device)
        y = y.to(device).squeeze()
        scores = self.forward(context, answers)
        loss, accuracy = self.loss(scores, y)
        return loss, accuracy

def train_steps(model, dataset, optimizer, device=CUDA, 
                verbose=True, writer=None, log_interval=100, write_interval=1000,
                metrics_labels=None):
    """
    Trains the model.
    """
    model.train()
    metrics_mean = []
    for batch_id, data in enumerate(dataset):
        optimizer.zero_grad()
        loss, (*metrics) = model.batch_forward(data, device)
        loss.backward()
        optimizer.step()

        # train_loss += loss.item()
        # metrics_mean.append([x.item() for x in metrics])
        if batch_id % log_interval == 0 and verbose:
            print('Train step: {}, loss: {}'.format(
                batch_id, loss.item()))
            if metrics_labels:
                print(", ".join(list(map(lambda x: "%s: %.5f" % x, zip(metrics_labels, metrics)))))
            else:
                print(metrics)
        if batch_id % write_interval == 0 and writer:
            writer.add_scalar('train/loss', loss.item(), batch_id)
            if metrics_labels:
                for label, metric in zip(metrics_labels, metrics):
                    writer.add_scalar('train/'+label, metric, batch_id)

def test(model, dataset, verbose=True, device=CUDA, 
        metrics_labels=None, writer=None, experiment_id=0):
    """
    Evaluates the model
    """
    model.eval()
    test_loss = 0
    metrics_mean = []
    with torch.no_grad():
        for batch_id, data in enumerate(dataset):
            loss, (*metrics) = model.batch_forward(data, device=device)
            metrics_mean.append([x for x in metrics])
            test_loss += loss.item()

    metrics_mean = np.array(metrics_mean)
    test_loss /= len(dataset.dataset)
    metrics_mean = np.sum(metrics_mean, axis=0)/len(dataset.dataset)
    # metrics = [x.item()/len(dataset.dataset) for x in metrics]
    if verbose:
        if metrics_labels:
            print(", ".join(list(map(lambda x: "%s: %.5f" % x, zip(metrics_labels, metrics_mean)))))
        print("Eval: ", test_loss)
    if writer:
        writer.add_scalar('test/loss', test_loss, experiment_id)
        for label, metric in zip(metrics_labels, metrics_mean):
            writer.add_scalar('test/'+label, metric, experiment_id)
    return test_loss, metrics_mean
