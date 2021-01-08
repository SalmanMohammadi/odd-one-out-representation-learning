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
        self.z_dim = 256
        self.c1 = nn.Conv2d(n_channels, 32, kernel_size=4, stride=2)
        self.c2 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.c3 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.c4 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        # self.fc1 = nn.Linear()

    def forward(self, x):
        for c in ([self.c1, self.c2, self.c3, self.c4]):
            x = F.relu(c(x))
        return x

class RelationNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(2*embedding_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
    
    def forward(self, x):
        for fc in [self.fc1, self.fc2, self.fc3, self.fc4]:
            x = F.relu(fc(x))
        return x
    
class WReN(nn.Module):
    def __init__(self, n_channels=3, embedder=CNNEmbedder, cuda=True):
        super().__init__()
        if embedder == CNNEmbedder:
            self.embedder = embedder()
            self.embedding_size = self.embedder.z_dim
        elif embedder == 'values':
            self.embedder = 'values'
            self.embedding_size = 6
        else:
            self.embedder = embedder.batch_representation
            self.embedding_size = embedder.z_dim
        
        self.relation_net = RelationNetwork(self.embedding_size + 9)

        # scores embeddings
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=0.5)
        
        if cuda:
            self.cuda()

    def score_embeddings(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)

    def stack_context_answer_embeddings(self, context_embeddings, answer_embeddings):
        answer_blocks = []
        for i in range(answer_embeddings.shape[1]):
            ith_answer = answer_embeddings.select(-2, i).unsqueeze(-2)
            ith_answer_block = torch.cat([context_embeddings, ith_answer], axis=-2)
            answer_blocks.append(ith_answer_block)
        return torch.stack(answer_blocks, axis=-3)

    def panel_tags_like(self, stacked_panels):
        tags = torch.eye(stacked_panels.shape[-2]).to(CUDA)
        tags = tags.reshape((1, 1, 9, 9))
        mult = list(stacked_panels.shape)
        mult[-2] = 1
        mult[-1] = 1     
        return tags.repeat(mult)
    
    def pair_embeddings(self, tagged_embeddings):
        return torch.cat(
                (tagged_embeddings.unsqueeze(-2).repeat([1, 1, 1, 9, 1]),
                tagged_embeddings.unsqueeze(-3).repeat([1, 1, 9, 1, 1])),
            axis=-1)

    def forward(self, context, answers):
        # x - (batch_size, (context, answers), 3, 64, 64)
        # context - (batch_size, 8, 3, 64, 64)
        # answers - (batch_size, 6, 3, 64, 64)
        if self.embedder == 'values':
            context_embeddings = context
            answer_embeddings = answers
        else:
            context_embeddings = self.embedder(context).view(-1, self.embedding_size)
            answer_embeddings = self.embedder(answers).view(-1, self.embedding_size)
            
        context_embeddings = context_embeddings.reshape(-1, 8, self.embedding_size)
        answer_embeddings = answer_embeddings.reshape(-1, 6, self.embedding_size)
        stacked_panels = self.stack_context_answer_embeddings(context_embeddings, answer_embeddings)
        tagged_embeddings = torch.cat([stacked_panels, self.panel_tags_like(stacked_panels)], axis=-1)
        paired_embeddings = self.pair_embeddings(tagged_embeddings)

        # g_theta
        rn_paired_embeddings = self.relation_net(paired_embeddings)
        rn_paired_embeddings = rn_paired_embeddings.sum(-2).sum(-2)

        # f_theta
        scores = self.score_embeddings(rn_paired_embeddings).sum(-1)
        return scores

    def loss(self, scores, y):
        ce_loss = F.cross_entropy(scores, y, reduction='sum').div(scores.shape[0])
        return ce_loss

    def batch_forward(self, x, device):
        context, answers, y, context_values, answers_values = x
        # drop the last panel from the context
        if self.embedder == 'values':
            context = context_values.view(-1, 9, 6)[:, :-1].to(device)
            answers = answers_values.view(-1, 6, 6).to(device)
        else:
            context = context.view(-1, 9, 3, 64, 64)[:, :-1]
            context = context.reshape(-1, 3, 64, 64).to(device)
            answers = answers.view(-1, 3, 64, 64).to(device)
        y = y.to(device).squeeze()
        scores = self.forward(context, answers)
        return self.loss(scores, y), scores

def train_steps(model, dataset, val_dataset, optimizer, device=CUDA, 
                verbose=True, writer=None, log_interval=100, write_interval=1000,
                metrics_labels=None, eval_iter=1000):
    """
    Trains the model.
    """
    model.train()
    metrics_mean = []
    dataset_len = len(dataset)
    for batch_id, data in enumerate(dataset):
        optimizer.zero_grad()
        loss, _ =  model.batch_forward(data, device)
        loss.backward()
        optimizer.step()

        # train_loss += loss.item()
        # metrics_mean.append([x.item() for x in metrics])
        if batch_id % eval_iter == 0 or batch_id == dataset_len - 1:
            data = next(iter(val_dataset))
            (_, _, y, _, _) = data
            _ , scores = model.batch_forward(data, device)
            y = y.to(device)
            accuracy = (torch.argmax(scores, 1) == y).sum().item() / scores.shape[0]
            print("Step: {}, Accuracy: {}".format(batch_id, accuracy))
            if writer:
                writer.add_scalar('train/accuracy', accuracy, batch_id)
        if batch_id % log_interval == 0 and verbose:
            print('Train step: {}, loss: {}'.format(
                batch_id, loss.item()))
        if batch_id % write_interval == 0 and writer:
            writer.add_scalar('train/loss', loss.item(), batch_id)

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
