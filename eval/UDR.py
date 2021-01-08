import models
from models.models_disentanglement import AdaGVAE, TVAE, AdaTVAE, VAE, TCVAE, KLTVAE, batch_sample_latents
from data import dsprites_data as dsprites
from data import rpm_data as rpm
from sklearn import linear_model
from sklearn import preprocessing
from data.dsprites_data import IterableDSPritesIID
from data.rpm_data import ColourDSprites, ColourDSpritesLoader
from sklearn import preprocessing
import numpy as np
import scipy
from six.moves import range
from sklearn.ensemble import GradientBoostingClassifier
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter 

CUDA = torch.device("cuda")

def relative_strength_disentanglement(corr_matrix):
    score_x = np.nanmean(
      np.nan_to_num(
          np.power(np.ndarray.max(corr_matrix, axis=0), 2) /
          np.sum(corr_matrix, axis=0), 0))
    score_y = np.nanmean(
      np.nan_to_num(
          np.power(np.ndarray.max(corr_matrix, axis=1), 2) /
          np.sum(corr_matrix, axis=1), 0))
    return (score_x + score_y) / 2

def lasso_correlation_matrix(vec1, vec2):
    model = linear_model.Lasso(alpha=0.1)
    model.fit(vec1, vec2)
    return np.transpose(np.absolute(model.coef_))

def _generate_representation_batch(data, models, batch_size=16):
    observations, _ = data.sample()
    observations = observations.to(CUDA).squeeze()
    
    res = [fn.batch_repr_kl(observations) for fn in models]
    return [(x.cpu().detach(), y.cpu().detach()) for (x, y) in res]
    
    
def generate_representation_dataset(models, data, batch_size=16, num_data_points=1000):
    representation_points = []
    kl_divergence = []

    for i in range(int(num_data_points / batch_size)):
        representation_batch = _generate_representation_batch(
            data, models, batch_size)
        
        for j in range(len(models)):
            # Initialize the outputs if it hasn't been created yet.
            if len(representation_points) <= j:
                kl_divergence.append(
                    np.zeros((int(num_data_points / batch_size),
                              representation_batch[j][1].shape[0])))
                representation_points.append(
                    np.zeros((num_data_points, representation_batch[j][0].shape[1])))
            kl_divergence[j][i, :] = representation_batch[j][1]
            representation_points[j][i * batch_size:(i + 1) * batch_size, :] = (
              representation_batch[j][0])
    return representation_points, [np.mean(kl, axis=0) for kl in kl_divergence]

def compute_udr(data, models):
    kl_filter_threshold=0.01
    inferred_model_reps, kl = generate_representation_dataset(models, data)
    num_models = len(inferred_model_reps)
    latent_dim = inferred_model_reps[0].shape[1]
    corr_matrix_all = np.zeros((num_models, num_models, latent_dim, latent_dim))
    kl_mask = []
    for i in range(len(inferred_model_reps)):
        scaler = preprocessing.StandardScaler()
        scaler.fit(inferred_model_reps[i])
        inferred_model_reps[i] = scaler.transform(inferred_model_reps[i])
        inferred_model_reps[i] = inferred_model_reps[i] * np.greater(kl[i], 0.01)
        kl_mask.append(kl[i] > kl_filter_threshold)
    disentanglement = np.zeros((num_models, num_models, 1))
    
    for i in range(num_models):
        for j in range(num_models):
            if i == j:
                continue
            corr_matrix = lasso_correlation_matrix(inferred_model_reps[i],
                                                   inferred_model_reps[j])
            
            corr_matrix = corr_matrix[kl_mask[i], ...][..., kl_mask[j]]
            disentanglement[i, j] = relative_strength_disentanglement(corr_matrix)

    scores_dict = {}
    scores_dict["pairwise_disentanglement_scores"] = disentanglement.tolist()
    
    
    model_scores = []
    for i in range(num_models):
        model_scores.append(np.median(np.delete(disentanglement[:, i], i)))

    scores_dict["model_scores"] = model_scores

    return scores_dict