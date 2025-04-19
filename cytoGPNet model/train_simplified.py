#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 04:29:48 2022

@author: jz259
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable

from loaddata import CyTOF_Dataset
from loadmodel import simple_AE, GaussianProcessLayer, Attention_Layer, Simple_Classifier

import os
import argparse
import numpy as np
from tqdm import tqdm

import math
import torch.nn.functional as F
import warnings
import gpytorch
import seaborn as sns
from matplotlib import pyplot as plt
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.distributions import base_distributions, MultivariateNormal
from gpytorch.likelihoods.noise_models import HomoskedasticNoise
from gpytorch import settings
from gpytorch.utils.errors import CachingError
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.variational._variational_strategy import _VariationalStrategy
from gpytorch.variational.variational_strategy import _ensure_updated_strategy_flag_set
from gpytorch.utils.memoize import cached, clear_cache_hook, pop_from_cache_ignore_args
from gpytorch.lazy import DiagLazyTensor, MatmulLazyTensor, RootLazyTensor, SumLazyTensor, TriangularLazyTensor, delazify


torch.manual_seed(1)

def setup_args():

    options = argparse.ArgumentParser()

    # data directory
    options.add_argument('-datadir', '--data-dir', action="store", dest="data_dir", default='./HEUvsUE')
    options.add_argument('-fold', action="store", dest="fold", default = 1, type=int)


    # save and directory options
    options.add_argument('-sd', '--save-dir', action="store", dest="save_dir", default='./cytoGPNet_output')
    options.add_argument('--save-freq', action="store", dest="save_freq", default=10, type=int)
    options.add_argument('--pretrained-file', action="store", dest="pretrained_file")

    # training parameters
    options.add_argument('-bs', '--batch-size', action="store", dest="batch_size", default=1, type=int)
    options.add_argument('-w', '--num-workers', action="store", dest="num_workers", default=10, type=int)
    options.add_argument('-lrAE', '--learning-rate-AE', action="store", dest="learning_rate_AE", default=1e-6, type=float)
    options.add_argument('-lrD', '--learning-rate-D', action="store", dest="learning_rate_D", default=1e-3, type=float)
    options.add_argument('-e', '--max-epochs', action="store", dest="max_epochs", default=100, type=int)
    options.add_argument('-wd', '--weight-decay', action="store", dest="weight_decay", default=0, type=float)


    # hyperparameters
    # options.add_argument('--alpha', action="store", default=1., type=float) # weight for classfication loss compared to discriminative loss
    options.add_argument('--hidden-dims', action="store", dest="hidden_dims", default=4, type=int)
    options.add_argument('--latent-dims', action="store", dest="latent_dims", default=2, type=int) # size of dimension for latent space of autoencoder
    options.add_argument('--num-inducing-points', action="store", dest="num_inducing_points", default=500, type=int)


    # gpu options
    options.add_argument('-gpu', '--use-gpu', action="store_false", dest="use_gpu")

    return options.parse_args()


args = setup_args()
if not torch.cuda.is_available():
    args.use_gpu = False
device = torch.device("cuda" if args.use_gpu else "cpu")

os.makedirs(args.save_dir, exist_ok=True)


#============= TRAINING INITIALIZATION ==============

# ----- Load and preprocess data -----
fold_path = os.path.join(args.data_dir, f"fold{args.fold}")
dataset = CyTOF_Dataset(datadir= fold_path, name="train_Data.obj", mode='train')
cyto_tensor = torch.from_numpy(dataset.data[1]).float()  # (n_samples, n_channels = 1, n_cells, n_markers)

N, C, T, M = cyto_tensor.shape
cyto_concat = cyto_tensor.permute(0, 2, 1, 3).reshape(N, C * T, M)  # (n_samples, n_channels * n_cells, n_markers)
labels = torch.tensor(dataset.data[0]["label"].values).float()  # (n_samples,)


# Dataset
data_loader = DataLoader(TensorDataset(cyto_concat, labels), batch_size=args.batch_size, shuffle=True)


# ----- Load pretrained AE -----
ae_path = args.pretrained_file
autoencoder = simple_AE(input_dim=M, embed_dim=args.latent_dims, hidden_dim=args.hidden_dims).to(device)
autoencoder.load_state_dict(torch.load(ae_path))
autoencoder.train()  # Unfreeze for fine-tuning

# ----- Initialize attention + classifier -----
attention = Attention_Layer().to(device)
classifier = Simple_Classifier(nz=1).to(device)

# ----- Prepare GP Layer -----
# Sample a subset of encoded points to initialize inducing points
with torch.no_grad():
    flat = cyto_concat.reshape(-1, M)[:args.num_inducing_points, :].to(device)
    z_induce, _ = autoencoder(flat)
    


gp_layer = GaussianProcessLayer(
    input_dim=2,
    num_inducing_points=z_induce.size(0),
    inducing_points=z_induce.clone(),
    mean_inducing_points=z_induce.clone(),
    grid_bounds=[
    (z_induce[:, 0].min().item(), z_induce[:, 0].max().item()),
    (z_induce[:, 1].min().item(), z_induce[:, 1].max().item())],
    likelihood_type='classification', 
    using_ngd = True, 
    using_ksi = False, 
    using_ciq = False, 
    using_sor = False, 
    using_OrthogonallyDecouple = False, 
).to(device)

likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)
mll = gpytorch.mlls.VariationalELBO(likelihood, gp_layer, num_data=N)

# ----- Optimizer -----
optimizer = torch.optim.Adam([
    {'params': autoencoder.parameters(), 'lr': args.learning_rate_AE},  # slow learning for AE
    {'params': gp_layer.parameters(), 'lr': args.learning_rate_D},   # standard LR for GP
    {'params': attention.parameters(), 'lr': args.learning_rate_D},
    {'params': classifier.parameters(), 'lr': args.learning_rate_D},
])

# ----- Training Loop -----
for epoch in range(args.max_epochs):
    total_loss = 0
    gp_layer.train(); likelihood.train(); attention.train(); classifier.train(); autoencoder.train()

    for x_batch, y_batch in tqdm(data_loader, desc=f"Epoch {epoch+1}"):
        x_batch = x_batch.to(device)  # (B, T, M)
        y_batch = y_batch.to(device)

        # Encode cells via AE (no gradient)
        with torch.no_grad():
            z, _ = autoencoder(x_batch.reshape(-1, M))  # (B*T, 2)

        # GP layer forward
        output = gp_layer(z)
        f_samples = output.mean.view(-1, T)  # (B, T)

        # Attention -> pooled sample
        pooled = attention(f_samples)  # (B,)

        preds = classifier(pooled.unsqueeze(-1)).squeeze(-1)  # (B,)

        # Binary classification loss
        loss = F.binary_cross_entropy(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    # Save model every frequency epochs
    if (epoch+1) % args.save_freq == 0:
        torch.save(autoencoder.state_dict(), os.path.join(args.save_dir, f"simpleAE_finetune_epoch{epoch+1}.pth"))
        torch.save(gp_layer.state_dict(), os.path.join(args.save_dir, f"GaussianProcessLayer_epoch{epoch+1}.pth"))
        torch.save(likelihood.state_dict(), os.path.join(args.save_dir, f"BernoulliLikelihood_epoch{epoch+1}.pth"))
        torch.save(attention.state_dict(), os.path.join(args.save_dir, f"Attention_Layer_epoch{epoch+1}.pth"))
        torch.save(classifier.state_dict(), os.path.join(args.save_dir, f"Simple_Classifier_epoch{epoch+1}.pth"))
    print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(data_loader):.4f}")
