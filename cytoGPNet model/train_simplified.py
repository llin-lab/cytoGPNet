#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 04:29:48 2022

@author: jz259
"""

import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable

from loaddata import CyTOF_Dataset
from loadmodel import simple_AE, GaussianProcessLayer, Attention_Layer, Simple_Classifier

import os
import argparse
import numpy as np

torch.manual_seed(1)

def setup_args():

    options = argparse.ArgumentParser()

    # data directory
    options.add_argument('-datadir', '--data-dir', action="store", dest="data_dir", default='./HEUvsUE')

    # save and directory options
    options.add_argument('-sd', '--save-dir', action="store", dest="save_dir", default='./cytoGPNet_output')
    options.add_argument('--save-freq', action="store", dest="save_freq", default=10, type=int)
    #options.add_argument('--pretrained-file', action="store", dest="pretrained_file")

    # training parameters
    options.add_argument('-bs', '--batch-size', action="store", dest="batch_size", default=1, type=int)
    options.add_argument('-w', '--num-workers', action="store", dest="num_workers", default=10, type=int)
    options.add_argument('-lrAE', '--learning-rate-AE', action="store", dest="learning_rate_AE", default=1e-6, type=float)
    options.add_argument('-lrD', '--learning-rate-D', action="store", dest="learning_rate_D", default=1e-6, type=float)
    options.add_argument('-e', '--max-epochs', action="store", dest="max_epochs", default=100, type=int)
    options.add_argument('-wd', '--weight-decay', action="store", dest="weight_decay", default=0, type=float)


    # hyperparameters
    # options.add_argument('--alpha', action="store", default=1., type=float) # weight for classfication loss compared to discriminative loss
    options.add_argument('--hidden-dims', action="store", dest="hidden_dims", default=4, type=int)
    options.add_argument('--latent-dims', action="store", dest="latent_dims", default=2, type=int) # size of dimension for latent space of autoencoder
    options.add_argument('--num-inducing-points', action="store", dest="num_inducing_points", default=100, type=int) # size of dimension for latent space of autoencoder


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
dataset = CyTOF_Dataset(datadir=args.data_dir, name="train_Data.obj", mode='train')
cyto_tensor = torch.from_numpy(dataset.data[1]).float()  # (n_samples, n_channels, n_cells, n_markers)

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
    flat = cyto_concat[:args.num_inducing_points].reshape(-1, M).to(device)
    z_induce, _ = autoencoder(flat)
    z_induce = z_induce.view(-1, 1)

gp_layer = GaussianProcessLayer(
    input_dim=1,
    num_inducing_points=z_induce.size(0),
    inducing_points=z_induce.clone(),
    mean_inducing_points=z_induce.clone(),
    grid_bounds=[(z_induce.min().item(), z_induce.max().item())],
    likelihood_type='classification'
).to(device)

likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)
mll = gpytorch.mlls.VariationalELBO(likelihood, gp_layer, num_data=N)

# ----- Optimizer -----
optimizer = torch.optim.Adam(
    list(gp_layer.parameters()) + list(attention.parameters()) + list(classifier.parameters()),
    lr=args.learning_rate_D
)

# ----- Training Loop -----
for epoch in range(args.max_epochs):
    total_loss = 0
    gp_layer.train(); likelihood.train(); attention.train(); classifier.train(); autoencoder.train()

    for x_batch, y_batch in tqdm(data_loader, desc=f"Epoch {epoch+1}"):
        x_batch = x_batch.to(device)  # (B, T, M)
        y_batch = y_batch.to(device)

        # Encode cells via AE (no gradient)
        with torch.no_grad():
            z, _ = autoencoder(x_batch.reshape(-1, M))  # (B*T, 1)
        z = z.view(x_batch.size(0), -1, 1)  # (B, T, 1)

        # GP layer forward
        output = gp_layer(z)
        f_samples = output.rsample().squeeze(-1)  # (B, T)

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
    if epoch % args.save_freq == 0:
        torch.save(autoencoder.state_dict(), os.path.join(args.save_dir, f"simpleAE_finetune_epoch{epoch}.pth"))
        torch.save(gp_layer.state_dict(), os.path.join(args.save_dir, f"GaussianProcessLayer_epoch{epoch}.pth"))
        torch.save(likelihood.state_dict(), os.path.join(args.save_dir, f"BernoulliLikelihood_epoch{epoch}.pth"))
        torch.save(attention.state_dict(), os.path.join(args.save_dir, f"Attention_Layer_epoch{epoch}.pth"))
        torch.save(classifier.state_dict(), os.path.join(args.save_dir, f"Simple_Classifier_epoch{epoch}.pth"))
    print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(data_loader):.4f}")
