#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 04:29:48 2024

@author: jz259
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from torch.autograd import Variable

from loaddata import CyTOF_Dataset

from loadmodel import simple_AE

import os
import argparse
import numpy as np


torch.manual_seed(1)


def setup_args():

    options = argparse.ArgumentParser()

    # data directory
    options.add_argument('-datadir', '--data-dir', action="store", dest="data_dir", default='./HEUvsUE')

    # save and directory options
    options.add_argument('-sd', '--save-dir', action="store", dest="save_dir", default='./ae_output')
    options.add_argument('--save-freq', action="store", dest="save_freq", default=10, type=int)
    #options.add_argument('--pretrained-file', action="store")

    # training parameters
    options.add_argument('-bs', '--batch-size', action="store", dest="batch_size", default=128, type=int)
    options.add_argument('-w', '--num-workers', action="store", dest="num_workers", default=10, type=int)
    options.add_argument('-lrAE', '--learning-rate-AE', action="store", dest="learning_rate_AE", default=1e-6, type=float)
    options.add_argument('-lrD', '--learning-rate-D', action="store", dest="learning_rate_D", default=1e-4, type=float)
    options.add_argument('-e', '--max-epochs', action="store", dest="max_epochs", default=1000, type=int)
    options.add_argument('-wd', '--weight-decay', action="store", dest="weight_decay", default=0, type=float)


    # hyperparameters
    # options.add_argument('--alpha', action="store", default=1., type=float) # weight for classfication loss compared to discriminative loss
    options.add_argument('--hidden-dims', action="store", dest="hidden_dims", default=4, type=int)
    options.add_argument('--latent-dims', action="store", dest="latent_dims", default=2, type=int) # size of dimension for latent space of autoencoder

    # gpu options
    options.add_argument('-gpu', '--use-gpu', action="store_false", dest="use_gpu")

    return options.parse_args()


args = setup_args()
if not torch.cuda.is_available():
    args.use_gpu = False
device = torch.device("cuda" if args.use_gpu else "cpu")

os.makedirs(args.save_dir, exist_ok=True)




# --------- Load and process data ---------
dataset = CyTOF_Dataset(datadir=args.data_dir, name="train_Data.obj", mode='train')
cyto_tensor = torch.from_numpy(dataset.data[1]).float()  # (n_samples, n_channels, n_cells, n_markers)

n_samples, n_channels, n_cells, n_markers = cyto_tensor.shape

# Concatenate along the cell axis: (n_samples, n_channels * n_cells, n_markers)
cyto_concat = cyto_tensor.permute(0, 2, 1, 3).reshape(n_samples, n_channels * n_cells, n_markers)

# Flatten everything for AE training: (total_cells, n_markers)
cyto_flat = cyto_concat.reshape(-1, n_markers)

# Create a DataLoader for training
data_loader = DataLoader(cyto_flat, batch_size=args.batch_size, shuffle=True)

# --------- Initialize model, loss, optimizer ---------
ae_model = simple_AE(input_dim=n_markers, embed_dim=args.latent_dims)
criterion = nn.MSELoss()
optimizer = optim.Adam(ae_model.parameters(), lr=args.learning_rate_AE)

if args.use_gpu:
    ae_model.cuda()

# --------- Training loop ---------
os.makedirs(args.save_dir, exist_ok=True)

for epoch in range(args.max_epochs):
    ae_model.train()
    epoch_loss = 0

    for batch in data_loader:
        batch = Variable(batch)
        if args.use_gpu:
            batch = batch.cuda()

        optimizer.zero_grad()
        _, recon = ae_model(batch)
	# maybe store `encoding.detach().cpu().numpy()` for later
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()


        epoch_loss += loss.item() * batch.size(0)

    epoch_loss /= len(data_loader.dataset)
    print(f"Epoch {epoch:03d} - Recon Loss: {epoch_loss:.6f}")

    # Save model every frequency epochs
    if (epoch+1) % args.save_freq == 0:
        torch.save(ae_model.state_dict(), os.path.join(args.save_dir, f"simpleAE_epoch{epoch+1}.pth"))
