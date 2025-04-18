#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 17:10:30 2022

@author: jz259
"""

iimport torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable

from loaddata import CyTOF_Dataset
from AE import Autoencoder, Attention_Layer, avg_pool
from gp_utils import DGPXRLModel, CutstomizedGaussianLikelihood

import os
import argparse
import numpy as np

torch.manual_seed(1)


def setup_args():

    options = argparse.ArgumentParser()

    # data directory
    options.add_argument('-datadir', '--data-dir', action="store", dest="data_dir", default='./HEUvsUE')
    options.add_argument('-fold', action="store", dest="fold", default = 1, type=int)

    # save and directory options
    options.add_argument('-sd', '--save-dir', action="store", dest="save_dir", default='./cytoGPNet_output')
    options.add_argument('--save-freq', action="store", dest="save_freq", default=10, type=int)
    #options.add_argument('--pretrained-file', action="store", dest="pretrained_file")

    # training parameters
    options.add_argument('-bs', '--batch-size', action="store", dest="batch_size", default=1, type=int)
    options.add_argument('-w', '--num-workers', action="store", dest="num_workers", default=10, type=int)
    options.add_argument('-lrAE', '--learning-rate-AE', action="store", dest="learning_rate_AE", default=1e-6, type=float)
    options.add_argument('-lrD', '--learning-rate-D', action="store", dest="learning_rate_D", default=1e-4, type=float)
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

def accuracy(output, target):
    pred = output.argmax(dim=1).view(-1)
    correct = pred.eq(target.view(-1)).float().sum().item()
    return correct

device = torch.device("cuda" if args.use_gpu else "cpu")

# ========== Load Test Data ==========
fold_path = os.path.join(args.data_dir, f"fold{args.fold}")
dataset = CyTOF_Dataset(datadir= fold_path, name="test_Data.obj", mode='test')
cyto_tensor = torch.from_numpy(dataset.data[1]).float()  # (n_samples, n_channels, n_cells, n_markers)
labels = torch.tensor(dataset.data[0]["label"].values).float()
patient_ids = dataset.data[0]["patient_id"].values

N, C, T, M = cyto_tensor.shape
cyto_concat = cyto_tensor.permute(0, 2, 1, 3).reshape(N, C * T, M)  # (n_samples, n_channels * n_cells, n_markers)

test_loader = DataLoader(TensorDataset(cyto_concat, labels), batch_size=args.batch_size, shuffle=False)

# ========== Load Trained Model Components ==========
autoencoder = simple_AE(input_dim=M, embed_dim=args.latent_dims, hidden_dim=args.hidden_dims).to(device)
autoencoder.load_state_dict(torch.load(os.path.join(args.save_dir, f"simpleAE_finetune_epoch{args.max_epochs}.pth")))
autoencoder.eval()

attention = Attention_Layer().to(device)
attention.load_state_dict(torch.load(os.path.join(args.save_dir, f"Attention_Layer_epoch{args.max_epochs}.pth")))
attention.eval()

classifier = Simple_Classifier(nz=1).to(device)
classifier.load_state_dict(torch.load(os.path.join(args.save_dir, f"Simple_Classifier_epoch{args.max_epochs}.pth")))
classifier.eval()

gp_layer = GaussianProcessLayer(
    input_dim=1,
    num_inducing_points=args.num_inducing_points,
    inducing_points=torch.randn(args.num_inducing_points, 1),  # dummy init
    mean_inducing_points=torch.randn(args.num_inducing_points, 1),
    grid_bounds=[(-1.0, 1.0)],
    likelihood_type='classification'
).to(device)
gp_layer.load_state_dict(torch.load(os.path.join(args.save_dir, f"GaussianProcessLayer_epoch{args.max_epochs}.pth")))
gp_layer.eval()

likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)
likelihood.load_state_dict(torch.load(os.path.join(args.save_dir, f"BernoulliLikelihood_epoch{args.max_epochs}.pth")))
likelihood.eval()

# ========== Inference ==========
predictions = []

with torch.no_grad():
    for i, (x_batch, y_batch) in enumerate(tqdm(test_loader, desc="Testing")):
        x_batch = x_batch.to(device)  # (B, T, M)

        z, _ = autoencoder(x_batch.reshape(-1, M))  # (B*T, 1)
        z = z.view(x_batch.size(0), -1, 1)  # (B, T, 1)

        output = gp_layer(z)
        f_samples = output.rsample().squeeze(-1)  # (B, T)

        pooled = attention(f_samples)  # (B,)
        preds = classifier(pooled.unsqueeze(-1)).squeeze(-1)  # (B,)

        for j in range(x_batch.size(0)):
            predictions.append({
                "patient_id": patient_ids[i * args.batch_size + j],
                "label": y_batch[j].item(),
                "prediction_score": preds[j].item()
            })

# ========== Save Prediction Results ==========
results_df = pd.DataFrame(predictions)
results_df.to_csv(os.path.join(args.save_dir, f"test_predictions{args.fold}.csv"), index=False)
print(f"Saved predictions to test_predictions{args.fold}.csv")

