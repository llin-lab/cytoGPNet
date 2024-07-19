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

    # save and directory options
    options.add_argument('-sd', '--save-dir', action="store", dest="save_dir", default='./eval')
    options.add_argument('--save-freq', action="store", dest="save_freq", default=20, type=int)
    #options.add_argument('--pretrained-file', action="store")

    # training parameters
    options.add_argument('-bs', '--batch-size', action="store", dest="batch_size", default=1, type=int)
    options.add_argument('-w', '--num-workers', action="store", dest="num_workers", default=10, type=int)
    options.add_argument('-lrAE', '--learning-rate-AE', action="store", dest="learning_rate_AE", default=1e-4, type=float)
    options.add_argument('-lrD', '--learning-rate-D', action="store", dest="learning_rate_D", default=1e-4, type=float)
    options.add_argument('-e', '--max-epochs', action="store", dest="max_epochs", default=1001, type=int)
    options.add_argument('-wd', '--weight-decay', action="store", dest="weight_decay", default=0, type=float)

    # hyperparameters
    options.add_argument('--alpha', action="store", default=1., type=float) # weight for classfication loss compared to discriminative loss
    options.add_argument('--latent-dims', action="store", default=4, type=int) # size of dimension for latent space of autoencoder

    # gpu options
    options.add_argument('-gpu', '--use-gpu', action="store_false", dest="use_gpu")

    return options.parse_args()

args = setup_args()
if not torch.cuda.is_available():
    args.use_gpu = False

os.makedirs(args.save_dir, exist_ok=True)

def accuracy(output, target):
    pred = output.argmax(dim=1).view(-1)
    correct = pred.eq(target.view(-1)).float().sum().item()
    return correct


# load data
preData = CyTOF_Dataset(datadir="FCS_to_Array", name = "train_preData.obj", mode='train')
postData = CyTOF_Dataset(datadir="FCS_to_Array",name = "train_postData.obj", mode='train')
clinic_data = np.array(postData.data[0])[:,1].astype('float')
# TODO: target_data should be label and it should be either 0 or 1.
# target_data = np.array(postData.data[0])[:,1].astype('int') # I run into error at this line. Cannot convert str into int.I rewrote it with the correct syntex but the semantic seems wrong.
# np.array(postData.data[0])[:,1] is ['study1', 'study2', 'study3'], converting it into int is [1, 2, 3], not 0 or 1.
target_data = np.array(postData.data[0])[:,2].astype('float')

p = preData.data[0].shape[1] - 2


pre_inputs, post_inputs, clinic_inputs, targets = torch.from_numpy(preData.data[1]).float(), torch.from_numpy(postData.data[1]).float(), torch.from_numpy(clinic_data).float(), torch.from_numpy(target_data).float()


print("Train Data loaded.")

# load additional model
m = nn.Sequential(
    nn.AvgPool2d(kernel_size=(preData.data[1].shape[2], 1)),
    nn.Flatten(),
    nn.Flatten())

print('model initialized')



pred = {}

for epoch in args.max_epochs:
    if epoch % args.save_freq == 0:
        print(epoch)
        netAE = Autoencoder(n_input=preData.data[1].shape, nz=args.latent_dims)  # common encoder
        netGP = DGPXRLModel(netAE, preData, postData)
        netAtt1 = Attention_layer(n_input=preData.data[1])
        netAtt2 = Attention_layer(n_input=postData.data[1])
        netClf = Simple_Classifier(nz=p + 2 * args.latent_dims)
        
        
        netAE.load_state_dict(torch.load(os.path.join(args.save_dir, "netAE_%s.pth" % epoch)))
        netGP.load_state_dict(torch.load(os.path.join(args.save_dir, "netGP_%s.pth" % epoch)))
        netAtt1.load_state_dict(torch.load(os.path.join(args.save_dir, "netAtt1_%s.pth" % epoch)))
        netAtt2.load_state_dict(torch.load(os.path.join(args.save_dir, "netAtt2_%s.pth" % epoch)))
        netClf.load_state_dict(torch.load(os.path.join(args.save_dir, "netClf_%s.pth" % epoch)))
        
        netAE.eval()
        netGP.eval()
        netAtt1.eval()
        netAtt2.eval()
        netClf.eval()
        
        
        if args.use_gpu:
            netAE.cuda()
            netGP.cuda()
            netAtt1.cuda()
            netAtt2.cuda()
            netClf.cuda()
        
        # forward pass
        pre_latent = netAE.encoder(pre_inputs)
        # pre_latent2 = netPre2(pre_inputs)
        post_latent = netAE.encoder(post_inputs)
        # post_latent2 = netPost2(post_inputs)
        latent = netGP(torch.concat([pre_latent, post_latent], dim=1))
        pre_output = avg_pool(netAtt1(latent[:, :pre_inputs.size(1)]))
        post_output = avg_pool(netAtt2(latent[:, pre_inputs.size(1):post_inputs.size(1)]))
        d = torch.cat((pre_output, post_output, clinic_inputs), dim=1)
        scores = netClf(d)
        
        pred["epoch" + str(epoch)] = scores.cpu().detach().numpy().reshape(preData.data[1].shape[0])


pred["target"] = target_data.reshape(preData.data[1].shape[0])

df = pd.DataFrame.from_dict(pred)
df.to_csv(os.path.join(args.save_dir, "train_result.csv"))


# load data
preData = CyTOF_Dataset(datadir="FCS_to_Array", name = "test_preData.obj", mode='test')
postData = CyTOF_Dataset(datadir="FCS_to_Array",name = "test_postData.obj", mode='test')
clinic_data = np.array(postData.data[0])[:,1].astype('float')
# TODO: target_data should be label and it should be either 0 or 1.
# target_data = np.array(postData.data[0])[:,1].astype('int') # I run into error at this line. Cannot convert str into int.I rewrote it with the correct syntex but the semantic seems wrong.
# np.array(postData.data[0])[:,1] is ['study1', 'study2', 'study3'], converting it into int is [1, 2, 3], not 0 or 1.
target_data = np.array(postData.data[0])[:,2].astype('float')

p = preData.data[0].shape[1] - 2

preCyTOF_loader = torch.utils.data.DataLoader(torch.from_numpy(preData.data[1]).float(), batch_size=args.batch_size, drop_last=True, shuffle=False)
postCyTOF_loader = torch.utils.data.DataLoader(torch.from_numpy(postData.data[1]).float(), batch_size=args.batch_size, drop_last=True, shuffle=False)
clinic_loader = torch.utils.data.DataLoader(torch.from_numpy(clinic_data).float(), batch_size=args.batch_size, drop_last=True, shuffle=False)
target_laoder = torch.utils.data.DataLoader(torch.from_numpy(target_data).float(), batch_size=args.batch_size, drop_last=True, shuffle=False)

pre_inputs, post_inputs, clinic_inputs, targets = torch.from_numpy(preData.data[1]).float(), torch.from_numpy(postData.data[1]).float(), torch.from_numpy(clinic_data).float(), torch.from_numpy(target_data).float()


print("Test Data loaded.")

# load additional model
m = nn.Sequential(
    nn.AvgPool2d(kernel_size=(preData.data[1].shape[2], 1)),
    nn.Flatten(),
    nn.Flatten())

print('model initialized')

pred = {}

for epoch in args.max_epochs:
    if epoch % args.save_freq == 0:
        print(epoch)
        netAE = Autoencoder(n_input=preData.data[1].shape, nz=args.latent_dims)  # common encoder
        netGP = DGPXRLModel(netAE, preData, postData)
        netAtt1 = Attention_layer(n_input=preData.data[1])
        netAtt2 = Attention_layer(n_input=postData.data[1])
        netClf = Simple_Classifier(nz=p + 2 * args.latent_dims)

        netAE.load_state_dict(torch.load(os.path.join(args.save_dir, "netAE_%s.pth" % epoch)))
        netGP.load_state_dict(torch.load(os.path.join(args.save_dir, "netGP_%s.pth" % epoch)))
        netAtt1.load_state_dict(torch.load(os.path.join(args.save_dir, "netAtt1_%s.pth" % epoch)))
        netAtt2.load_state_dict(torch.load(os.path.join(args.save_dir, "netAtt2_%s.pth" % epoch)))
        netClf.load_state_dict(torch.load(os.path.join(args.save_dir, "netClf_%s.pth" % epoch)))

        netAE.eval()
        netGP.eval()
        netAtt1.eval()
        netAtt2.eval()
        netClf.eval()

        if args.use_gpu:
            netAE.cuda()
            netGP.cuda()
            netAtt1.cuda()
            netAtt2.cuda()
            netClf.cuda()

        # forward pass
        pre_latent = netAE.encoder(pre_inputs)
        # pre_latent2 = netPre2(pre_inputs)
        post_latent = netAE.encoder(post_inputs)
        # post_latent2 = netPost2(post_inputs)
        latent = netGP(torch.concat([pre_latent, post_latent], dim=1))
        pre_output = avg_pool(netAtt1(latent[:, :pre_inputs.size(1)]))
        post_output = avg_pool(netAtt2(latent[:, pre_inputs.size(1):post_inputs.size(1)]))
        d = torch.cat((pre_output, post_output, clinic_inputs), dim=1)
        scores = netClf(d)

        pred["epoch" + str(epoch)] = scores.cpu().detach().numpy().reshape(preData.data[1].shape[0])

pred["target"] = target_data.reshape(preData.data[1].shape[0])

df = pd.DataFrame.from_dict(pred)
df.to_csv(os.path.join(args.save_dir, "test_result.csv"))




