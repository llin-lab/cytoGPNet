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


#============= TRAINING INITIALIZATION ==============

# load data, shape (3, 1, 10000, 24)
preData = CyTOF_Dataset(datadir="FCS_to_Array", name = "train_preData.obj", mode='train')
postData = CyTOF_Dataset(datadir="FCS_to_Array",name = "train_postData.obj", mode='train')
clinic_data = np.array(postData.data[0])[:,1].astype('float')
# TODO: target_data should be label and it should be either 0 or 1.
# target_data = np.array(postData.data[0])[:,1].astype('int') # I run into error at this line. Cannot convert str into int.I rewrote it with the correct syntex but the semantic seems wrong.
# np.array(postData.data[0])[:,1] is ['study1', 'study2', 'study3'], converting it into int is [1, 2, 3], not 0 or 1.
target_data = np.array([int(i[-1]) for i in np.array(postData.data[0])[:,2]])

p = preData.data[0].shape[1] - 2

preCyTOF_loader = torch.utils.data.DataLoader(torch.from_numpy(preData.data[1]).float(), batch_size=args.batch_size, drop_last=True, shuffle=False)
postCyTOF_loader = torch.utils.data.DataLoader(torch.from_numpy(postData.data[1]).float(), batch_size=args.batch_size, drop_last=True, shuffle=False)
clinic_loader = torch.utils.data.DataLoader(torch.from_numpy(clinic_data).float(), batch_size=args.batch_size, drop_last=True, shuffle=False)
target_laoder = torch.utils.data.DataLoader(torch.from_numpy(target_data).float(), batch_size=args.batch_size, drop_last=True, shuffle=False)

print("Data loaded.")


# initialize autoencoder
# TODO: maybe let netPre1 and netPost1 share parameters; let netPre2 and netPost2 share parameters (Do this later).
# TODO: since you add the pooling layer to the classifier, you can still use the reconstruction error for these encoders  (Do this later).

netAE = Autoencoder(n_input=preData.data[1].shape, nz=args.latent_dims) # common encoder
netAE.load_state_dict(torch.load(os.path.join(args.save_dir, "netAE_%s.pth" % args.max_epochs)))
netGP = DGPXRLModel(netAE, preData, postData)
netAtt1 = Attention_layer(n_input=preData.data[1])
netAtt2 = Attention_layer(n_input=postData.data[1])
netClf = Simple_Classifier(nz = p + 2 * args.latent_dims)

if args.use_gpu:
    netAE.cuda()
    netAtt1.cuda()
    netAtt2.cuda()
    netClf.cuda()

m = nn.Sequential(
    nn.AvgPool2d(kernel_size=(preData.data[1].shape[2], 1)),
    nn.Flatten(),
    nn.Flatten())

opt_netAE = optim.Adam(list(netAE.parameters()), lr=args.learning_rate_AE)
opt_netGP = optim.Adam(list(netAE.parameters()), lr=args.learning_rate_AE)
opt_netAtt1 = optim.Adam(list(netAtt1.parameters()), lr=args.learning_rate_AE)
opt_netAtt2 = optim.Adam(list(netAtt2.parameters()), lr=args.learning_rate_AE)
opt_netClf = optim.Adam(list(netClf.parameters()), lr=args.learning_rate_D, weight_decay=args.weight_decay)

print("Model Initialized.")

# loss criteria
criterion_discriminate= nn.MSELoss(reduction = 'none')
criterion_classify = nn.BCELoss()


# setup logger
with open(os.path.join(args.save_dir, 'log.txt'), 'w') as f:
    print(args, file=f)
    print(netAE, file=f)
    print(netAtt1, file=f)
    print(netAtt2, file=f)
    print(netClf, file=f)




def train_classifier(pre_inputs, post_inputs, clinic_inputs, targets):

    netAE.eval()
    netGP.train()
    netAtt1.train()
    netAtt2.train()
    netClf.train()

    # process input data
    pre_inputs, post_inputs, clinic_inputs, targets = Variable(pre_inputs), Variable(post_inputs), Variable(clinic_inputs), Variable(targets)

    if args.use_gpu:
        pre_inputs, post_inputs, clinic_inputs, targets = pre_inputs.cuda(), post_inputs.cuda(), clinic_inputs.cuda(), targets.cuda()

    # reset parameter gradients
    netGP.zero_grad()
    netAtt1.zero_grad()
    netAtt2.zero_grad()
    netClf.zero_grad()

    # forward pass
    pre_latent = netAE.encoder(pre_inputs)
    #pre_latent2 = netPre2(pre_inputs)
    post_latent = netAE.encoder(post_inputs)
    #post_latent2 = netPost2(post_inputs)
    latent = netGP(torch.concat([pre_latent, post_latent], dim = 1))
    pre_output = avg_pool(netAtt1(latent[:, :pre_inputs.size(1)]))
    post_output = avg_pool(netAtt2(latent[:, pre_inputs.size(1):post_inputs.size(1)]))
    d = torch.cat((pre_output, post_output, clinic_inputs), dim = 1)
    scores = netClf(d)


    # compute losses
    clf_loss = criterion_classify(scores, targets)

    loss = args.beta*clf_loss

    # backpropagate and update model
    loss.backward()
    opt_netClf.step()

    summary_stats = {'clf_loss': clf_loss, 'accuracy': accuracy(scores, targets)}

    return summary_stats


def train_general(pre_inputs, post_inputs, clinic_inputs, targets):
    netAE.train()
    netGP.train()
    netAtt1.train()
    netAtt2.train()
    netClf.train()

    # process input data
    pre_inputs, post_inputs, clinic_inputs, targets = Variable(pre_inputs), Variable(post_inputs), Variable(
        clinic_inputs), Variable(targets)

    if args.use_gpu:
        pre_inputs, post_inputs, clinic_inputs, targets = pre_inputs.cuda(), post_inputs.cuda(), clinic_inputs.cuda(), targets.cuda()

    # reset parameter gradients
    netGP.zero_grad()
    netAtt1.zero_grad()
    netAtt2.zero_grad()
    netClf.zero_grad()

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

    # compute losses
    clf_loss = criterion_classify(scores, targets)

    loss = args.beta * clf_loss

    # backpropagate and update model
    loss.backward()
    opt_netClf.step()

    summary_stats = {'clf_loss': clf_loss, 'accuracy': accuracy(scores, targets)}

    return summary_stats


def accuracy(y_prob, y_true):
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)


### main training loop
for epoch in range(args.max_epochs):
    print(epoch)

    recon_loss = 0
    clf_loss = 0
    

    for idx, (pre_inputs, post_inputs, clinic_inputs, targets) in enumerate(zip(preCyTOF_loader, postCyTOF_loader, clinic_loader, target_laoder)):
        out = train_general(pre_inputs, post_inputs, clinic_inputs, targets)
        clf_loss += out['clf_loss']
        
    with open(os.path.join(args.save_dir, 'log.txt'), 'a') as f:
        print('Epoch: ', epoch, ', disc loss 1: %.8f' % float(disc_loss1),
                ', clf loss: %.8f' % float(clf_loss), file=f)

    # save model
    if epoch % args.save_freq == 0:
        torch.save(netAE.cpu().state_dict(), os.path.join(args.save_dir,"netAE_%s.pth" % epoch))
        torch.save(netGP.cpu().state_dict(), os.path.join(args.save_dir,"netGP_%s.pth" % epoch))
        torch.save(netAtt1.cpu().state_dict(), os.path.join(args.save_dir,"netAtt1_%s.pth" % epoch))
        torch.save(netAtt2.cpu().state_dict(), os.path.join(args.save_dir,"netAtt2_%s.pth" % epoch))
        torch.save(netClf.cpu().state_dict(), os.path.join(args.save_dir,"netClf_%s.pth" % epoch))
    

    if args.use_gpu:
        netAE.cuda()
        netGP.cuda()
        netAtt1.cuda()
        netAtt2.cuda()
        netClf.cuda()
        
        
        
        
        

