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
    options.add_argument('-lrAE', '--learning-rate-AE', action="store", dest="learning_rate_AE", default=1e-6, type=float)
    options.add_argument('-lrD', '--learning-rate-D', action="store", dest="learning_rate_D", default=1e-6, type=float)
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
#netPre2 = Autoencoder(n_input=preData.data[1].shape, nz=args.latent_dims) # specific encoder
netAtt = Attention_Layer() # common encoder
#netPost2 = Autoencoder(n_input=postData.data[1].shape, nz=args.latent_dims) # specific encoder
netClf = Simple_Classifier(nz = p + 2 * args.latent_dims)

if args.use_gpu:
    netAE.cuda()
    #netAtt.cuda()
    #netClf.cuda()

#m = nn.Sequential(
#   nn.AvgPool2d(kernel_size=(preData.data[1].shape[2], 1)),
#    nn.Flatten(),
#    nn.Flatten())

opt_netAE = optim.Adam(list(netAE.parameters()), lr=args.learning_rate_AE)

print("Model Initialized.")

# loss criteria
criterion_reconstruction = nn.MSELoss(reduction = 'none')


# setup logger
with open(os.path.join(args.save_dir, 'log.txt'), 'w') as f:
    print(args, file=f)
    print(netAE, file=f)



def train_autoencoders(pre_inputs, post_inputs):
    
    netAE.train()
    

    
    # process input data
    pre_inputs, post_inputs = Variable(pre_inputs), Variable(post_inputs)

    if args.use_gpu:
        pre_inputs, post_inputs = pre_inputs.cuda(), post_inputs.cuda()

    # reset parameter gradients
    netAE.zero_grad()

    
    # forward pass
    pre_recon = netAE(pre_inputs)
    post_recon = netAE(post_inputs)

    

    
    # compute losses

    loss = criterion_reconstruction(pre_recon, pre_inputs) + criterion_reconstruction(post_recon, post_inputs)


    # backpropagate and update model
    loss.backward()
    opt_netAE.step()


    
    summary_stats = {'recon_loss1': loss.item()}
    
    
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
        out = train_autoencoders(pre_inputs, post_inputs, clinic_inputs, targets)
        recon_loss += out['recon_loss']

    with open(os.path.join(args.save_dir, 'log.txt'), 'a') as f:
        print('Epoch: ', epoch, ', recon loss : %.8f' % float(recon_loss), file=f)

    # save model
    if epoch % args.save_freq == 0:
        torch.save(netAE.cpu().state_dict(), os.path.join(args.save_dir,"netAE_%s.pth" % epoch))

    

    if args.use_gpu:
        netAE.cuda()

        
        
        

