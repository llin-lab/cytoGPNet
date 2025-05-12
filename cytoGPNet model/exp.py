#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:08:02 2024

@author: jz259
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

class BBMPExp(nn.Module):
    def __init__(self, model, num_markers, device='cuda',
                 lam=1.0, gamma=0.1, lr=1e-2, num_iter=300, sample_num=10):
        """
        Args:
            model: full CytoGPNet model
            num_markers: number of marker dimensions (e.g., 27)
            device: torch device
            lam: sparsity regularization strength
            gamma: entropy regularization strength
            lr: learning rate for mask optimization
            num_iter: number of mask optimization steps
            sample_num: Monte Carlo samples per step
        """
        super().__init__()
        self.model = model.eval()
        self.num_markers = num_markers
        self.device = device
        self.lam = lam
        self.gamma = gamma
        self.lr = lr
        self.num_iter = num_iter
        self.sample_num = sample_num

    def explain(self, x, target_label):
        """
        Args:
            x: input tensor (1, seq_len, num_markers)
            target_label: int (0 or 1) — class to explain
        Returns:
            final_mask: tensor of shape (num_markers,)
        """
        x = x.to(self.device)  # [1, seq_len, M]
        seq_len = x.shape[1]

        # Learnable mask over marker dimensions (shared across sequence)
        log_alpha = nn.Parameter(torch.zeros(1, 1, self.num_markers, device=self.device))
        optimizer = torch.optim.Adam([log_alpha], lr=self.lr)

        for _ in range(self.num_iter):
            z = self.sample_concrete(log_alpha)  # [sample_num, 1, 1, M]
            z = z.expand(-1, 1, seq_len, -1)     # [sample_num, 1, seq_len, M]
            x_repeat = x.expand(self.sample_num, -1, -1)  # [sample_num, seq_len, M]
            x_masked = x_repeat * z.squeeze(1)  # [sample_num, seq_len, M]

            # Model forward pass
            y_pred = self.model(x_masked)       # [sample_num, 1]
            prob = y_pred.squeeze()             # [sample_num]
            if prob.dim() == 0:
                prob = prob.unsqueeze(0)
            if target_label == 0:
                prob = 1 - prob  # focus on p(class=0)

            # Regularization
            mask_prob = torch.sigmoid(log_alpha)
            sparsity_loss = mask_prob.mean()
            entropy_loss = -(mask_prob * torch.log(mask_prob + 1e-8) +
                             (1 - mask_prob) * torch.log(1 - mask_prob + 1e-8)).mean()

            loss = -prob.mean() + self.lam * sparsity_loss + self.gamma * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_mask = torch.sigmoid(log_alpha).detach().cpu().squeeze()  # [num_markers]
        return final_mask

    def sample_concrete(self, log_alpha, temp=0.1):
        """
        Sample a differentiable binary mask using Binary Concrete (Gumbel-Sigmoid)
        Returns:
            Tensor of shape [sample_num, 1, 1, num_markers]
        """
        u = torch.rand(self.sample_num, *log_alpha.shape, device=self.device)
        gumbel = -torch.log(-torch.log(u + 1e-8) + 1e-8)
        gate_inputs = (log_alpha + gumbel) / temp
        return torch.sigmoid(gate_inputs)