# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
from Ops import resblock_ll
from sub_unit_predictor import subunit_predictor
from expert_filters import expert_filters




class pose_regressor(nn.Module):
    def __init__(self, one_hot_size, label_dim, num_coeff, hidden_size, rng, Xdim, Ydim, keep_prob):
        super(pose_regressor, self).__init__()

        self.one_hot_size = one_hot_size
        self.Xdim = Xdim
        self.Ydim = Ydim
        # subunit predictor
        self.pose_subs = subunit_predictor(label_dim, num_coeff, hidden_size, rng, keep_prob)

        # expert filters
        self.filt0 = expert_filters([num_coeff, 2, Xdim / 2, 32], rng)
        self.filt1 = expert_filters([num_coeff, 2, 32, Xdim / 2], rng)

        #pose_regressor
        self.RL = nn.Linear(one_hot_size + num_coeff, num_coeff)
        self.hidden0 = resblock_ll([Xdim / 2, 32])
        self.hidden1 = resblock_ll([32, Xdim / 2])
        self.out_layer = nn.Linear(Xdim, Ydim)



    def forward(self, X, subs_input):
        # x = X.unsqueeze(-1)
        # x = x.reshape([1, 2, self.Xdim / 2])

        # pose subunits
        units = self.pose_subs(subs_input)

        # combination and down-projection
        bc_label = torch.cat((units.t(), subs_input[:, :self.one_hot_size]), 1)
        down_p = self.RL(bc_label)

        # pose regressor
        X = X.permute(0, 2, 1)
        H0 = self.hidden0(X, self.filt0.get_filt(down_p.t(), (self.Xdim, 32)))
        H1 = self.hidden1(H0, self.filt1.get_filt(down_p.t(), (32, self.Ydim)))
        H2 = torch.flatten(H1, start_dim=1, end_dim=-1)
        F3 = self.out_layer(H2)

        return F3



