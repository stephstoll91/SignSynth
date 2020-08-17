import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class resblock_ll(nn.Module):
    def __init__(self, dims):
        super(resblock_ll, self).__init__()
        self.input_dim = dims[0]
        self.output_dim = dims[1]
        self.padding = nn.ConstantPad1d((0, 1), 0)
        self.res_conv = nn.Conv1d(int(self.input_dim), int(self.output_dim), 1)

    def forward(self, x, filters):
        filt0 = filters[0]
        filt1 = filters[-1]
        res = self.res_conv(x)
        resblock0 = F.conv1d(self.padding(x), filt0)
        resblock1 = F.elu(resblock0)
        resblock2 = F.conv1d(self.padding(resblock1), filt1)
        return F.elu(resblock2 + res)
