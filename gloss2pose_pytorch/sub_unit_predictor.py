import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd import Variable

class subunit_predictor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, rng, keep_prob):
        super(subunit_predictor, self).__init__()
        #self.input = input
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.rng = rng
        self.keep_prob = keep_prob

        self.w0 = torch.tensor(self.initial_weight([self.hidden_size, self.input_size]), requires_grad=True)
        self.w1 = torch.tensor(self.initial_weight([self.hidden_size, self.hidden_size]), requires_grad=True)
        self.w2 = torch.tensor(self.initial_weight([self.output_size, self.hidden_size]), requires_grad=True)

        self.b0 = torch.tensor(self.initial_bias([self.hidden_size, 1]), requires_grad=True)
        self.b1 = torch.tensor(self.initial_bias([self.hidden_size, 1]), requires_grad=True)
        self.b2 = torch.tensor(self.initial_bias([self.output_size, 1]), requires_grad=True)

        #self.classed = self.fp()

    def forward(self, input_x):
        #H0 = input_x.permute(0, 1)
        H1 = torch.matmul(self.w0, input_x.t()) + self.b0
        H1 = F.elu(H1)
        H1 = F.dropout(H1, p=self.keep_prob)

        H2 = torch.matmul(self.w1, H1) + self.b1
        H2 = F.elu(H2)
        H2 = F.dropout(H2, p=self.keep_prob)

        H3 = torch.matmul(self.w2, H2) + self.b2
        H3 = F.softmax(H3, dim=0)

        return H3

    def initial_weight(self, shape):
        rng = self.rng
        weight_bound = np.sqrt(6. / np.sum(shape[-2:]))
        weight = np.asarray(
            rng.uniform(low=-weight_bound, high=weight_bound, size=shape),
            dtype=np.float32)
        return torch.from_numpy(weight)

    def initial_bias(self, shape):
        return torch.zeros(shape, dtype=torch.float32)
