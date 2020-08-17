import numpy as np
import torch
#from torch.autograd import Variable

class expert_filters(object):
    def __init__(self, shape, rng):
        self.shape = shape
        self.rng = rng

        self.filter0 = torch.tensor(self.initial_alpha(shape), requires_grad=True)
        self.filter1 = torch.tensor(self.initial_alpha([shape[0], shape[1], shape[-1], shape[-1]]), requires_grad=True)

    def get_filt(self, classed, batch_size):
        a = torch.unsqueeze(self.filter0, 1)
        # a.expand(1, batch_size, 1, 1, 1)
        w = classed
        w = torch.unsqueeze(w, -1)
        w = torch.unsqueeze(w, -1)
        w = torch.unsqueeze(w, -1)

        r = torch.mul(w, a)
        f0 = torch.sum(r, dim=0)
        f0 = f0[0, :, :, :]
        # f0 = tf.slice(f0, [0, 0, 0, 0], [1, self.shape[1], self.shape[2], self.shape[3]])

        a = torch.unsqueeze(self.filter1, 1)
        # a.expand(1, batch_size, 1, 1, 1)
        w = classed
        w = torch.unsqueeze(w, -1)
        w = torch.unsqueeze(w, -1)
        w = torch.unsqueeze(w, -1)

        r = torch.mul(w, a)
        f1 = torch.sum(r, dim=0)
        f1 = f1[0, :, :, :]
        # f1 = tf.slice(f1, [0, 0, 0, 0], [1, self.shape[1], self.shape[-1], self.shape[-1]])
        f0 = f0.permute(2, 1, 0)
        f1 = f1.permute(2, 1, 0)
        return torch.squeeze(f0), torch.squeeze(f1)

    def initial_alpha_np(self, shape):
        rng = self.rng
        shape = [int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])]
        alpha_bound = np.sqrt(6. / np.prod(shape[-2:]))
        alpha = np.asarray(
            rng.uniform(low=-alpha_bound, high=alpha_bound, size=shape),
            dtype=np.float32)
        return alpha

    def initial_alpha(self, shape):
        alpha = self.initial_alpha_np(shape)
        return torch.from_numpy(alpha)
