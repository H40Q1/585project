from __future__ import print_function
import torch
import torch.nn as nn


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach()
        self.std = std.clone().detach()

    def forward(self, img):
        # normalize img
        self.mean = torch.tensor(self.mean).view(-1, 1, 1)
        self.std = torch.tensor(self.std).view(-1, 1, 1)
        normalized_img = (img - self.mean) / self.std
        return normalized_img