from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


# content loss
class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


# style loss
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    gram = G.div(a * b * c * d)
    return gram

class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        G = gram_matrix(input)
        A = gram_matrix(self.target)
        self.loss = F.mse_loss(G, A)
        return input