import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from data import parse
from data import Loader

params = parse()
train_loader = Loader(params)
test_loader = Loader(params, "../Data/t_given_s_test.txt")

def masked_cross_entropy_loss():
    raise NotImplemented

class Encoder(nn.Module):

    def __init__(self, params):
        super(Encoder, self).__init__()
        raise NotImplemented

    def forward(self):
        raise NotImplemented

    def save_model(self, path):
        raise NotImplemented

class Decoder(nn.module):

    def __init__(self, params):
        super(Decoder, self).__init__()
        raise NotImplemented

    def forward(self):
        raise NotImplemented

def train():
    raise NotImplemented

def test():
    raise NotImplemented

def evaluate():
    raise NotImplemented

