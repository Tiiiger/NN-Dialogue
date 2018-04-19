import torch
from torch import Tensor, LongTensor
import torch.nn as nn
from torch.nn import Module
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from data import  OpenSub, pad_batch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tensorboardX import SummaryWriter
import os

def mle():
    raise NotImplementedError

def beam_search(beam_size):
    raise NotImplementedError

def mmi_anti_lm():
    raise NotImplementedError

def mmi_bidi():
    raise NotImplementedError

def evaluate():
    """
    TODO:
    1. get BLEU score
    """
    raise NotImplementedError

def sample_sentence():
    """
    1. Write sample sentence to disk.
    2. Save attention
    """
    raise NotImplementedError
